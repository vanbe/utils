#!/usr/bin/env python3
"""
pdf_vision_ocr.py — Faithful OCR of a PDF using a local vision-capable model.

Hybrid approach: each page is processed with BOTH a text extract (PyMuPDF)
and a rendered image (vision model). The text extract provides correct words
(no hallucination); the image provides reading order and structure for
multi-column layouts, headings, tables, and figure descriptions.

Output is written page-by-page in append mode — safe to interrupt and resume.
If the output file already exists, already-processed pages are skipped.

Engine configuration in .env:
  AI_ENGINE=lmstudio           # or ollama
  LM_STUDIO_HOST=http://localhost:1234
  LM_STUDIO_MODEL=your_model_id_here
  OLLAMA_HOST=http://localhost:11434
  OLLAMA_MODEL=your_model_name:tag

Dependencies:
  pip install openai pymupdf

Usage:
  python pdf_vision_ocr.py document.pdf
  python pdf_vision_ocr.py document.pdf -o extracted.md
  python pdf_vision_ocr.py document.pdf --pages 1-5
  python pdf_vision_ocr.py document.pdf --dpi 300          # dense/small text
  python pdf_vision_ocr.py document.pdf --lang French
  python pdf_vision_ocr.py document.pdf --prompt "Pay special attention to equations"
"""

import sys
import os
import re
import base64
import argparse
import time
from collections import Counter
from datetime import datetime, timedelta

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import create_backend, resolve_host  # noqa: E402
from ai_common import (                          # noqa: E402
    load_env, parse_num_gpu, get_model_info, ensure_model_loaded,
    step as _step, ok as _ok, elapsed as _elapsed, SEP as _SEP,
)

_DEFAULT_DPI        = 200
_PAGE_OUTPUT_TOKENS = 4096   # generous per-page budget to avoid truncation

# Vision OCR needs room for: system prompt (~300 tok) + image (~500–1500 tok)
# + text extract (~500 tok) + output (up to 4096 tok) → 8 192 is the minimum
# that reliably fits a full page.  The text-pipeline env var OLLAMA_MODEL_A_CTX
# is intentionally NOT read here — its 4096 value is correct for small text
# chunks but would silently truncate vision input and produce empty pages.
_OCR_DEFAULT_CTX    = 8192


# ---------------------------------------------------------------------------
# Page range parsing
# ---------------------------------------------------------------------------

def _parse_pages(spec: str, total: int) -> list[int]:
    """Parse "1-3,5,7-9" → sorted list of 0-based indices."""
    indices: set[int] = set()
    for part in spec.split(','):
        part = part.strip()
        m = re.fullmatch(r'(\d+)-(\d+)', part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            for p in range(lo, hi + 1):
                if 1 <= p <= total:
                    indices.add(p - 1)
        elif re.fullmatch(r'\d+', part):
            p = int(part)
            if 1 <= p <= total:
                indices.add(p - 1)
        else:
            print(f"Warning: ignoring unrecognised page spec '{part}'.")
    return sorted(indices)


# ---------------------------------------------------------------------------
# PDF → PNG (in memory, lossless)
# ---------------------------------------------------------------------------

def _render_page_png(page, dpi: int) -> bytes:
    zoom   = dpi / 72.0
    matrix = __import__('fitz').Matrix(zoom, zoom)
    pix    = page.get_pixmap(matrix=matrix, alpha=False)
    return pix.tobytes('png')


def _png_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode('ascii')
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Text extraction (PyMuPDF) — column-aware reading order
# ---------------------------------------------------------------------------

def _extract_page_text(page) -> str:
    """
    Extract raw text from the page in approximate reading order.

    Special layout elements are detected and tagged with Markdown hints:
    - Sidebar / inset boxes (found via filled or bordered rectangles in the
      PDF drawing layer): rendered as blockquote lines (> text).
    - Pull quotes (blocks significantly larger than body font, near full-width):
      rendered as "> *text*" blockquote-italics.
    - Two-column layouts: left column before right column.
    - Multi-column sidebars: sorted left-to-right within y-bands.

    Returns empty string when no text layer exists (scanned / image-only PDFs).
    """
    # -- Font-size data (needed for pull-quote detection) --------------------
    page_dict = page.get_text("dict")
    all_char_sizes: list = []
    block_sizes: dict    = {}   # (x0_rounded, y0_rounded) → dominant font size

    for db in page_dict.get('blocks', []):
        if db.get('type') != 0:
            continue
        bb   = db['bbox']
        key  = (round(bb[0]), round(bb[1]))
        char_sizes_here: list = []
        for line in db.get('lines', []):
            for span in line.get('spans', []):
                n = len(span.get('text', '').strip())
                if n:
                    char_sizes_here.extend([span['size']] * n)
        if char_sizes_here:
            block_sizes[key] = Counter(char_sizes_here).most_common(1)[0][0]
            all_char_sizes.extend(char_sizes_here)

    body_size = Counter(all_char_sizes).most_common(1)[0][0] if all_char_sizes else 10.0

    def _block_font_size(b) -> float:
        return block_sizes.get((round(b[0]), round(b[1])), body_size)

    # -- Simple blocks for layout -------------------------------------------
    blocks      = page.get_text("blocks")   # (x0,y0,x1,y1,text,block_no,type)
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    if not text_blocks:
        return ''

    pw = page.rect.width
    ph = page.rect.height

    # -- Detect sidebar regions (filled or bordered rectangles) -------------
    min_area  = pw * ph * 0.03   # at least 3 % of page area
    min_width = pw * 0.35        # at least 35 % of page width
    sidebar_rects = []
    for draw in page.get_drawings():
        r = draw.get('rect')
        if r is None:
            continue
        w = r.x1 - r.x0
        h = r.y1 - r.y0
        if w < min_width or w * h < min_area:
            continue
        fill  = draw.get('fill')
        color = draw.get('color')
        lw    = draw.get('width') or 0
        has_colored_fill = fill is not None and fill not in ((1.0, 1.0, 1.0), (1, 1, 1))
        has_border       = color is not None and lw > 0.3
        if has_colored_fill or has_border:
            sidebar_rects.append(r)

    # -- Classify each block ------------------------------------------------
    def _in_sidebar(b):
        cx = (b[0] + b[2]) / 2
        cy = (b[1] + b[3]) / 2
        return any(r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1 for r in sidebar_rects)

    def _is_pullquote(b):
        # Near-full-width block with significantly larger font than body text
        if b[2] - b[0] <= pw * 0.45:
            return False
        return _block_font_size(b) >= body_size * 1.45

    def _classify_col(b):
        if b[2] - b[0] > pw * 0.65:
            return 'full'
        return 'left' if (b[0] + b[2]) / 2 < pw / 2 else 'right'

    sidebar_blks  = [b for b in text_blocks if _in_sidebar(b)]
    remaining     = [b for b in text_blocks if not _in_sidebar(b)]

    # -- Extend sidebar downward: absorb blocks that belong to the same sidebar
    # but lie BELOW the detected rectangle (common when the PDF has a coloured
    # title bar detected as a drawing, but the white body area below it is not).
    # Also absorbs diagram labels that are spatially contained within the sidebar.
    if sidebar_blks:
        sb_x0 = min(b[0] for b in sidebar_blks)
        sb_x1 = max(b[2] for b in sidebar_blks)
        sb_y1 = max(b[3] for b in sidebar_blks)

        growing = True
        while growing:
            growing = False
            new_remaining = []
            for b in remaining:
                # Block must be spatially CONTAINED within the sidebar x-bounds
                # (with a 25 px margin) — this ensures right-column text is
                # never accidentally absorbed even when it sits alongside the sidebar.
                x_contained = b[0] >= sb_x0 - 25 and b[2] <= sb_x1 + 25
                # Block must be directly below (within 55 px) the current sidebar bottom
                y_adjacent  = b[1] <= sb_y1 + 55
                if x_contained and y_adjacent:
                    sidebar_blks.append(b)
                    sb_y1  = max(sb_y1, b[3])
                    growing = True
                else:
                    new_remaining.append(b)
            remaining = new_remaining

    pullquote_blks = sorted([b for b in remaining if _is_pullquote(b)], key=lambda b: b[1])
    main_blks      = [b for b in remaining if not _is_pullquote(b)]

    # -- Format sidebar as blockquote ---------------------------------------
    parts = []

    if sidebar_blks:
        # Sort by y-band (15 px) then x → handles multi-column sidebars
        sb_sorted = sorted(sidebar_blks, key=lambda b: (int(b[1] / 15) * 15, b[0]))

        # Heuristic: topmost block is the title when it reads like a heading.
        # We flatten newlines before measuring length so multi-line headings
        # (e.g. long ALL-CAPS titles that wrap) are handled correctly.
        title_block = None
        if sb_sorted:
            first_text_flat = ' '.join(sb_sorted[0][4].split())   # collapse whitespace
            first_size      = _block_font_size(sb_sorted[0])
            # Title if: notably larger font than body, OR short enough to be a heading
            if first_size >= body_size * 1.05 or len(first_text_flat) < 100:
                title_block = sb_sorted[0]

        sb_lines = []
        for b in sb_sorted:
            if b is title_block:
                # Collapse multi-line title to a single bolded line
                title_text = ' '.join(b[4].split())
                sb_lines.append(f"> **{title_text}**")
            else:
                for line in b[4].strip().split('\n'):
                    line = line.strip()
                    if line:
                        sb_lines.append(f"> {line}")
                sb_lines.append(">")   # blank separator between blocks
        while sb_lines and sb_lines[-1] == ">":
            sb_lines.pop()
        if sb_lines:
            # Wrap in horizontal rules so the insert is unmistakably distinct
            parts.append('---\n' + '\n'.join(sb_lines) + '\n---')

    # -- Order main blocks (column-aware) -----------------------------------
    left   = sorted([b for b in main_blks if _classify_col(b) == 'left'],  key=lambda b: b[1])
    right  = sorted([b for b in main_blks if _classify_col(b) == 'right'], key=lambda b: b[1])
    full_m = sorted([b for b in main_blks if _classify_col(b) == 'full'],  key=lambda b: b[1])

    if left and right:
        col_top    = min(b[1] for b in left + right)
        col_bottom = max(b[3] for b in left + right)
        pre  = [b for b in full_m if b[3] <= col_top]
        post = [b for b in full_m if b[1] >= col_bottom]
        mid  = [b for b in full_m if b not in pre and b not in post]
        ordered = pre + left + mid + right + post
    else:
        ordered = sorted(main_blks, key=lambda b: b[1])

    # -- Interleave pull quotes at their vertical position ------------------
    main_segments = []
    pq_idx = 0
    for b in ordered:
        while pq_idx < len(pullquote_blks) and pullquote_blks[pq_idx][1] < b[1]:
            pq = pullquote_blks[pq_idx]
            main_segments.append(f"> *{pq[4].strip()}*")
            pq_idx += 1
        main_segments.append(b[4].strip())
    while pq_idx < len(pullquote_blks):
        main_segments.append(f"> *{pullquote_blks[pq_idx][4].strip()}*")
        pq_idx += 1

    main_text = '\n\n'.join(s for s in main_segments if s.strip())
    if main_text:
        parts.append(main_text)

    return '\n\n'.join(parts)


# ---------------------------------------------------------------------------
# OCR system prompts
# ---------------------------------------------------------------------------

# Hybrid mode: used when a text layer is available.
# The model uses the image for reading order / structure, and the text for word accuracy.
_OCR_SYSTEM_HYBRID = (
    "You are a faithful document transcription assistant. "
    "You receive a PAGE IMAGE and a RAW TEXT EXTRACT from the same page.\n\n"
    "The raw text extract contains the correct words but may have:\n"
    "- Column-ordering issues (multi-column layouts interleaved instead of column-by-column)\n"
    "- Hyphenation artifacts from line breaks (e.g. 'busi-\\nness')\n"
    "- No heading or table formatting\n"
    "- Pre-formatted Markdown blockquote lines (> ...) marking sidebars and pull quotes\n\n"
    "Your task:\n"
    "1. Use the IMAGE to determine the correct reading order (especially for multi-column pages) "
    "and identify the document structure: headings, lists, tables, figures, sidebars.\n"
    "2. Use the RAW TEXT as your word-level source — transcribe words exactly as they appear "
    "in the extract. Do not invent, change, omit, or paraphrase any content.\n"
    "   The extract uses Markdown layout hints that you must honour:\n"
    "   • '> **Title**' followed by '> lines' → a sidebar or inset box. "
    "Reproduce as a Markdown blockquote block with the bold title.\n"
    "   • '> *text*' → a pull quote. Reproduce as a Markdown blockquote in italics.\n"
    "   Verify their position and extent against the IMAGE.\n"
    "3. Format the output in Markdown: # ## ### for headings, | for tables, - / 1. for lists. "
    "Use > blockquotes for sidebar/inset boxes and pull quotes as indicated above.\n"
    "4. Rejoin words hyphenated at line breaks: 'busi-\\nness' → 'business'.\n"
    "5. For ALL visual elements — figures, charts, diagrams, photos, decorative illustrations, "
    "historical engravings, artistic images — write a description in square brackets where "
    "it appears in the flow:\n"
    "   [Figure: detailed description — include title, labels, values, colours, axes, "
    "overall subject matter, and visual style (e.g. 'woodcut-style black-and-white engraving "
    "of workers in a factory')]\n"
    "   Never silently skip any visual element, even if it appears purely decorative.\n"
    "   When a sidebar section in the extract contains scattered short fragments (individual "
    "words or short phrases that look like diagram node labels), describe the whole diagram "
    "as a [Figure: ...] using the image rather than listing the labels individually.\n"
    "6. Transcribe mathematical formulas in LaTeX ($...$ inline, $$...$$ block).\n"
    "7. Omit repeated page headers and footers (running titles, page numbers) "
    "if they add no unique content.\n"
    "8. Do not add commentary, invented footnotes, transitional phrases, or any text "
    "not visible on this page.\n"
    "9. If the page is entirely blank: output exactly: [Blank page]\n"
    "10. CRITICAL — stop the moment you have transcribed all real content. "
    "Never repeat a sentence or paragraph. If you notice yourself writing the same "
    "text twice, stop immediately. Do not pad or fill with invented content."
)

# Vision-only mode: fallback for scanned PDFs with no text layer.
_OCR_SYSTEM_VISION = (
    "You are a faithful document transcription assistant. "
    "Reproduce the full content of this page exactly as a human would write it out by hand.\n\n"
    "Rules:\n"
    "1. Transcribe ALL text verbatim — every word, every sentence. "
    "Do not skip, shorten, or paraphrase anything.\n"
    "2. Preserve document structure in Markdown: # ## ### for headings, "
    "| for tables, - or 1. for lists.\n"
    "3. For ALL visual elements — figures, charts, diagrams, photos, decorative illustrations, "
    "historical engravings, artistic images — insert a description in square brackets:\n"
    "   [Figure: detailed description including title, labels, values, colours, visual style "
    "(e.g. 'woodcut-style black-and-white engraving of workers')]\n"
    "   Never silently skip any visual element, even if it appears purely decorative.\n"
    "4. Transcribe mathematical formulas in LaTeX ($...$ or $$...$$).\n"
    "5. Footnotes: transcribe at page bottom, prefixed with their reference marker.\n"
    "6. Omit repeated page headers and footers on subsequent pages.\n"
    "7. Decorative elements with no informational content: skip silently.\n"
    "8. If the page is entirely blank: output exactly: [Blank page]\n"
    "9. Do not add commentary, preamble, or explanation — output only the transcribed content.\n"
    "10. CRITICAL — stop the moment you have transcribed all real content. "
    "Never repeat a sentence or paragraph. If you notice yourself writing the same "
    "text twice, stop immediately."
)


def _ocr_page(
    client,
    model: str,
    png_bytes: bytes,
    text_hint: str,
    extra_prompt: str,
    lang: str,
    output_tokens: int,
    extra_body: dict = None,
    tuned_prompt: str = '',
) -> str:
    if tuned_prompt:
        # Specialised OCR model (e.g. DeepSeek-OCR) — use the model-specific
        # trigger phrase verbatim. Skip the system prompt and the
        # text-hint / lang / extra_prompt augmentations entirely; those
        # confuse purpose-built OCR models trained on a fixed instruction.
        user_text = tuned_prompt.replace('\\n', '\n')
        messages = [{
            "role": "user",
            "content": [
                {"type": "text",      "text": user_text},
                {"type": "image_url", "image_url": {"url": _png_to_data_url(png_bytes)}},
            ],
        }]
    else:
        if text_hint:
            system   = _OCR_SYSTEM_HYBRID
            if lang:
                system += f"\n10. The document language is {lang}. Preserve it exactly."
            user_text = "Transcribe this page faithfully using the image and the raw text extract below."
            if extra_prompt:
                user_text += f" Additional focus: {extra_prompt}"
            user_text += f"\n\nRAW TEXT EXTRACT:\n---\n{text_hint}\n---"
        else:
            system   = _OCR_SYSTEM_VISION
            if lang:
                system += f"\n10. The document language is {lang}. Preserve it exactly."
            user_text = "Transcribe this document page fully and faithfully."
            if extra_prompt:
                user_text += f" Additional focus: {extra_prompt}"

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": user_text},
                    {"type": "image_url", "image_url": {"url": _png_to_data_url(png_bytes)}},
                ],
            },
        ]

    kwargs: dict = dict(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=output_tokens,
        frequency_penalty=0.15,
    )
    if extra_body:
        kwargs['extra_body'] = extra_body

    for attempt in range(1, 4):   # up to 3 attempts
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ''
        except Exception as exc:
            if attempt == 3:
                raise
            wait = 15 * attempt   # 15 s, then 30 s before the final attempt
            print(f"\n    timeout/error (attempt {attempt}/3): {exc.__class__.__name__} — "
                  f"retrying in {wait}s …", end='', flush=True)
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Resume support — detect already-processed pages
# ---------------------------------------------------------------------------

def _done_pages(out_path: str) -> set[int]:
    """Return set of 1-based page numbers already written to the output file."""
    if not os.path.isfile(out_path):
        return set()
    done: set[int] = set()
    with open(out_path, encoding='utf-8') as f:
        for line in f:
            m = re.match(r'<!--\s*page\s+(\d+)\s*-->', line)
            if m:
                done.add(int(m.group(1)))
    return done


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _eta_str(t_start: float, done: int, total: int) -> str:
    """Return 'ETA: Xm Ys  ·  ~HH:MM', or '' when all pages are done."""
    if done >= total:
        return ''
    avg       = (time.time() - t_start) / done
    remaining = avg * (total - done)
    finish    = datetime.now() + timedelta(seconds=remaining)
    rem_str   = (f"{int(remaining // 60)}m {int(remaining % 60):02d}s"
                 if remaining >= 60 else f"{remaining:.0f}s")
    return f"ETA: {rem_str}  ·  ~{finish.strftime('%H:%M')}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_env()

    _engine_env   = os.environ.get('AI_ENGINE',            'lmstudio').lower()
    _lm_host_env  = os.environ.get('LM_STUDIO_HOST',       'http://localhost:1234')
    _lm_ocr_model = os.environ.get('LM_STUDIO_OCR_MODEL',  '')
    _lm_ocr_ctx   = int(os.environ.get('LM_STUDIO_OCR_MODEL_CTX', '0'))
    _ol_host_env  = os.environ.get('OLLAMA_HOST',          'http://localhost:11434')
    _ol_ocr_model = os.environ.get('OLLAMA_OCR_MODEL',     '')
    _ol_ocr_ctx   = int(os.environ.get('OLLAMA_OCR_MODEL_CTX', '0'))
    _ol_ocr_gpu   = parse_num_gpu(os.environ.get('OLLAMA_OCR_NUM_GPU', ''))
    _ocr_tuned_prompt = os.environ.get('OCR_TUNED_PROMPT', '').strip()

    parser = argparse.ArgumentParser(
        description='Faithful OCR of a PDF — verbatim text + figure descriptions, page by page.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('pdf_file', help='Input PDF file')
    parser.add_argument('-o', '--output',
                        help='Output Markdown file (default: <name>_ocr.md)')
    parser.add_argument('--pages', default='', metavar='RANGE',
                        help='Pages to process, e.g. "1-3,5,8-10" (default: all)')
    parser.add_argument('--dpi', type=int, default=_DEFAULT_DPI,
                        help=f'Rendering DPI (default: {_DEFAULT_DPI}; use 300 for dense/small text)')
    parser.add_argument('--lang', default='English',
                        help='Document language (default: English)')
    parser.add_argument('--prompt', '-p', default='',
                        help='Extra focus instructions appended to every page request')
    parser.add_argument('--engine', default=_engine_env, choices=['lmstudio', 'ollama'],
                        help='AI engine (default: from AI_ENGINE env var)')
    parser.add_argument('--host', default='', metavar='URL',
                        help='API base URL (default: from engine env vars)')
    parser.add_argument('--model', default='', metavar='ID',
                        help='Override the OCR model (default: OLLAMA_OCR_MODEL / LM_STUDIO_OCR_MODEL)')
    parser.add_argument('--context-size', type=int, default=0, metavar='N',
                        help='Override context window size in tokens')
    parser.add_argument('--output-tokens', type=int, default=_PAGE_OUTPUT_TOKENS, metavar='N',
                        help=f'Max output tokens per page (default: {_PAGE_OUTPUT_TOKENS})')
    parser.add_argument('--vision-only', action='store_true',
                        help='Skip text extraction — use vision model only (slower, for scanned PDFs)')
    args = parser.parse_args()

    if args.engine == 'ollama':
        host_default  = args.host or _ol_host_env
        model_default = args.model or _ol_ocr_model
        ctx_env       = _ol_ocr_ctx
        num_gpu       = _ol_ocr_gpu
    else:
        host_default  = args.host or _lm_host_env
        model_default = args.model or _lm_ocr_model
        ctx_env       = _lm_ocr_ctx
        num_gpu       = None   # LMStudio always uses max GPU offload

    if not model_default:
        env_key = 'OLLAMA_OCR_MODEL' if args.engine == 'ollama' else 'LM_STUDIO_OCR_MODEL'
        print(f"No OCR model configured. Set {env_key} in .env or use --model.")
        sys.exit(1)

    pdf_path = os.path.abspath(args.pdf_file)
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    try:
        import fitz  # noqa: F401
    except ImportError:
        print("pymupdf not installed. Run: pip install pymupdf")
        sys.exit(1)

    try:
        from openai import OpenAI  # noqa: F401
    except ImportError:
        print("openai not installed. Run: pip install openai")
        sys.exit(1)

    resolved_host = resolve_host(host_default)
    backend = create_backend(args.engine, resolved_host)
    client  = backend.create_client()

    model, context_window = get_model_info(backend, model_default)
    if ctx_env:
        # OLLAMA_OCR_MODEL_CTX / LM_STUDIO_OCR_MODEL_CTX explicitly set
        context_window = ctx_env
    elif context_window > _OCR_DEFAULT_CTX:
        # Model architecture default too large — cap to safe OCR minimum
        print(f"  ctx capped: {context_window} → {_OCR_DEFAULT_CTX}"
              f"  (vision OCR minimum — set OLLAMA_OCR_MODEL_CTX or --context-size to override)")
        context_window = _OCR_DEFAULT_CTX
    if args.context_size:  # --context-size CLI flag takes final precedence
        context_window = args.context_size

    # Determine output path
    out_path = (os.path.abspath(args.output) if args.output
                else os.path.splitext(pdf_path)[0] + '_ocr.md')

    # Open PDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # Resolve page selection
    if args.pages:
        page_indices = _parse_pages(args.pages, total_pages)
        if not page_indices:
            print("No valid pages in the given range.")
            sys.exit(1)
    else:
        page_indices = list(range(total_pages))

    # Resume: detect already-processed pages
    already_done = _done_pages(out_path)
    to_process   = [i for i in page_indices if (i + 1) not in already_done]
    skipped      = len(page_indices) - len(to_process)

    # ── Header ──────────────────────────────────────────────────────────────
    mode_label = 'vision only' if args.vision_only else 'text + vision'
    print(_SEP)
    print(f"  {os.path.basename(pdf_path)}")
    print(f"  engine: {args.engine}  ·  model: {model}")
    print(f"  pages: {len(page_indices)}  ·  DPI: {args.dpi}"
          f"  ·  lang: {args.lang}  ·  {mode_label}")
    if _ocr_tuned_prompt:
        print(f"  tuned-OCR prompt: {_ocr_tuned_prompt[:60]}"
              f"{'…' if len(_ocr_tuned_prompt) > 60 else ''}")
    if skipped:
        print(f"  resuming — {skipped} page(s) already done, {len(to_process)} to process")
    print(_SEP)

    if not to_process:
        print("  All pages already processed.")
        print(f"\n  Output: {os.path.basename(out_path)}")
        doc.close()
        return

    t_total = time.time()

    # ── [0] Model loading ────────────────────────────────────────────────────
    ensure_model_loaded(backend, model, context_window, num_gpu=num_gpu)

    # Write document title on first run (fresh file)
    if not already_done:
        doc_title = os.path.splitext(os.path.basename(pdf_path))[0]
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"# {doc_title}\n\n")

    # ── Process pages ────────────────────────────────────────────────────────
    t_pages = time.time()

    for n_done, page_idx in enumerate(to_process):
        page_num  = page_idx + 1
        seq_label = f"[{n_done + 1}/{len(to_process)}]  Page {page_num}"
        print(f"  {seq_label:<38}", end='', flush=True)
        t_page = time.time()

        page      = doc.load_page(page_idx)
        png_bytes = _render_page_png(page, args.dpi)
        text_hint = '' if args.vision_only else _extract_page_text(page)
        text      = _ocr_page(
            client, model,
            png_bytes    = png_bytes,
            text_hint    = text_hint,
            extra_prompt = args.prompt,
            lang         = args.lang,
            output_tokens= args.output_tokens,
            extra_body   = backend.get_inference_extra_body(),
            tuned_prompt = _ocr_tuned_prompt,
        )

        # Write this page immediately — crash-safe
        with open(out_path, 'a', encoding='utf-8') as f:
            f.write(f"<!-- page {page_num} -->\n\n")
            f.write(text.strip())
            f.write("\n\n")

        elapsed = _elapsed(t_page)
        eta     = _eta_str(t_pages, n_done + 1, len(to_process))
        suffix  = f"   {eta}" if eta else ''
        print(f"  ✓  {elapsed}{suffix}")

    doc.close()

    # ── [N] Unload model ─────────────────────────────────────────────────────
    t = _step("Unload model")
    backend.unload(model)
    _ok(t, "unloaded")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print(_SEP)
    print(f"  Total:  {_elapsed(t_total)}")
    print(f"  Pages:  {len(to_process)} processed")
    print(f"  Saved:  {os.path.basename(out_path)}")
    print(_SEP)


if __name__ == '__main__':
    main()
