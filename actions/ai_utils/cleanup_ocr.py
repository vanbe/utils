#!/usr/bin/env python3
"""
cleanup_ocr.py — Restore reading flow + fix OCR artefacts in a Markdown file.

This is a *content-preserving* transformation: every sentence, fact, and
detail of the source survives. Only the OCR artefacts get repaired:

    - Spurious spaces inside words ("busi ness" → "business")
    - Soft-hyphen + space breaks at column edges ("trade - offs" → "trade-offs")
    - Words split across page/column boundaries
    - Citations / pull quotes / sidebars formatted as # headings
    - Paragraphs split by an inserted sidebar — sidebar moved out, paragraph rejoined
    - Output kept Obsidian-friendly (ATX headings, > blockquotes, GFM tables)

Why a separate script (rather than another transform_md.py template):
the existing pipeline is built around *compression* (Map → Reduce → Refine
with a hard target word count). Cleanup is editing — there is nothing to
reduce. Trying to fit cleanup into that pipeline produces either silent
truncation (Reduce iterates until the doc fits) or wrong-length validation
loops. A simple "send doc, get cleaned doc back" loop is the right shape.

Modes:
  - Single-pass when the doc + cleaned-doc + prompt fits in Model B's context.
  - Chunked otherwise: paragraph-aware split via transform_pipeline.build_chunks,
    each chunk cleaned independently, results concatenated. Cross-chunk word
    splits are best-effort (chunks fall on paragraph boundaries to minimise risk).

.env configuration (re-uses Model B from transform_md.py):
  AI_ENGINE=lmstudio              # or 'ollama'
  LM_STUDIO_HOST=http://localhost:1234
  LM_STUDIO_MODEL_B=<model id>
  LM_STUDIO_MODEL_B_CTX=52500
  OLLAMA_HOST=http://localhost:11434
  OLLAMA_MODEL_B=model:tag
  OLLAMA_MODEL_B_CTX=8192

Usage:
  python cleanup_ocr.py document.md
  python cleanup_ocr.py document.md --lang French
  python cleanup_ocr.py document.md --no-model-swap
  python cleanup_ocr.py document.md -o cleaned.md
"""

import argparse
import os
import sys
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import create_backend, resolve_host                # noqa: E402
from ai_common import (                                        # noqa: E402
    load_env, parse_num_gpu, ensure_model_loaded, get_model_info,
    step as _step, ok as _ok, elapsed as _elapsed, SEP as _SEP,
)
from transform_pipeline import (                               # noqa: E402
    parse_md_blocks, build_chunks, tok, tok_to_words,
    detect_language, infer, log,
)


# ---------------------------------------------------------------------------
# System prompt — the heart of this feature
# ---------------------------------------------------------------------------

_CLEANUP_SYSTEM = """You are an OCR text restorer. You receive a Markdown document produced by an OCR pipeline applied to a magazine-style PDF. Your job: emit a corrected version that reads as the source did, with ALL content preserved word-for-word — only the artefacts fixed.

CORRECT THESE OCR ARTEFACTS:

1. Spurious spaces inside words — common at column edges:
   "busi ness" → "business", "syst em" → "system", "manag ers" → "managers",
   "compe titive" → "competitive", "capi tal ist" → "capitalist".

2. Soft-hyphen + space breaks at column boundaries:
   "compe - titive" → "competitive", "prob - lems" → "problems",
   "trade - offs" → "trade-offs" (KEEP the hyphen for genuine compound words like
   "end-to-end", "long-term", "trade-offs").

3. Words split across page or column boundaries:
   "...connec-" followed by "tions between..." → "...connections between..."
   Use context to decide whether the joined word is correct.

4. Orphan letters at line / page starts:
   "Tbusiness" or "T business" at the start of a paragraph likely belongs to
   a hidden "The" — restore from context only when obvious. If unsure, leave alone.

5. Citations / pull-quotes / sidebars formatted as headings:
   Lines starting with #, ##, ### that are actually short quoted text, an
   attribution, or sidebar content must become Obsidian callouts (see rule 6
   for the exact syntax). Genuine section/chapter headings (e.g. "Moving
   Beyond Trade-Offs", "Reconceiving Products and Markets") MUST remain
   headings — they are NEVER inserts.

6. Citations, pull-quotes, sidebars, and inserted boxes — MOVE them; never
   drop, never duplicate, never swallow surrounding structure.

   What IS an insert (move + wrap as callout):
     - A short pull-quote in a different style ("...we must reconnect business
       and society.") — typically one sentence to a short paragraph.
     - A boxed sidebar of related-but-tangential content.
     - An attributed citation (a quote with a source name).
     - Visually offset text that interrupts the column / prose flow.

   What IS NOT an insert (leave it exactly where it is — do NOT touch):
     - A regular section that has its own heading followed by paragraphs of
       continuous prose. Even when it appears right after an image or at the
       bottom of a page, "heading + prose" is part of the main narrative and
       NEVER becomes a callout.
     - A figure caption attached to an image — keep it with the image.
     - The opening paragraph of a section, even if visually offset (drop cap,
       different font, indented).
   Rule of thumb: real sections are LONG and have a heading. Inserts are
   SHORT and tangential. When in doubt, leave it as prose.

   Treatment for true inserts:
     a. REMOVE the insert from wherever it interrupted the prose.
     b. MERGE the two halves of any interrupted paragraph so the sentence
        flows as the author wrote it.
     c. APPEND the insert at the END of the section that contained the
        interruption point — placed BEFORE the next heading. NEVER delete,
        rename, demote, merge, or replace that heading or any heading.
     d. WRAP the insert as an Obsidian callout using this exact syntax:
          > [!Info]
          > <verbatim text, one source line per `> ` line>
        Use a one-line title only when the source provides one (attribution,
        a label like "Sidebar:", a quote source). Otherwise the first line
        of content goes directly under `> [!Info]`.
     e. Multiple inserts in one section: append each as its own `> [!Info]`
        callout at the end of that section, in source order.

   FORBIDDEN — these break the document:
     - Dropping an insert (silently omitting its text).
     - DUPLICATING content. If text X is moved into a callout, the original
       location MUST NO LONGER contain X. Every sentence appears EXACTLY
       ONCE in the output.
     - Removing, renaming, demoting, or replacing any real section/chapter
       heading. Headings always survive untouched, including the heading
       that follows an inserted callout you've just placed.
     - Paraphrasing or shortening insert content — verbatim only.
     - Leaving an insert in its original mid-paragraph position when it
       interrupts the prose. If a paragraph is split by an insert, you MUST
       extract the insert and rejoin the paragraph (rule 6.a / 6.b).

7. Inserts split across a page or column boundary:
   In the source, a sidebar / pull-quote sometimes starts in one column or
   page and finishes in the next, with the OCR placing the two halves into
   what look like two different sections (because the surrounding prose
   changed). Detect this:
     - A section ends with an insert-like fragment that does NOT finish its
       thought (sentence fragment, dangling clause, open quotation mark).
     - The next section BEGINS with text that syntactically completes that
       fragment (mid-sentence, lower-case start, closing quotation mark,
       continuation of the same idea).
   When detected:
     a. UNITE the two halves into a SINGLE `> [!Info]` callout.
     b. PLACE the unified callout at the end of the section where the insert
        VISUALLY STARTED (typically the first of the two), BEFORE that
        section's next heading.
     c. The intervening section heading and its own prose REMAIN in place
        and untouched — the insert wraps around the heading visually only
        because of the page layout, not because it belongs to that section.
     d. Both surrounding sections must read coherently after the move (no
        dangling half-sentences, no broken quotations).

PRESERVE STRICTLY:
- ALL content. Every fact, sentence, and detail. NEVER paraphrase, summarise,
  omit, add, or DUPLICATE. The output should be approximately the SAME LENGTH
  as the input. This applies with full force to citations, pull-quotes, and
  sidebar inserts — they are not decoration, they are content, and they must
  appear EXACTLY ONCE.
- Real Markdown headings, lists, tables, code blocks, image references
  (![alt](path)), and existing blockquotes (>) and callouts (> [!...]).
  ALL section / chapter headings survive untouched — never delete, rename,
  demote, or merge a heading, especially when placing an insert callout
  before it.
- Hyphenated compound words ("end-to-end", "trade-offs", "long-term").
- The document's language. Do NOT translate. Source language: {lang}.
- YAML frontmatter (--- ... ---) at the top of the document, if present.

OBSIDIAN-FRIENDLY MARKDOWN:
- ATX-style headings only (# ## ### — never underline-style).
- Obsidian callouts `> [!Info]` for sidebars / pull quotes / citations / inserts
  (see rule 6). Plain `>` blockquotes are reserved for in-line author quotations
  that already lived inside the prose.
- Image references on their own line: ![](path)
- GFM table syntax (| col | col |) for real tables.
- One blank line between blocks; two between major sections.

OUTPUT FORMAT:
- Output ONLY the cleaned Markdown. No preamble, no explanation, no closing
  notes, no "Here is the cleaned version:" prefix.
- Do NOT wrap the output in a code block.
- Resist any urge to shorten — this is editing, not summarising."""


_CHUNK_NOTE = (
    "\n\nNOTE: You are cleaning ONE chunk of a longer document. Output ONLY "
    "the cleaned text of THIS chunk. Do not invent transitions to other parts, "
    "do not add a heading or framing, do not signal that more text follows."
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_env()

    _engine_env = os.environ.get('AI_ENGINE', 'lmstudio').lower()
    _lm_host    = os.environ.get('LM_STUDIO_HOST',          'http://localhost:1234')
    _lm_b       = os.environ.get('LM_STUDIO_MODEL_B',       os.environ.get('LM_STUDIO_MODEL', ''))
    _lm_b_ctx   = int(os.environ.get('LM_STUDIO_MODEL_B_CTX', '52500'))
    _ol_host    = os.environ.get('OLLAMA_HOST',             'http://localhost:11434')
    _ol_b       = os.environ.get('OLLAMA_MODEL_B',          os.environ.get('OLLAMA_MODEL', ''))
    _ol_b_ctx   = int(os.environ.get('OLLAMA_MODEL_B_CTX',   '8192'))
    _ol_ng_b    = parse_num_gpu(
        os.environ.get('OLLAMA_NUM_GPU_B', os.environ.get('OLLAMA_NUM_GPU', ''))
    )

    parser = argparse.ArgumentParser(
        description='Content-preserving OCR cleanup for Markdown files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('md_file', help='Input Markdown file')
    parser.add_argument('-o', '--output',
                        help='Output file (default: <name>_cleaned.md)')
    parser.add_argument('--engine', default=_engine_env, choices=['lmstudio', 'ollama'],
                        help='AI engine (default: from AI_ENGINE env var)')
    parser.add_argument('--host', default='', metavar='URL',
                        help='API base URL (default: from engine env vars)')
    parser.add_argument('--model', default='', metavar='ID',
                        help='Override the cleanup model (default: Model B)')
    parser.add_argument('--context-size', type=int, default=0, metavar='N',
                        help='Override context window size in tokens')
    parser.add_argument('--lang', default='', metavar='NAME',
                        help='Source language (default: auto-detect)')
    parser.add_argument('--no-model-swap', action='store_true',
                        help='Skip engine model load/unload (model is already loaded)')
    parser.add_argument('--prompt', '-p', default='',
                        help='Extra focus instructions appended to the system prompt')
    args = parser.parse_args()

    # ── Resolve engine-specific defaults ────────────────────────────────────
    if args.engine == 'ollama':
        host_default  = args.host  or _ol_host
        model_default = args.model or _ol_b
        ctx_env       = _ol_b_ctx
        num_gpu       = _ol_ng_b
    else:
        host_default  = args.host  or _lm_host
        model_default = args.model or _lm_b
        ctx_env       = _lm_b_ctx
        num_gpu       = None

    if not model_default:
        env_key = 'OLLAMA_MODEL_B' if args.engine == 'ollama' else 'LM_STUDIO_MODEL_B'
        log.error(f"No cleanup model configured. Set {env_key} in .env or pass --model.")
        sys.exit(1)

    md_path = os.path.abspath(args.md_file)
    if not os.path.isfile(md_path):
        log.error(f"File not found: {md_path}")
        sys.exit(1)

    out_path = (os.path.abspath(args.output) if args.output
                else os.path.splitext(md_path)[0] + '_cleaned.md')

    with open(md_path, encoding='utf-8') as f:
        md_text = f.read()
    if not md_text.strip():
        log.error("Input file is empty.")
        sys.exit(1)

    # ── Resolve backend / model / context ────────────────────────────────────
    resolved_host = resolve_host(host_default)
    backend       = create_backend(args.engine, resolved_host)
    client        = backend.create_client()

    model, ctx = get_model_info(backend, model_default)
    if ctx_env:
        ctx = ctx_env
    if args.context_size:
        ctx = args.context_size

    # ── Detect language ──────────────────────────────────────────────────────
    if args.lang:
        lang = args.lang
    else:
        _, lang = detect_language(md_text)

    # ── Decide single-pass vs chunked ───────────────────────────────────────
    # Output budget = source size × 1.2 (cleanup may slightly grow text when
    # rejoining split words). Reserve ~600 tokens for prompt + structure overhead.
    # Single-pass requires both input and output to fit alongside the prompt
    # within 85 % of the model context.
    t_doc          = tok(md_text)
    output_budget  = max(512, int(t_doc * 1.2))
    prompt_reserve = 600
    headroom       = int(ctx * 0.85)
    fits_single    = (prompt_reserve + t_doc + output_budget) <= headroom

    # ── Header ──────────────────────────────────────────────────────────────
    print(_SEP)
    print(f"  {os.path.basename(md_path)}")
    print(f"  engine: {args.engine}  ·  model: {model}")
    print(f"  ctx:    {ctx}  ·  doc: {t_doc} tok ({tok_to_words(t_doc)} words)"
          f"  ·  lang: {lang}")
    print(f"  mode:   {'single-pass' if fits_single else 'chunked'}")
    print(f"  output: {os.path.basename(out_path)}")
    print(_SEP)

    # ── Load model ──────────────────────────────────────────────────────────
    if not args.no_model_swap:
        ensure_model_loaded(backend, model, ctx, num_gpu=num_gpu)
    extra_body = backend.get_inference_extra_body()

    system_prompt = _CLEANUP_SYSTEM.format(lang=lang)
    if args.prompt:
        system_prompt += f"\n\nADDITIONAL FOCUS: {args.prompt}"

    t0 = time.time()

    # ── Run ─────────────────────────────────────────────────────────────────
    # Stream tokens to the output file as they arrive — if the user interrupts
    # (Ctrl-C, kill, crash), the partial cleaned text is already on disk.
    tee = open(out_path, 'w', encoding='utf-8')
    try:
        if fits_single:
            cleaned = infer(
                client, model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': md_text},
                ],
                max_tokens=output_budget,
                temperature=0.2,
                stream=True,
                label='Cleaning the whole document…',
                extra_body=extra_body,
                tee_file=tee,
                show_progress=True,
            )
            result = cleaned.strip()
        else:
            # Chunked: paragraph-aware splits, no overlap (clean text only — chunks
            # break at paragraph boundaries so cross-chunk word splits are rare).
            max_content = max(1024, int(ctx * 0.40))
            blocks      = parse_md_blocks(md_text)
            chunks      = build_chunks(blocks, max_content, overlap_tokens=0)
            log.info(f"Split into {len(chunks)} chunk(s) of ≤ {max_content} tokens.")

            chunk_system = system_prompt + _CHUNK_NOTE
            cleaned_parts: list[str] = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    tee.write('\n\n'); tee.flush()
                chunk_tok    = tok(chunk)
                chunk_budget = max(512, int(chunk_tok * 1.25))
                cleaned = infer(
                    client, model,
                    messages=[
                        {'role': 'system', 'content': chunk_system},
                        {'role': 'user',   'content': chunk},
                    ],
                    max_tokens=chunk_budget,
                    temperature=0.2,
                    stream=True,
                    label=f'  Chunk {i + 1}/{len(chunks)}  ({chunk_tok} tok)…',
                    extra_body=extra_body,
                    tee_file=tee,
                    show_progress=True,
                )
                cleaned_parts.append(cleaned.strip())

            result = '\n\n'.join(cleaned_parts)
    finally:
        tee.close()

    # ── Unload ──────────────────────────────────────────────────────────────
    if not args.no_model_swap:
        t = _step("Unload model")
        backend.unload(model)
        _ok(t, "unloaded")

    # ── Save + summary ──────────────────────────────────────────────────────
    # Final rewrite: the streamed file has chunk-edge whitespace from the model;
    # `result` is the canonical stripped/joined version. Same content, cleaner edges.
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(result)

    out_tok   = tok(result)
    out_words = tok_to_words(out_tok)
    src_words = tok_to_words(t_doc)
    delta_pct = ((out_words - src_words) / src_words * 100) if src_words else 0
    el_total  = time.time() - t0
    avg_rate  = out_tok / el_total if el_total > 0 else 0

    print()
    print(_SEP)
    print(f"  Total:   {_elapsed(t0)}")
    print(f"  Source:  {src_words} words")
    print(f"  Output:  {out_words} words   ({'+' if delta_pct >= 0 else ''}{delta_pct:.1f} %)")
    print(f"  Speed:   {avg_rate:.1f} tok/s   ({out_tok} tok generated)")
    print(f"  Saved:   {out_path}")
    print(_SEP)


if __name__ == '__main__':
    main()
