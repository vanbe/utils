#!/usr/bin/env python3
"""
transform_pipeline.py — Shared Map-Reduce-Refine pipeline for AI document transformation.

Architecture
------------
Two execution paths chosen dynamically at runtime:

  Path Alpha (Direct)
      Model B processes the full document in two passes within a single context.
      Pass 1: structured content extraction.  Pass 2: final synthesis.
      Selected when document fits Model B's context AND compression ratio < threshold.

  Path Beta (Map-Reduce-Refine)
      Model A extracts content from each chunk (JSON-enforced, temperature 0.1).
      Partial outputs are reduced until they fit Model B's context.
      Model B performs the final coherence/fluency pass.

Transformation Templates
------------------------
All AI prompts are defined in TransformTemplate dataclasses.
Adding a new transformation type = defining a new TransformTemplate and
registering it in TEMPLATES. No pipeline code changes needed.

Built-in templates:
  summary  — Concise factual summary
  podcast  — Spoken-word transcript for audio/TTS synthesis

ModelManager handles VRAM: only one model loaded at a time.
"""

import sys, os, re, json, time, logging
from dataclasses import dataclass

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_LOG_DIR      = os.path.join(_PROJECT_ROOT, 'log')
_LOG_FILE     = os.path.join(_LOG_DIR, 'ai_utils.log')

# ---------------------------------------------------------------------------
# Model A context budget ratios
# ---------------------------------------------------------------------------
_A_PROMPT_RATIO   = 0.20   # system prompt + JSON framing overhead
_A_CONTENT_RATIO  = 0.60   # source text (including overlap prefix)
_A_OUTPUT_RATIO   = 0.20   # partial output per chunk
_A_OVERLAP_TOKENS = 200    # cross-chunk continuity window

# ---------------------------------------------------------------------------
# Path-selection thresholds
# ---------------------------------------------------------------------------
_BUFFER_RATIO         = 0.15   # safety margin on Model B context
_FIDELITY_THRESHOLD   = 15.0   # max compression ratio for Path Alpha
_VALIDATION_MIN_RATIO = 0.80   # expand if output < 80% of target tokens
_VALIDATION_MAX_RATIO = 1.30   # condense if output > 130% of target tokens
_TOKENS_PER_WORD      = 1.35   # EN/FR average tokens per word

# ---------------------------------------------------------------------------
# Retry / timeout
# ---------------------------------------------------------------------------
_MAX_RETRIES  = 3
_RETRY_DELAY  = 2    # seconds, exponential base
_API_TIMEOUT  = 180  # seconds per inference call
_LOAD_TIMEOUT = 120  # seconds to wait for model ready


# ===========================================================================
# Logging
# ===========================================================================

def _setup_logging() -> logging.Logger:
    os.makedirs(_LOG_DIR, exist_ok=True)
    log = logging.getLogger('transform_md')
    log.setLevel(logging.DEBUG)
    if log.handlers:
        return log
    fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(_LOG_FILE, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter('  %(message)s'))
    log.addHandler(fh)
    log.addHandler(sh)
    return log

log = _setup_logging()


# ===========================================================================
# Token utilities
# ===========================================================================

try:
    import tiktoken as _tiktoken
    _ENC = _tiktoken.get_encoding('cl100k_base')
except ImportError:
    print("tiktoken not installed. Run: pip install tiktoken", file=sys.stderr)
    sys.exit(1)

def tok(text: str) -> int:
    return len(_ENC.encode(text))

def words_to_tok(n: int) -> int:
    return int(n * _TOKENS_PER_WORD)

def tok_to_words(n: int) -> int:
    return int(n / _TOKENS_PER_WORD)


# ===========================================================================
# Language detection
# ===========================================================================

_LANG_NAMES = {
    'af':'Afrikaans','ar':'Arabic','bg':'Bulgarian','ca':'Catalan','cs':'Czech',
    'da':'Danish','de':'German','el':'Greek','en':'English','es':'Spanish',
    'et':'Estonian','fa':'Persian','fi':'Finnish','fr':'French','he':'Hebrew',
    'hi':'Hindi','hr':'Croatian','hu':'Hungarian','id':'Indonesian','it':'Italian',
    'ja':'Japanese','ko':'Korean','lt':'Lithuanian','lv':'Latvian','nl':'Dutch',
    'no':'Norwegian','pl':'Polish','pt':'Portuguese','ro':'Romanian','ru':'Russian',
    'sk':'Slovak','sl':'Slovenian','sv':'Swedish','th':'Thai','tr':'Turkish',
    'uk':'Ukrainian','vi':'Vietnamese','zh-cn':'Chinese','zh-tw':'Chinese (Traditional)',
}

def detect_language(text: str) -> 'tuple[str, str]':
    try:
        from langdetect import detect
    except ImportError:
        log.warning("langdetect not installed — defaulting to English.")
        return 'en', 'English'
    sample = re.sub(r'```[\s\S]*?```|`[^`]+`|!\[.*?\]\(.*?\)', '', text[:3000])
    sample = re.sub(r'\[([^\]]+)\]\([^)]*\)|^#{1,6}\s+|[*_~|`>]', '', sample,
                    flags=re.MULTILINE).strip()
    try:
        code = detect(sample)
        name = _LANG_NAMES.get(code, _LANG_NAMES.get(code.split('-')[0], code))
        return code, name
    except Exception:
        return 'en', 'English'


# ===========================================================================
# Markdown structure parser  (atomic blocks — never split across chunks)
# ===========================================================================

def parse_md_blocks(md_text: str) -> list:
    """
    Parse Markdown into atomic blocks: {'type': str, 'content': str}.
    Types: heading | code_block | table | list | paragraph.
    Tables and lists are treated as indivisible units.
    """
    blocks, lines, i = [], md_text.splitlines(keepends=True), 0

    while i < len(lines):
        line, stripped = lines[i], lines[i].strip()

        # Fenced code block
        if stripped.startswith('```') or stripped.startswith('~~~'):
            fence, buf = stripped[:3], [line]
            i += 1
            while i < len(lines):
                buf.append(lines[i])
                if lines[i].strip().startswith(fence) and len(buf) > 1:
                    i += 1; break
                i += 1
            blocks.append({'type': 'code_block', 'content': ''.join(buf)}); continue

        # Table (all consecutive pipe lines are one atomic block)
        if stripped.startswith('|') and '|' in stripped[1:]:
            buf = [line]; i += 1
            while i < len(lines):
                s = lines[i].strip()
                if s.startswith('|') and '|' in s[1:]:
                    buf.append(lines[i]); i += 1
                else: break
            blocks.append({'type': 'table', 'content': ''.join(buf)}); continue

        # Heading
        if re.match(r'^#{1,6} ', stripped):
            blocks.append({'type': 'heading', 'content': line}); i += 1; continue

        # List (collect all items + continuation lines)
        if re.match(r'^(\s{0,3})([-*+]|\d+[.)]) ', stripped):
            buf = [line]; i += 1
            while i < len(lines):
                s, ss = lines[i], lines[i].strip()
                if re.match(r'^(\s{0,3})([-*+]|\d+[.)]) ', ss):
                    buf.append(s); i += 1
                elif s.startswith('    ') or s.startswith('\t'):
                    buf.append(s); i += 1
                elif ss == '' and i + 1 < len(lines) and \
                        re.match(r'^(\s{0,3})([-*+]|\d+[.)]) ', lines[i+1].strip()):
                    buf.append(s); i += 1
                else: break
            blocks.append({'type': 'list', 'content': ''.join(buf)}); continue

        # Blank line
        if stripped == '':
            i += 1; continue

        # Paragraph (until blank line or structural marker)
        buf = [line]; i += 1
        while i < len(lines):
            s, ss = lines[i], lines[i].strip()
            if ss == '':
                i += 1; break
            if re.match(r'^#{1,6} |^```|^~~~|^\|', ss) or \
               re.match(r'^(\s{0,3})([-*+]|\d+[.)]) ', ss):
                break
            buf.append(s); i += 1
        blocks.append({'type': 'paragraph', 'content': ''.join(buf)})

    return blocks


def build_chunks(blocks: list, max_content_tokens: int,
                 overlap_tokens: int = _A_OVERLAP_TOKENS) -> list:
    """
    Greedily pack atomic blocks into chunks ≤ max_content_tokens.
    Headings are always pinned to the content block that follows them.
    Overlap: the last `overlap_tokens` of the previous chunk are prepended
    as context (marked […]) to maintain cross-boundary continuity.
    """
    chunks, current_buf, current_tok, overlap_tail = [], [], 0, ''

    def flush():
        nonlocal current_buf, current_tok, overlap_tail
        if not current_buf: return
        text = ''.join(current_buf)
        chunks.append(text)
        tail = _ENC.encode(text)
        overlap_tail = _ENC.decode(tail[-overlap_tokens:] if len(tail) > overlap_tokens else tail)
        current_buf, current_tok = [], 0

    def overlap_prefix() -> str:
        return f'[…]\n{overlap_tail}\n\n' if overlap_tail else ''

    pending_heading = None
    for block in blocks:
        content, btype = block['content'], block['type']

        if btype == 'heading':
            pending_heading = content; continue

        parts = ([pending_heading] if pending_heading else []) + [content]
        pending_heading = None
        combined, combined_tok = ''.join(parts), tok(''.join(parts))

        if combined_tok > max_content_tokens:
            # Block too large to fit in any chunk — emit as its own chunk
            flush()
            chunks.append(overlap_prefix() + combined)
            tail = _ENC.encode(combined)
            overlap_tail = _ENC.decode(tail[-overlap_tokens:] if len(tail) > overlap_tokens else tail)
        elif current_tok + combined_tok > max_content_tokens:
            flush()
            pfx = overlap_prefix()
            current_buf = [pfx + combined]
            current_tok = tok(current_buf[0])
        else:
            current_buf.append(combined)
            current_tok += combined_tok

    if pending_heading:
        current_buf.append(pending_heading)
    flush()
    return chunks


# ===========================================================================
# Pivot calculation
# ===========================================================================

def calculate_pivot(t_doc: int, t_target: int, model_b_ctx: int,
                    force_beta: bool = False) -> 'tuple[str, dict]':
    buffer_tokens     = int(model_b_ctx * _BUFFER_RATIO)
    context_headroom  = model_b_ctx - buffer_tokens
    fits_in_context   = (t_doc + t_target) < context_headroom
    compression_ratio = t_doc / t_target if t_target > 0 else float('inf')
    fidelity_ok       = compression_ratio < _FIDELITY_THRESHOLD
    path = 'beta'
    if not force_beta and fits_in_context and fidelity_ok:
        path = 'alpha'
    metrics = {
        'T_doc':             t_doc,
        'T_target':          t_target,
        'C_max (Model B)':   model_b_ctx,
        'Buffer (15%)':      buffer_tokens,
        'Context headroom':  context_headroom,
        'Fits in context':   fits_in_context,
        'Compression ratio': f'{compression_ratio:.1f}:1',
        'Fidelity OK (<15)': fidelity_ok,
        'Forced Beta':       force_beta,
        'Selected path':     f'PATH {"ALPHA (Direct)" if path == "alpha" else "BETA (Map-Reduce)"}',
    }
    log.info("─" * 60)
    log.info("PIVOT DECISION")
    for k, v in metrics.items():
        log.info(f"  {k:<26} {v}")
    log.info("─" * 60)
    return path, metrics


# ===========================================================================
# JSON enforcement helpers  (Model A output is always JSON)
# ===========================================================================

def _parse_partial(raw: str) -> 'str | None':
    """Extract 'partial_summary' value from model JSON output (multi-strategy)."""
    raw = raw.strip()
    try:
        d = json.loads(raw)
        if isinstance(d, dict) and 'partial_summary' in d:
            return str(d['partial_summary'])
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', raw)
    if m:
        try:
            d = json.loads(m.group(1))
            if 'partial_summary' in d:
                return str(d['partial_summary'])
        except json.JSONDecodeError:
            pass
    m = re.search(r'"partial_summary"\s*:\s*"([\s\S]*?)"\s*[,}]', raw)
    if m:
        try:
            return json.loads(f'"{m.group(1)}"')
        except Exception:
            return m.group(1)
    return None


# ===========================================================================
# AI residue cleanup
# ===========================================================================

_RESIDUE_PATTERNS = [
    r'(?i)^(here(?:\'s| is)|voici|voilà|below is|the following is|ci-dessous)[^.:\n]*[.:\s]+',
    r'(?i)^(this (?:is a )?(?:summary|résumé|transcript|script)|ce document)[^.\n]*\.?\n?',
    r'(?i)^(i have summarized|j\'ai résumé|in summary|en résumé|to summarize)[^.\n]*\.?\n?',
    r'(?i)(i hope this helps|n\'hésitez pas|feel free to ask|let me know)[^.]*\.?\s*$',
    r'(?i)^note\s*:\s*this (summary|transcript|script)[^.\n]*\.?\n?',
]

def strip_residue(text: str) -> str:
    for p in _RESIDUE_PATTERNS:
        text = re.sub(p, '', text, flags=re.MULTILINE).strip()
    return text


# ===========================================================================
# Inference wrapper with retry + exponential backoff
# ===========================================================================

def infer(client, model_id: str, messages: list, max_tokens: int,
          temperature: float, stream: bool = False, label: str = '',
          extra_body: dict = None, tee_file=None,
          show_progress: bool = False) -> str:
    if label:
        log.info(label)
    kwargs: dict = dict(
        model=model_id, messages=messages, temperature=temperature,
        max_tokens=max_tokens, stream=stream, timeout=_API_TIMEOUT,
    )
    if extra_body:
        kwargs['extra_body'] = extra_body
    for attempt in range(1, _MAX_RETRIES + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(**kwargs)
            if stream:
                parts = []
                tok_count = 0
                last_status = 0.0
                if show_progress:
                    # Initial tick before any tokens arrive — prompt-eval on a
                    # large input can take many seconds; without this the screen
                    # looks frozen between the label and the first token.
                    sys.stderr.write('\r  waiting for first token…\033[K')
                    sys.stderr.flush()
                for chunk in resp:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        if show_progress:
                            tok_count += 1
                            now = time.time()
                            if now - last_status >= 0.25:
                                el = now - t0
                                rate = tok_count / el if el > 0 else 0
                                sys.stderr.write(f'\r  {tok_count} tok  {rate:.1f} tok/s\033[K')
                                sys.stderr.flush()
                                last_status = now
                        else:
                            sys.stderr.write(delta); sys.stderr.flush()
                        if tee_file is not None:
                            tee_file.write(delta); tee_file.flush()
                        parts.append(delta)
                if show_progress:
                    el = time.time() - t0
                    final_tok = tok(''.join(parts))
                    rate = final_tok / el if el > 0 else 0
                    sys.stderr.write(f'\r  {final_tok} tok in {el:.1f}s  avg {rate:.1f} tok/s\033[K\n')
                    sys.stderr.flush()
                else:
                    sys.stderr.write('\n'); sys.stderr.flush()
                result = ''.join(parts)
            else:
                result = resp.choices[0].message.content or ''
            log.debug(f"  → {tok(result)} tokens in {time.time()-t0:.1f}s (attempt {attempt})")
            return result
        except Exception as e:
            log.warning(f"API error attempt {attempt}/{_MAX_RETRIES}: {e}")
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY ** attempt)
    log.error(f"All {_MAX_RETRIES} attempts failed for: {label or model_id}")
    return ''


# ===========================================================================
# Transformation templates
# ===========================================================================

@dataclass
class TransformTemplate:
    """
    Defines all AI prompts for a document transformation type.

    Placeholder conventions (filled via .format() at call time):
      {target_words}  — integer target word count
      {lang}          — language name (e.g. 'French', 'English')
      {l_chunk}       — integer per-chunk output token budget
    """
    id:          str
    label:       str
    description: str

    # Alpha path  (document fits in Model B's context)
    alpha_extract_sys: str   # Pass 1: extract content from the full document
    alpha_synth_sys:   str   # Pass 2: synthesize to final form  {target_words} {lang}

    # Beta path  (Map-Reduce-Refine for large documents)
    map_sys:    str          # Per-chunk extraction  {l_chunk}
    reduce_sys: str          # Consolidate partials  {l_chunk}
    refine_sys: str          # Final fluency pass    {target_words} {lang}

    # Validation prompts
    expand_sys:   str        # Output too short  {target_words} {lang}
    condense_sys: str        # Output too long   {target_words} {lang}


# ---------------------------------------------------------------------------
# Template: Summary
# ---------------------------------------------------------------------------

_SUMMARY = TransformTemplate(
    id          = 'summary',
    label       = 'Summary',
    description = 'Concise summary preserving key facts, arguments, and insights',

    alpha_extract_sys = (
        "You are a precise information extractor. "
        "Read the document and extract every key fact, argument, data point, decision, "
        "and conclusion as a comprehensive structured bullet list. "
        "Preserve specific numbers, names, and examples exactly. "
        "Do not summarize or interpret — extract faithfully. "
        "Output only the bullet list, no preamble."
    ),
    alpha_synth_sys = (
        "You are a professional writer and editor. "
        "Using ONLY the facts provided in the structured list, write a coherent, "
        "fluent document of approximately {target_words} words. "
        "Do not invent any information not present in the list. "
        "Do not add introductory or concluding meta-sentences. "
        "Write in {lang}. Output only the final text — no title, no preamble."
    ),

    map_sys = (
        "You are a neutral information extractor. "
        "Extract the essential facts, arguments, and conclusions from the text below. "
        "Do not invent information not present in the text. "
        "Do not generate introductory or concluding sentences. "
        "Target length: approximately {l_chunk} tokens. "
        'Reply ONLY with valid JSON: {{"partial_summary": "..."}} '
        "No text outside the JSON object."
    ),
    reduce_sys = (
        "You are a neutral information extractor. "
        "Compress the following partial summaries into a single coherent intermediate summary. "
        "Remove redundancies. Preserve all distinct facts. "
        "Target length: approximately {l_chunk} tokens. "
        'Reply ONLY with valid JSON: {{"partial_summary": "..."}} '
        "No text outside the JSON object."
    ),
    refine_sys = (
        "You are a professional editor. "
        "Rewrite the draft below into a coherent, fluent document of approximately "
        "{target_words} words. "
        "Remove all seams, redundancies, and list-like artefacts. "
        "Write in {lang}. "
        "Output only the final text — no preamble, no title, no meta-commentary."
    ),

    expand_sys = (
        "The following text is too brief. Expand it by adding more specific details, "
        "examples, and analytical depth based on its existing content. "
        "Do not add information not already implied. "
        "Target: approximately {target_words} words. "
        "Write in {lang}. Output only the expanded text."
    ),
    condense_sys = (
        "The following text is too long. Condense it to approximately {target_words} words "
        "while preserving all key points. "
        "Write in {lang}. Output only the condensed text."
    ),
)


# ---------------------------------------------------------------------------
# Template: Podcast script
# ---------------------------------------------------------------------------

_PODCAST = TransformTemplate(
    id          = 'podcast',
    label       = 'Podcast script',
    description = 'Spoken-word transcript for audio/TTS — conversational, no visual artefacts',

    alpha_extract_sys = (
        "You are a content analyst preparing material for a spoken podcast. "
        "Read the document and extract every piece of content that works when heard aloud:\n"
        "- Key facts, arguments, conclusions\n"
        "- Concrete examples, stories, and analogies\n"
        "- Surprising or insightful observations\n"
        "- Context that helps a listener understand without seeing the page\n\n"
        "REMOVE entirely:\n"
        "- Footnote markers: [1], [2], ibid., op. cit., (see note 3)\n"
        "- Figure/table references: 'see Figure 3', 'Table 2 shows', 'as illustrated above'\n"
        "- Parenthetical citations: (Smith, 2020), (p. 47)\n"
        "- URLs, DOIs, ISBNs, page numbers, running headers\n"
        "- Mathematical formulas — paraphrase the concept in plain language instead\n\n"
        "PARAPHRASE visual cues into audio equivalents: "
        "'the chart shows a rise' → 'the data shows a rise'.\n"
        "Output a structured list of spoken content points. No preamble."
    ),
    alpha_synth_sys = (
        "You are a professional podcast scriptwriter. "
        "Using ONLY the content points provided, write a podcast transcript of approximately "
        "{target_words} words.\n\n"
        "Rules:\n"
        "- Natural, conversational spoken language — explaining to an intelligent friend\n"
        "- Smooth transitions between topics: 'Now,', 'What's interesting here is...', "
        "'In other words,...', 'Let's look at...'\n"
        "- No bullet lists, no citations, no URLs, no 'see Table 2' references\n"
        "- Expand abbreviations to their spoken form on first use\n"
        "- Keep technical terms but briefly clarify them in context\n"
        "- Single narrator voice throughout — no host names, no episode framing\n"
        "- No intro sentence like 'In this episode' or 'Today we discuss'\n\n"
        "Write in {lang}. Output only the script — no title, no preamble."
    ),

    map_sys = (
        "You are a content extractor for audio production. "
        "Extract the substantive content from the text below — facts, ideas, arguments, "
        "examples, and insights — that are meaningful when heard aloud.\n\n"
        "REMOVE: footnote markers, figure/table references ('see Figure 3'), "
        "parenthetical citations, URLs, page numbers, headers, mathematical formulas "
        "(paraphrase the concept instead).\n"
        "PARAPHRASE visual references into audio-friendly language.\n\n"
        "Target length: approximately {l_chunk} tokens. "
        'Reply ONLY with valid JSON: {{"partial_summary": "..."}} '
        "No text outside the JSON object."
    ),
    reduce_sys = (
        "You are a content consolidator for a podcast script. "
        "Merge the following passages into a single coherent spoken narrative segment. "
        "Remove redundancies. Preserve all distinct insights, facts, and examples. "
        "Keep the language natural and audio-friendly — no visual artefacts. "
        "Target length: approximately {l_chunk} tokens. "
        'Reply ONLY with valid JSON: {{"partial_summary": "..."}} '
        "No text outside the JSON object."
    ),
    refine_sys = (
        "You are a professional podcast scriptwriter. "
        "Rewrite the draft below as a podcast transcript of approximately {target_words} words.\n\n"
        "Rules:\n"
        "- Natural, conversational spoken language\n"
        "- Smooth transitions between topics\n"
        "- No bullet lists, no citations, no URLs, no figure/table references\n"
        "- Expand abbreviations on first use; briefly clarify technical terms in context\n"
        "- Single narrator voice — no host names, no episode framing\n\n"
        "Write in {lang}. "
        "Output only the script — no preamble, no title, no meta-commentary."
    ),

    expand_sys = (
        "The following podcast script is too brief. Expand it with more detail, examples, "
        "and context from the existing content while keeping the conversational, spoken style. "
        "Target: approximately {target_words} words. "
        "Write in {lang}. Output only the expanded script."
    ),
    condense_sys = (
        "The following podcast script is too long. Trim it to approximately {target_words} words "
        "while preserving the most important insights and the conversational flow. "
        "Write in {lang}. Output only the condensed script."
    ),
)


# ---------------------------------------------------------------------------
# Public template registry — register new templates here
# ---------------------------------------------------------------------------

TEMPLATES: dict = {
    'summary': _SUMMARY,
    'podcast': _PODCAST,
}


# ===========================================================================
# Path Alpha  (document fits in a single Model B context)
# ===========================================================================

def run_alpha(client, model_b_id: str, md_text: str,
              target_words: int, lang: str, extra_prompt: str,
              template: TransformTemplate, extra_body: dict = None) -> str:
    extract_sys = template.alpha_extract_sys
    if extra_prompt:
        extract_sys += f'\nFocus particularly on: {extra_prompt}'

    log.info("Path Alpha — Pass 1: structured extraction…")
    extracted = infer(
        client, model_b_id,
        messages=[
            {'role': 'system', 'content': extract_sys},
            {'role': 'user',   'content': md_text},
        ],
        max_tokens=words_to_tok(int(target_words * 3)),
        temperature=0.1,
        label='  Pass 1: extracting content…',
        extra_body=extra_body,
    )
    log.info(f"  Extracted: {tok(extracted)} tokens")

    synth_sys = template.alpha_synth_sys.format(target_words=target_words, lang=lang)
    log.info("Path Alpha — Pass 2: synthesis…")
    result = infer(
        client, model_b_id,
        messages=[
            {'role': 'system', 'content': synth_sys},
            {'role': 'user',   'content': extracted},
        ],
        max_tokens=words_to_tok(int(target_words * 1.4)),
        temperature=0.3, stream=True,
        label='  Pass 2: synthesizing…',
        extra_body=extra_body,
    )
    return strip_residue(result)


# ===========================================================================
# Path Beta  (Map-Reduce-Refine for large documents)
# ===========================================================================

def _map_chunk(client, model_a_id: str, chunk: str, idx: int, total: int,
               l_chunk: int, extra_prompt: str, template: TransformTemplate,
               extra_body: dict = None) -> str:
    system = template.map_sys.format(l_chunk=l_chunk)
    if extra_prompt:
        system += f'\nFocus: {extra_prompt}'

    for attempt in range(1, _MAX_RETRIES + 1):
        t0  = time.time()
        raw = infer(
            client, model_a_id,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user',   'content': chunk},
            ],
            max_tokens=l_chunk + 60, temperature=0.1,
            label=f'  Chunk {idx+1}/{total}…',
            extra_body=extra_body,
        )
        parsed = _parse_partial(raw)
        if parsed is not None:
            log.info(f"  Chunk {idx+1}/{total}: {tok(chunk)}→{tok(parsed)} tokens | "
                     f"{time.time()-t0:.1f}s | attempt {attempt}")
            return parsed
        log.warning(f"  Chunk {idx+1}/{total} attempt {attempt}: JSON parse failed. "
                    f"Raw: {raw[:200]}")
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY ** attempt)

    log.error(f"  Chunk {idx+1}/{total}: all retries exhausted — using raw output.")
    return raw


def _reduce(client, model_a_id: str, model_a_ctx: int,
            partials: list, t_target: int, model_b_ctx: int,
            extra_prompt: str, template: TransformTemplate,
            extra_body: dict = None) -> str:
    """
    Concatenate partials and iteratively re-map until the result fits
    Model B's refine budget. Max 5 rounds.
    """
    refine_budget = (int(model_b_ctx * (1 - _BUFFER_RATIO))
                     - words_to_tok(int(tok_to_words(t_target) * 1.4)))
    consolidated  = '\n\n'.join(partials)
    c_tok         = tok(consolidated)

    log.info(f"Reduce: {c_tok} tokens consolidated | Model B budget: {refine_budget} tokens")

    for round_n in range(1, 6):
        if c_tok <= refine_budget:
            break
        log.info(f"Reduce round {round_n}: still too large ({c_tok} > {refine_budget})")

        s_chunk = int(model_a_ctx * _A_CONTENT_RATIO)
        chunks  = build_chunks(parse_md_blocks(consolidated), s_chunk)
        n       = len(chunks)
        l_chunk = min(int(model_a_ctx * _A_OUTPUT_RATIO),
                      max(64, int(t_target * 1.25 / n)))

        log.info(f"  Round {round_n}: {n} sub-chunks | L_chunk={l_chunk} tokens")
        consolidated = '\n\n'.join(
            _map_chunk(client, model_a_id, chunk, i, n, l_chunk, extra_prompt, template,
                       extra_body=extra_body)
            for i, chunk in enumerate(chunks)
        )
        c_tok = tok(consolidated)
        if round_n == 5:
            log.warning("Reduce: stopped after 5 rounds — proceeding.")

    log.info(f"Reduce complete: {c_tok} tokens — fits Model B context.")
    return consolidated


def run_beta(manager, client_a, md_text: str,
             target_words: int, lang: str, extra_prompt: str,
             template: TransformTemplate, extra_body_a: dict = None) -> str:
    model_a_id  = manager.model_id('a')
    model_a_ctx = manager.model_ctx('a')
    model_b_id  = manager.model_id('b')
    model_b_ctx = manager.model_ctx('b')
    t_target    = words_to_tok(target_words)

    s_chunk = int(model_a_ctx * _A_CONTENT_RATIO)
    chunks  = build_chunks(parse_md_blocks(md_text), s_chunk)
    n       = len(chunks)
    l_chunk = min(int(model_a_ctx * _A_OUTPUT_RATIO),
                  max(64, int(t_target * 1.25 / n)))

    log.info("─" * 60)
    log.info("PATH BETA — CHUNK CALIBRATION")
    log.info(f"  S_chunk (60% of {model_a_ctx})  = {s_chunk} tokens")
    log.info(f"  Total chunks                    = {n}")
    log.info(f"  L_chunk (T_target×1.25 / n)     = {l_chunk} tokens")
    log.info("─" * 60)

    log.info(f"Phase 1 — Map ({n} chunks, Model A: {model_a_id})")
    partials = [
        _map_chunk(client_a, model_a_id, chunk, i, n, l_chunk, extra_prompt, template,
                   extra_body=extra_body_a)
        for i, chunk in enumerate(chunks)
    ]

    log.info("Phase 2 — Reduce")
    consolidated = _reduce(
        client_a, model_a_id, model_a_ctx,
        partials, t_target, model_b_ctx, extra_prompt, template,
        extra_body=extra_body_a,
    )

    manager.ensure('b')
    client_b      = manager.client()
    extra_body_b  = manager.inference_extra_body()   # options for the newly-loaded Model B
    output_tokens = words_to_tok(int(target_words * 1.4))
    refine_sys    = template.refine_sys.format(target_words=target_words, lang=lang)
    if extra_prompt:
        refine_sys += f'\nFocus: {extra_prompt}'

    log.info(f"Phase 3 — Refine (Model B: {model_b_id})")
    result = infer(
        client_b, model_b_id,
        messages=[
            {'role': 'system', 'content': refine_sys},
            {'role': 'user',   'content': f'<draft>\n{consolidated}\n</draft>'},
        ],
        max_tokens=output_tokens, temperature=0.3, stream=True,
        label='  Refining…',
        extra_body=extra_body_b,
    )
    return strip_residue(result)


# ===========================================================================
# Validation guardrail  (expand or condense to hit the target word count)
# ===========================================================================

def validate_and_adjust(client, model_b_id: str, result: str,
                        target_words: int, lang: str,
                        template: TransformTemplate,
                        extra_body: dict = None) -> str:
    t_target = words_to_tok(target_words)
    t_actual = tok(result)
    ratio    = t_actual / t_target if t_target > 0 else 1.0

    log.info("─" * 60)
    log.info("VALIDATION")
    log.info(f"  Target:  {target_words} words (~{t_target} tokens)")
    log.info(f"  Actual:  {tok_to_words(t_actual)} words  ({t_actual} tokens)")
    log.info(f"  Ratio:   {ratio:.0%}  "
             f"(acceptable: {int(_VALIDATION_MIN_RATIO*100)}%–{int(_VALIDATION_MAX_RATIO*100)}%)")

    if ratio < _VALIDATION_MIN_RATIO:
        log.info("  → Too short — expanding…")
        result = infer(
            client, model_b_id,
            messages=[
                {'role': 'system', 'content': template.expand_sys.format(
                    target_words=target_words, lang=lang)},
                {'role': 'user', 'content': result},
            ],
            max_tokens=words_to_tok(int(target_words * 1.5)),
            temperature=0.3, stream=True, label='  Expanding…',
            extra_body=extra_body,
        )
        result = strip_residue(result)
        log.info(f"  After expansion: {tok_to_words(tok(result))} words")

    elif ratio > _VALIDATION_MAX_RATIO:
        log.info("  → Too long — condensing…")
        result = infer(
            client, model_b_id,
            messages=[
                {'role': 'system', 'content': template.condense_sys.format(
                    target_words=target_words, lang=lang)},
                {'role': 'user', 'content': result},
            ],
            max_tokens=words_to_tok(int(target_words * 1.2)),
            temperature=0.2, stream=True, label='  Condensing…',
            extra_body=extra_body,
        )
        result = strip_residue(result)
        log.info(f"  After condensation: {tok_to_words(tok(result))} words")

    else:
        log.info("  → Within acceptable range — no adjustment needed.")

    log.info("─" * 60)
    return result


# ===========================================================================
# ModelManager  (delegates VRAM lifecycle to engine.Backend)
# ===========================================================================

class ModelManager:
    """
    Manages AI model loading/unloading for any supported backend (LMStudio, Ollama).
    Ensures only one model is in VRAM at a time — critical for limited GPU memory.
    """

    def __init__(self, backend, model_a: str, model_a_ctx: int,
                 model_b: str, model_b_ctx: int, no_swap: bool = False,
                 num_gpu_a: 'int | None' = None, num_gpu_b: 'int | None' = None):
        self.backend = backend
        self.no_swap = no_swap
        self._models = {
            'a': (model_a, model_a_ctx, num_gpu_a),
            'b': (model_b, model_b_ctx, num_gpu_b),
        }
        self._active = None

    def ensure(self, role: str) -> bool:
        """Load and verify the model for role 'a' or 'b'. Returns True when ready."""
        model_id, ctx, num_gpu = self._models[role]
        if self.no_swap:
            log.debug(f"Lifecycle disabled — assuming {model_id} is loaded.")
            return True
        current = self.backend.get_loaded()
        if current == model_id:
            log.info(f"Model already active: {model_id}")
            self._active = model_id
            return True
        if current:
            log.info(f"Unloading: {current}")
            self.backend.unload(current)
            time.sleep(1)
        gpu_note = f" (num_gpu={num_gpu})" if num_gpu is not None else ''
        log.info(f"Loading Model {'A' if role == 'a' else 'B'}: {model_id} (ctx {ctx}){gpu_note}…")
        if not self.backend.load(model_id, ctx, num_gpu=num_gpu):
            log.error(f"Load failed for {model_id}. Load it manually in the engine.")
            return False
        log.info(f"Waiting for {model_id} to be ready…")
        ready = self.backend.wait_ready(model_id)
        log.info(f"Model ready: {model_id}" if ready else f"Timeout waiting for {model_id}.")
        self._active = model_id if ready else None
        return ready

    def release(self):
        """Unload the active model on script exit."""
        if self.no_swap: return
        current = self.backend.get_loaded()
        if current:
            log.info(f"Cleanup: unloading {current}…")
            self.backend.unload(current)
            log.info("No model loaded.")

    def client(self):
        return self.backend.create_client()

    def model_id(self, role: str) -> str:
        return self._models[role][0]

    def model_ctx(self, role: str) -> int:
        return self._models[role][1]

    def model_num_gpu(self, role: str) -> 'int | None':
        return self._models[role][2]

    def inference_extra_body(self) -> dict:
        """Return options to inject into inference calls for the active model.

        Must be called AFTER ensure() so the correct model options are active.
        """
        return self.backend.get_inference_extra_body()


# ===========================================================================
# Orchestrator
# ===========================================================================

def run(md_text: str, manager: ModelManager, target_words: int, lang: str,
        extra_prompt: str, template: TransformTemplate, force_beta: bool) -> str:
    """
    Run the full transformation pipeline.
    Selects Path Alpha (direct) or Path Beta (Map-Reduce-Refine) automatically.
    """
    log.info(f"Template: {template.label}")
    t_doc, t_target = tok(md_text), words_to_tok(target_words)
    path, _         = calculate_pivot(t_doc, t_target, manager.model_ctx('b'), force_beta)

    if path == 'alpha':
        manager.ensure('b')
        client_b = manager.client()
        eb_b     = manager.inference_extra_body()
        result   = run_alpha(client_b, manager.model_id('b'),
                             md_text, target_words, lang, extra_prompt, template,
                             extra_body=eb_b)
        result   = validate_and_adjust(client_b, manager.model_id('b'),
                                       result, target_words, lang, template,
                                       extra_body=eb_b)
    else:
        manager.ensure('a')
        client_a = manager.client()
        eb_a     = manager.inference_extra_body()
        result   = run_beta(manager, client_a, md_text,
                            target_words, lang, extra_prompt, template,
                            extra_body_a=eb_a)
        manager.ensure('b')
        client_b = manager.client()
        eb_b     = manager.inference_extra_body()
        result   = validate_and_adjust(client_b, manager.model_id('b'),
                                       result, target_words, lang, template,
                                       extra_body=eb_b)

    return result
