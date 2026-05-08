#!/usr/bin/env python3
"""
summarize_md.py — Resource-aware Markdown summarization via LMStudio.

Two execution paths, chosen dynamically at runtime:

  Path Alpha (Direct)
      Model B processes the full document in two passes within a single context.
      Pass 1: structured fact extraction.  Pass 2: fluent synthesis to target length.
      Selected when the document fits Model B's context AND the compression ratio
      is below the fidelity threshold (default 15:1).

  Path Beta (Map-Reduce-Refine)
      Model A extracts facts from each chunk (JSON-enforced, temperature 0.1).
      Partial summaries are reduced until they fit Model B's context.
      Model B performs the final coherence/fluency pass.

ModelManager handles VRAM: only one model loaded at a time.
All decision variables are logged for full auditability.

.env:
  LM_STUDIO_HOST        = http://localhost:1234
  LM_STUDIO_MODEL_A     = <map model identifier>      # e.g. gemma-4-27b-a4b
  LM_STUDIO_MODEL_A_CTX = 4096
  LM_STUDIO_MODEL_B     = <refine model identifier>   # e.g. gemma-4-e4b-it
  LM_STUDIO_MODEL_B_CTX = 52500
  LM_STUDIO_MODEL       = <single-model fallback>

Dependencies: pip install openai tiktoken langdetect

Usage:
  python summarize_md.py doc.md --target-words 500
  python summarize_md.py doc.md --target-words 1000 --lang French -o out.md
  python summarize_md.py doc.md --no-model-swap     # skip lifecycle management
  python summarize_md.py doc.md --force-beta        # always use Map-Reduce path
"""

import sys, os, re, json, time, logging, argparse

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Engine module lives in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Backend, create_backend, resolve_host  # noqa: E402
from ai_common import load_env                            # noqa: E402
_LOG_DIR      = os.path.join(_PROJECT_ROOT, 'log')
_LOG_FILE     = os.path.join(_LOG_DIR, 'ai_utils.log')

# ---------------------------------------------------------------------------
# Model A budget constants  (% of context window)
# ---------------------------------------------------------------------------
_A_PROMPT_RATIO  = 0.20   # system prompt + JSON framing
_A_CONTENT_RATIO = 0.60   # source text (incl. overlap prefix)
_A_OUTPUT_RATIO  = 0.20   # partial_summary output
_A_OVERLAP_TOKENS = 200   # overlap between consecutive chunks

# ---------------------------------------------------------------------------
# Path-selection thresholds
# ---------------------------------------------------------------------------
_BUFFER_RATIO        = 0.15   # safety margin on Model B context
_FIDELITY_THRESHOLD  = 15.0   # max acceptable compression ratio for Path Alpha
_VALIDATION_MIN_RATIO = 0.80  # expand if output < 80% of target tokens
_VALIDATION_MAX_RATIO = 1.30  # condense if output > 130% of target tokens
_TOKENS_PER_WORD      = 1.35  # EN/FR average

# ---------------------------------------------------------------------------
# Retry / timeout
# ---------------------------------------------------------------------------
_MAX_RETRIES   = 3
_RETRY_DELAY   = 2    # seconds, exponential base
_API_TIMEOUT   = 180  # seconds per inference call
_LOAD_TIMEOUT  = 120  # seconds to wait for model ready


# ===========================================================================
# Logging
# ===========================================================================

def _setup_logging() -> logging.Logger:
    os.makedirs(_LOG_DIR, exist_ok=True)
    log = logging.getLogger('summarize_md')
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

def detect_language(text: str) -> tuple[str, str]:
    try:
        from langdetect import detect
    except ImportError:
        log.warning("langdetect not installed — defaulting to English.")
        return 'en', 'English'
    sample = text[:3000]
    sample = re.sub(r'```[\s\S]*?```|`[^`]+`|!\[.*?\]\(.*?\)', '', sample)
    sample = re.sub(r'\[([^\]]+)\]\([^)]*\)|^#{1,6}\s+|[*_~|`>]', '', sample,
                    flags=re.MULTILINE).strip()
    try:
        code = detect(sample)
        name = _LANG_NAMES.get(code, _LANG_NAMES.get(code.split('-')[0], code))
        return code, name
    except Exception:
        return 'en', 'English'


# ===========================================================================
# ModelManager — VRAM lifecycle  (delegates to engine.Backend)
# ===========================================================================

class ModelManager:
    """
    Manages AI model loading/unloading for any supported backend (LMStudio,
    Ollama). Ensures only one model is in VRAM at a time — critical for
    machines with limited GPU memory.
    """

    def __init__(self, backend: Backend, model_a: str, model_a_ctx: int,
                 model_b: str, model_b_ctx: int, no_swap: bool = False):
        self.backend   = backend
        self.no_swap   = no_swap
        self._models   = {
            'a': (model_a, model_a_ctx),
            'b': (model_b, model_b_ctx),
        }
        self._active: 'str | None' = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure(self, role: str) -> bool:
        """
        Ensure the model for `role` ('a' or 'b') is loaded and ready.
        Unloads whatever is currently active first if different.
        Returns True when the model is ready for inference.
        """
        model_id, ctx = self._models[role]
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
            time.sleep(1)   # brief settle

        log.info(f"Loading Model {'A' if role == 'a' else 'B'}: {model_id}  "
                 f"(context: {ctx} tokens)…")
        if not self.backend.load(model_id, ctx):
            log.error(f"Load request failed for {model_id}. "
                      "Ensure the model is loaded manually in the engine.")
            return False

        log.info(f"Waiting for {model_id} to be ready…")
        ready = self.backend.wait_ready(model_id)
        if ready:
            log.info(f"Model ready: {model_id}")
        else:
            log.warning(f"Timeout waiting for {model_id}.")
        self._active = model_id if ready else None
        return ready

    def release(self):
        """Unload active model on script exit (return GPU to neutral state)."""
        if self.no_swap:
            return
        current = self.backend.get_loaded()
        if current:
            log.info(f"Cleanup: unloading {current}…")
            self.backend.unload(current)
            log.info("No model loaded (neutral state).")

    def client(self):
        """Return an OpenAI-compatible client for the active backend."""
        return self.backend.create_client()

    def model_id(self, role: str) -> str:
        return self._models[role][0]

    def model_ctx(self, role: str) -> int:
        return self._models[role][1]


# ===========================================================================
# Markdown block parser  (structure-aware, atomic blocks)
# ===========================================================================

def parse_md_blocks(md_text: str) -> list[dict]:
    """
    Parse Markdown into atomic blocks that must not be split across chunk boundaries.
    Returns list of {'type': str, 'content': str} dicts.
    Types: heading | code_block | table | list | paragraph | blank
    """
    blocks: list[dict] = []
    lines  = md_text.splitlines(keepends=True)
    i      = 0

    while i < len(lines):
        line    = lines[i]
        stripped = line.strip()

        # --- Fenced code block ----------------------------------------
        if stripped.startswith('```') or stripped.startswith('~~~'):
            fence = stripped[:3]
            buf   = [line]
            i    += 1
            while i < len(lines):
                buf.append(lines[i])
                if lines[i].strip().startswith(fence) and len(buf) > 1:
                    i += 1
                    break
                i += 1
            blocks.append({'type': 'code_block', 'content': ''.join(buf)})
            continue

        # --- Table (all consecutive | lines are one atomic block) ------
        if stripped.startswith('|') and '|' in stripped[1:]:
            buf = [line]
            i  += 1
            while i < len(lines):
                s = lines[i].strip()
                if s.startswith('|') and '|' in s[1:]:
                    buf.append(lines[i])
                    i += 1
                else:
                    break
            blocks.append({'type': 'table', 'content': ''.join(buf)})
            continue

        # --- Heading ----------------------------------------------------
        if re.match(r'^#{1,6} ', stripped):
            blocks.append({'type': 'heading', 'content': line})
            i += 1
            continue

        # --- List (collect all consecutive items + continuations) ------
        if re.match(r'^(\s{0,3})([-*+]|\d+[.)]) ', stripped):
            buf = [line]
            i  += 1
            while i < len(lines):
                s = lines[i]
                ss = s.strip()
                # continuation: indented line, or another list marker
                if re.match(r'^(\s{0,3})([-*+]|\d+[.)]) ', ss):
                    buf.append(s); i += 1
                elif s.startswith('    ') or s.startswith('\t'):  # indented block
                    buf.append(s); i += 1
                elif ss == '' and i + 1 < len(lines) and \
                        re.match(r'^(\s{0,3})([-*+]|\d+[.)]) ', lines[i+1].strip()):
                    # blank line between items is allowed
                    buf.append(s); i += 1
                else:
                    break
            blocks.append({'type': 'list', 'content': ''.join(buf)})
            continue

        # --- Blank line -------------------------------------------------
        if stripped == '':
            i += 1
            continue

        # --- Paragraph (until blank line or structural marker) ----------
        buf = [line]
        i  += 1
        while i < len(lines):
            s  = lines[i]
            ss = s.strip()
            if ss == '':
                i += 1
                break
            if re.match(r'^#{1,6} |^```|^~~~|^\|', ss) or \
               re.match(r'^(\s{0,3})([-*+]|\d+[.)]) ', ss):
                break
            buf.append(s); i += 1
        blocks.append({'type': 'paragraph', 'content': ''.join(buf)})

    return blocks


def build_chunks(blocks: list[dict], max_content_tokens: int,
                 overlap_tokens: int = _A_OVERLAP_TOKENS) -> list[str]:
    """
    Pack atomic blocks into chunks ≤ max_content_tokens.
    Headings are always kept with the content that follows them.
    Overlap: the last `overlap_tokens` of the previous chunk are prepended as
    context (marked with `[…]`) so the model has cross-boundary continuity.
    Large individual blocks (e.g. huge tables) are passed as their own chunk.
    """
    chunks:       list[str] = []
    current_buf:  list[str] = []
    current_tok              = 0
    overlap_tail             = ''

    def flush():
        nonlocal current_buf, current_tok, overlap_tail
        if not current_buf:
            return
        text = ''.join(current_buf)
        chunks.append(text)
        tail_enc  = _ENC.encode(text)
        tail_enc  = tail_enc[-overlap_tokens:] if len(tail_enc) > overlap_tokens else tail_enc
        overlap_tail  = _ENC.decode(tail_enc)
        current_buf   = []
        current_tok   = 0

    def overlap_prefix() -> str:
        return f'[…]\n{overlap_tail}\n\n' if overlap_tail else ''

    pending_heading: str | None = None

    for block in blocks:
        content = block['content']
        btype   = block['type']
        bt      = tok(content)

        # Pin heading to following content: defer it
        if btype == 'heading':
            pending_heading = content
            continue

        # Materialize any pending heading before this block
        prefix_parts: list[str] = []
        if pending_heading:
            prefix_parts.append(pending_heading)
            pending_heading = None
        prefix_parts.append(content)
        combined      = ''.join(prefix_parts)
        combined_tok  = tok(combined)

        if combined_tok > max_content_tokens:
            # Block is individually too large — emit as its own chunk
            flush()
            pfx = overlap_prefix()
            chunks.append(pfx + combined)
            tail_enc     = _ENC.encode(combined)
            tail_enc     = tail_enc[-overlap_tokens:] if len(tail_enc) > overlap_tokens else tail_enc
            overlap_tail = _ENC.decode(tail_enc)
        elif current_tok + combined_tok > max_content_tokens:
            flush()
            pfx = overlap_prefix()
            current_buf = [pfx + combined]
            current_tok = tok(current_buf[0])
        else:
            current_buf.append(combined)
            current_tok += combined_tok

    if pending_heading:   # trailing heading with no content
        current_buf.append(pending_heading)

    flush()
    return chunks


# ===========================================================================
# Pivot calculation
# ===========================================================================

def calculate_pivot(t_doc: int, t_target: int, model_b_ctx: int,
                    force_beta: bool = False) -> tuple[str, dict]:
    """
    Determine the processing path and log all decision variables.

    Returns ('alpha', metrics) or ('beta', metrics).
    metrics dict is logged and available for audit.
    """
    buffer_tokens      = int(model_b_ctx * _BUFFER_RATIO)
    context_headroom   = model_b_ctx - buffer_tokens
    fits_in_context    = (t_doc + t_target) < context_headroom
    compression_ratio  = t_doc / t_target if t_target > 0 else float('inf')
    fidelity_ok        = compression_ratio < _FIDELITY_THRESHOLD

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
# JSON enforcement helpers
# ===========================================================================

def parse_partial_summary(raw: str) -> str | None:
    """Extract 'partial_summary' value from model JSON output."""
    raw = raw.strip()
    # Direct parse
    try:
        d = json.loads(raw)
        if isinstance(d, dict) and 'partial_summary' in d:
            return str(d['partial_summary'])
    except json.JSONDecodeError:
        pass
    # JSON inside markdown code fence
    m = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', raw)
    if m:
        try:
            d = json.loads(m.group(1))
            if 'partial_summary' in d:
                return str(d['partial_summary'])
        except json.JSONDecodeError:
            pass
    # Regex fallback
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

_RESIDUE = [
    r'(?i)^(here(?:\'s| is)|voici|voilà|below is|the following is|ci-dessous)[^.:\n]*[.:\s]+',
    r'(?i)^(this (?:is a )?(?:summary|résumé)|ce document)[^.\n]*\.?\n?',
    r'(?i)^(i have summarized|j\'ai résumé|in summary|en résumé|to summarize)[^.\n]*\.?\n?',
    r'(?i)(i hope this helps|n\'hésitez pas|feel free to ask|let me know)[^.]*\.?\s*$',
    r'(?i)^note\s*:\s*this summary[^.\n]*\.?\n?',
]

def strip_residue(text: str) -> str:
    for p in _RESIDUE:
        text = re.sub(p, '', text, flags=re.MULTILINE).strip()
    return text


# ===========================================================================
# Inference wrapper with retry
# ===========================================================================

def infer(client, model_id: str, messages: list[dict], max_tokens: int,
          temperature: float, stream: bool = False, label: str = '') -> str:
    """Call the chat completions API with retry + exponential backoff."""
    if label:
        log.info(label)
    for attempt in range(1, _MAX_RETRIES + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model_id, messages=messages, temperature=temperature,
                max_tokens=max_tokens, stream=stream, timeout=_API_TIMEOUT,
            )
            if stream:
                parts: list[str] = []
                for chunk in resp:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        sys.stderr.write(delta); sys.stderr.flush()
                        parts.append(delta)
                sys.stderr.write('\n'); sys.stderr.flush()
                result = ''.join(parts)
            else:
                result = resp.choices[0].message.content or ''

            log.debug(f"  → {tok(result)} tokens in {time.time()-t0:.1f}s "
                      f"(attempt {attempt})")
            return result

        except Exception as e:
            log.warning(f"API error attempt {attempt}/{_MAX_RETRIES}: {e}")
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY ** attempt)

    log.error(f"All {_MAX_RETRIES} attempts failed for: {label or model_id}")
    return ''


# ===========================================================================
# Path Alpha — Direct Two-Pass  (Model B only)
# ===========================================================================

_ALPHA_P1_SYS = (
    "You are a precise information extractor. "
    "Read the document and extract every key fact, argument, data point, decision, "
    "and conclusion as a comprehensive structured bullet list. "
    "Preserve specific numbers, names, and examples exactly. "
    "Do not summarize or interpret — extract faithfully. "
    "Output only the bullet list, no preamble."
)

_ALPHA_P2_SYS = (
    "You are a professional writer and editor. "
    "Using ONLY the facts provided in the structured list, write a coherent, "
    "fluent document of approximately {target_words} words. "
    "Do not invent any information not present in the list. "
    "Do not add introductory or concluding meta-sentences. "
    "Write in {lang}. "
    "Output only the final text — no title, no preamble."
)

def run_alpha(client, model_b_id: str, md_text: str,
              target_words: int, lang: str, extra_prompt: str) -> str:
    ctx       = words_to_tok(int(target_words * 3))   # generous for fact list
    p1_system = _ALPHA_P1_SYS
    if extra_prompt:
        p1_system += f' Focus particularly on: {extra_prompt}'

    log.info("Path Alpha — Pass 1: structured fact extraction…")
    facts = infer(
        client, model_b_id,
        messages=[
            {'role': 'system', 'content': p1_system},
            {'role': 'user',   'content': md_text},
        ],
        max_tokens=ctx, temperature=0.1,
        label='  Pass 1: extracting facts…',
    )
    log.info(f"  Facts extracted: {tok(facts)} tokens")

    p2_system = _ALPHA_P2_SYS.format(target_words=target_words, lang=lang)
    output_tokens = words_to_tok(int(target_words * 1.4))

    log.info("Path Alpha — Pass 2: synthesis…")
    result = infer(
        client, model_b_id,
        messages=[
            {'role': 'system', 'content': p2_system},
            {'role': 'user',   'content': facts},
        ],
        max_tokens=output_tokens, temperature=0.3, stream=True,
        label='  Pass 2: synthesizing…',
    )
    return strip_residue(result)


# ===========================================================================
# Path Beta — Map-Reduce-Refine  (Model A → Model B)
# ===========================================================================

_MAP_SYS = (
    "You are a neutral information extractor. "
    "Do not generate introductory or concluding sentences. "
    "Extract the essential facts, arguments, and conclusions from the text below. "
    "Do not invent information not present in the text. "
    "Target length: approximately {l_chunk} tokens. "
    'Reply ONLY with valid JSON: {{"partial_summary": "..."}} '
    "No text outside the JSON object."
)

_REDUCE_SYS = (
    "You are a neutral information extractor. "
    "Compress the following collection of partial summaries into a single coherent "
    "intermediate summary. Remove redundancies. Preserve all distinct facts. "
    "Target length: approximately {l_chunk} tokens. "
    'Reply ONLY with valid JSON: {{"partial_summary": "..."}} '
    "No text outside the JSON object."
)

_REFINE_SYS = (
    "You are a professional editor. "
    "Rewrite the draft below into a coherent, fluent document of exactly "
    "approximately {target_words} words. "
    "Remove all seams, redundancies, and list-like artefacts. "
    "Write in {lang}. "
    "Reply ONLY with the final text — no preamble, no title, no meta-commentary."
)


def _map_chunk(client, model_a_id: str, chunk: str, idx: int, total: int,
               l_chunk: int, extra_prompt: str) -> str:
    system = _MAP_SYS.format(l_chunk=l_chunk)
    if extra_prompt:
        system += f' Focus: {extra_prompt}'

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
        )
        parsed = parse_partial_summary(raw)
        if parsed is not None:
            log.info(
                f"  Chunk {idx+1}/{total}: {tok(chunk)}→{tok(parsed)} tokens | "
                f"{time.time()-t0:.1f}s | attempt {attempt}"
            )
            return parsed
        log.warning(f"  Chunk {idx+1}/{total} attempt {attempt}: JSON parse failed. "
                    f"Raw: {raw[:200]}")
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY ** attempt)

    log.error(f"  Chunk {idx+1}/{total}: all retries exhausted — using raw output.")
    return raw


def _reduce(client, model_a_id: str, model_a_ctx: int,
            partials: list[str], t_target: int, model_b_ctx: int,
            extra_prompt: str) -> str:
    """
    Concatenate partials. If too large for Model B's refine pass, run additional
    Map passes until it fits. Uses the same mathematical calibration as the map phase.
    """
    refine_budget = int(model_b_ctx * (1 - _BUFFER_RATIO)) - words_to_tok(int(tok_to_words(t_target) * 1.4))
    consolidated  = '\n\n'.join(partials)
    c_tok         = tok(consolidated)

    log.info(f"Reduce: {c_tok} tokens consolidated | Model B budget: {refine_budget} tokens")

    round_n = 0
    while c_tok > refine_budget:
        round_n += 1
        log.info(f"Reduce round {round_n}: still too large ({c_tok} > {refine_budget}) — re-mapping…")

        s_chunk   = int(model_a_ctx * _A_CONTENT_RATIO)
        blocks    = parse_md_blocks(consolidated)
        chunks    = build_chunks(blocks, s_chunk)
        n         = len(chunks)
        l_chunk   = min(int(model_a_ctx * _A_OUTPUT_RATIO),
                        max(64, int(t_target * 1.25 / n)))

        log.info(f"  Reduce round {round_n}: {n} sub-chunks | L_chunk={l_chunk} tokens")
        sub_partials = [
            _map_chunk(client, model_a_id, chunk, i, n, l_chunk, extra_prompt)
            for i, chunk in enumerate(chunks)
        ]
        consolidated = '\n\n'.join(sub_partials)
        c_tok        = tok(consolidated)

        if round_n >= 5:
            log.warning("Reduce: stopped after 5 rounds — proceeding with current size.")
            break

    log.info(f"Reduce complete: {c_tok} tokens → fits Model B context.")
    return consolidated


def run_beta(manager: ModelManager, client_a, md_text: str,
             target_words: int, lang: str, extra_prompt: str) -> str:
    model_a_id  = manager.model_id('a')
    model_a_ctx = manager.model_ctx('a')
    model_b_id  = manager.model_id('b')
    model_b_ctx = manager.model_ctx('b')
    t_target    = words_to_tok(target_words)

    # --- Chunk calibration -------------------------------------------
    # S_chunk: content budget = 60% of Model A context (20% prompt + 20% output)
    s_chunk = int(model_a_ctx * _A_CONTENT_RATIO)

    blocks = parse_md_blocks(md_text)
    chunks = build_chunks(blocks, s_chunk)
    n      = len(chunks)

    # L_chunk: per-chunk output target
    # L_chunk = (T_target × 1.25) / n, capped at 20% of model A context
    l_chunk = min(
        int(model_a_ctx * _A_OUTPUT_RATIO),
        max(64, int(t_target * 1.25 / n)),
    )

    log.info("─" * 60)
    log.info("PATH BETA — CHUNK CALIBRATION")
    log.info(f"  S_chunk (60% of {model_a_ctx})  = {s_chunk} tokens")
    log.info(f"  Total chunks                    = {n}")
    log.info(f"  L_chunk (T_target×1.25 / n)     = {l_chunk} tokens "
             f"(cap: {int(model_a_ctx * _A_OUTPUT_RATIO)})")
    log.info("─" * 60)

    # --- Phase 1: Map ---
    log.info(f"Phase 1 — Map ({n} chunks, Model A: {model_a_id})")
    partials = [
        _map_chunk(client_a, model_a_id, chunk, i, n, l_chunk, extra_prompt)
        for i, chunk in enumerate(chunks)
    ]

    # --- Phase 2: Reduce ---
    log.info("Phase 2 — Reduce")
    consolidated = _reduce(
        client_a, model_a_id, model_a_ctx,
        partials, t_target, model_b_ctx, extra_prompt,
    )

    # --- Phase 3: Refine (Model B) ---
    manager.ensure('b')
    client_b     = manager.client()
    output_tokens = words_to_tok(int(target_words * 1.4))
    system        = _REFINE_SYS.format(target_words=target_words, lang=lang)
    if extra_prompt:
        system += f' Focus: {extra_prompt}'

    log.info(f"Phase 3 — Refine (Model B: {model_b_id})")
    result = infer(
        client_b, model_b_id,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user',   'content': f'<draft_summary>\n{consolidated}\n</draft_summary>'},
        ],
        max_tokens=output_tokens, temperature=0.3, stream=True,
        label='  Refining…',
    )
    return strip_residue(result)


# ===========================================================================
# Validation & expansion guardrail
# ===========================================================================

_EXPAND_SYS = (
    "The following text is too brief. Expand it by adding more specific details, "
    "examples, and analytical depth based on its existing structure and content. "
    "Do not add information that is not already implied. "
    "Target: approximately {target_words} words. "
    "Write in {lang}. Output only the expanded text."
)

_CONDENSE_SYS = (
    "The following text is too long. Condense it to approximately {target_words} words "
    "while preserving all key points. "
    "Write in {lang}. Output only the condensed text."
)

def validate_and_adjust(client, model_b_id: str, result: str,
                        target_words: int, lang: str) -> str:
    t_target = words_to_tok(target_words)
    t_actual = tok(result)
    ratio    = t_actual / t_target if t_target > 0 else 1.0
    w_actual = tok_to_words(t_actual)

    log.info("─" * 60)
    log.info("VALIDATION")
    log.info(f"  Target:  {target_words} words (~{t_target} tokens)")
    log.info(f"  Actual:  {w_actual} words  ({t_actual} tokens)")
    log.info(f"  Ratio:   {ratio:.0%}  "
             f"(acceptable: {int(_VALIDATION_MIN_RATIO*100)}%–{int(_VALIDATION_MAX_RATIO*100)}%)")

    if ratio < _VALIDATION_MIN_RATIO:
        log.info("  → Output too short — triggering expansion pass…")
        system  = _EXPAND_SYS.format(target_words=target_words, lang=lang)
        result  = infer(
            client, model_b_id,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user',   'content': result},
            ],
            max_tokens=words_to_tok(int(target_words * 1.5)),
            temperature=0.3, stream=True, label='  Expanding…',
        )
        result = strip_residue(result)
        log.info(f"  After expansion: {tok_to_words(tok(result))} words")

    elif ratio > _VALIDATION_MAX_RATIO:
        log.info("  → Output too long — triggering condensation pass…")
        system = _CONDENSE_SYS.format(target_words=target_words, lang=lang)
        result = infer(
            client, model_b_id,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user',   'content': result},
            ],
            max_tokens=words_to_tok(int(target_words * 1.2)),
            temperature=0.2, stream=True, label='  Condensing…',
        )
        result = strip_residue(result)
        log.info(f"  After condensation: {tok_to_words(tok(result))} words")

    else:
        log.info("  → Within acceptable range — no adjustment needed.")

    log.info("─" * 60)
    return result


# ===========================================================================
# Orchestrator
# ===========================================================================

def run(md_text: str, manager: ModelManager, target_words: int, lang: str,
        extra_prompt: str, force_beta: bool) -> str:

    t_doc    = tok(md_text)
    t_target = words_to_tok(target_words)

    # --- Pivot decision ---
    path, _ = calculate_pivot(
        t_doc, t_target, manager.model_ctx('b'), force_beta
    )

    result: str

    if path == 'alpha':
        # Path Alpha: Model B only, two-pass in same context
        manager.ensure('b')
        client_b = manager.client()
        result   = run_alpha(
            client_b, manager.model_id('b'),
            md_text, target_words, lang, extra_prompt,
        )
        # Validation uses the same loaded Model B
        result = validate_and_adjust(
            client_b, manager.model_id('b'), result, target_words, lang
        )

    else:
        # Path Beta: Model A for map/reduce, Model B for refine
        manager.ensure('a')
        client_a = manager.client()
        result   = run_beta(
            manager, client_a, md_text, target_words, lang, extra_prompt
        )
        # Validation: Model B is already loaded after run_beta
        manager.ensure('b')
        client_b = manager.client()
        result   = validate_and_adjust(
            client_b, manager.model_id('b'), result, target_words, lang
        )

    return result


# ===========================================================================
# Main
# ===========================================================================

def main():
    load_env()

    # ------------------------------------------------------------------
    # Engine selection — read env vars for both engines up front so that
    # --engine on the CLI can switch without needing separate env sections.
    # ------------------------------------------------------------------
    _engine_env = os.environ.get('AI_ENGINE', 'lmstudio').lower()

    # LMStudio defaults
    _lm_model   = os.environ.get('LM_STUDIO_MODEL',       '')
    _lm_host    = os.environ.get('LM_STUDIO_HOST',        'http://localhost:1234')
    _lm_a       = os.environ.get('LM_STUDIO_MODEL_A',     _lm_model)
    _lm_a_ctx   = int(os.environ.get('LM_STUDIO_MODEL_A_CTX', '4096'))
    _lm_b       = os.environ.get('LM_STUDIO_MODEL_B',     _lm_model)
    _lm_b_ctx   = int(os.environ.get('LM_STUDIO_MODEL_B_CTX', '52500'))

    # Ollama defaults
    _ol_model   = os.environ.get('OLLAMA_MODEL',           '')
    _ol_host    = os.environ.get('OLLAMA_HOST',            'http://localhost:11434')
    _ol_a       = os.environ.get('OLLAMA_MODEL_A',         _ol_model)
    _ol_a_ctx   = int(os.environ.get('OLLAMA_MODEL_A_CTX', '4096'))
    _ol_b       = os.environ.get('OLLAMA_MODEL_B',         _ol_model)
    _ol_b_ctx   = int(os.environ.get('OLLAMA_MODEL_B_CTX', '8192'))

    parser = argparse.ArgumentParser(
        description='Resource-aware Markdown summarization — Path Alpha or Beta.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('md_file')
    parser.add_argument('-o', '--output',       help='Output file (default: <name>_summary.md)')
    parser.add_argument('--target-words', '-w', type=int, default=500,
                        help='Target word count (default: 500)')
    parser.add_argument('--prompt', '-p',       default='',
                        help='Extra focus instructions for all phases')
    parser.add_argument('--lang',               default='',
                        help='Output language override (e.g. "French")')
    parser.add_argument('--engine',             default=_engine_env,
                        choices=['lmstudio', 'ollama'],
                        help='AI engine backend (default from AI_ENGINE env var)')
    parser.add_argument('--host',               default='', metavar='URL',
                        help='API base URL (default: from engine env vars)')
    parser.add_argument('--model-a',            default='', metavar='ID',
                        help='Model A identifier (map phase; default: from env vars)')
    parser.add_argument('--model-a-ctx',        type=int, default=0, metavar='N',
                        help='Model A context window in tokens (default: from env vars)')
    parser.add_argument('--model-b',            default='', metavar='ID',
                        help='Model B identifier (refine phase; default: from env vars)')
    parser.add_argument('--model-b-ctx',        type=int, default=0, metavar='N',
                        help='Model B context window in tokens (default: from env vars)')
    parser.add_argument('--no-model-swap',      action='store_true',
                        help='Skip engine model lifecycle management')
    parser.add_argument('--force-beta',         action='store_true',
                        help='Always use Map-Reduce path regardless of document size')
    args = parser.parse_args()

    # Apply engine-specific defaults for any args not explicitly provided
    if args.engine == 'ollama':
        host        = args.host        or _ol_host
        model_a     = args.model_a     or _ol_a
        model_a_ctx = args.model_a_ctx or _ol_a_ctx
        model_b     = args.model_b     or _ol_b
        model_b_ctx = args.model_b_ctx or _ol_b_ctx
    else:
        host        = args.host        or _lm_host
        model_a     = args.model_a     or _lm_a
        model_a_ctx = args.model_a_ctx or _lm_a_ctx
        model_b     = args.model_b     or _lm_b
        model_b_ctx = args.model_b_ctx or _lm_b_ctx

    md_file = os.path.abspath(args.md_file)
    if not os.path.isfile(md_file):
        log.error(f"File not found: {md_file}"); sys.exit(1)

    try:
        import openai  # noqa: F401
    except ImportError:
        log.error("openai not installed. Run: pip install openai"); sys.exit(1)

    if not model_b:
        log.error(
            f"No model specified. Set "
            f"{'OLLAMA_MODEL_B' if args.engine == 'ollama' else 'LM_STUDIO_MODEL_B'} "
            f"(or {'OLLAMA_MODEL' if args.engine == 'ollama' else 'LM_STUDIO_MODEL'}) "
            f"in .env, or use --model-b."
        )
        sys.exit(1)

    # Single-model mode: no Model A → use Model B for all phases, no swap
    if not model_a:
        log.info("No Model A configured — using Model B for all phases (no swap).")
        model_a     = model_b
        model_a_ctx = model_b_ctx
        args.no_model_swap = True

    resolved_host = resolve_host(host)
    log.info(f"Engine: {args.engine}  |  Host: {resolved_host}")

    log.info(f"Reading {os.path.basename(md_file)}…")
    with open(md_file, encoding='utf-8') as f:
        md_text = f.read()

    if args.lang:
        lang = args.lang
        log.info(f"Language: {lang} (forced)")
    else:
        code, lang = detect_language(md_text)
        log.info(f"Language detected: {lang} ({code})")

    backend = create_backend(args.engine, resolved_host)
    manager = ModelManager(
        backend     = backend,
        model_a     = model_a,
        model_a_ctx = model_a_ctx,
        model_b     = model_b,
        model_b_ctx = model_b_ctx,
        no_swap     = args.no_model_swap,
    )

    log.info(f"Model A: {model_a} (ctx {model_a_ctx})")
    log.info(f"Model B: {model_b} (ctx {model_b_ctx})")
    log.info(f"Target:  {args.target_words} words")

    try:
        result = run(
            md_text      = md_text,
            manager      = manager,
            target_words = args.target_words,
            lang         = lang,
            extra_prompt = args.prompt,
            force_beta   = args.force_beta,
        )
    finally:
        manager.release()

    out_path = os.path.abspath(args.output) if args.output else \
               os.path.splitext(md_file)[0] + '_summary.md'

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(result)

    w = tok_to_words(tok(result))
    log.info(f"Saved: {out_path}  ({w} words)")


if __name__ == '__main__':
    main()
