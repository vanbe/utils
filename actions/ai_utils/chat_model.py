#!/usr/bin/env python3
"""
chat_model.py — Interactive terminal chat with a local AI model.

Conversation is saved to a Markdown file in real-time as tokens stream in.
Each user message and model reply is appended immediately — safe to read
while the chat is in progress.

Commands during chat:
  ?        show help
  bye      exit chat and unload the model
  Ctrl+C   interrupt current reply (stays in chat), Ctrl+C again to exit

Multi-line input: end a line with \\ to continue on the next line.

Usage:
  python chat_model.py [folder] --quality b --output "path/to/chat.md"
"""

import sys
import os
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import create_backend, resolve_host  # noqa: E402
from ai_common import (                          # noqa: E402
    load_env, parse_num_gpu, get_model_info, ensure_model_loaded,
    step as _step, ok as _ok, elapsed as _elapsed,
    SEP as _SEP, R as _R, B as _B, CYN as _CYN, GRN as _GRN,
    YEL as _YEL, RED as _RED, BLU as _BLU, GRY as _GRY, DIM as _DIM,
)


# ---------------------------------------------------------------------------
# Markdown file helpers
# ---------------------------------------------------------------------------

def _write_header(out_path: str, folder: str, model: str, engine: str, quality: str) -> None:
    """Write YAML front-matter and document title."""
    now = datetime.now()
    quality_label = 'A — best quality' if quality == 'a' else 'B — balanced'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('---\n')
        f.write(f'date:    {now.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'engine:  {engine}\n')
        f.write(f'model:   {model}\n')
        f.write(f'quality: {quality_label}\n')
        f.write(f'folder:  {folder}\n')
        f.write('---\n\n')
        f.write(f'# Chat — {now.strftime("%Y-%m-%d")}\n\n')


def _write_user(out_path: str, text: str) -> None:
    ts = datetime.now().strftime('%H:%M:%S')
    with open(out_path, 'a', encoding='utf-8') as f:
        f.write(f'---\n\n**You** · *{ts}*\n\n{text.strip()}\n\n')


def _write_assistant_open(out_path: str) -> None:
    ts = datetime.now().strftime('%H:%M:%S')
    with open(out_path, 'a', encoding='utf-8') as f:
        f.write(f'---\n\n**Assistant** · *{ts}*\n\n')


def _append(out_path: str, chunk: str) -> None:
    with open(out_path, 'a', encoding='utf-8') as f:
        f.write(chunk)


def _write_assistant_close(out_path: str) -> None:
    with open(out_path, 'a', encoding='utf-8') as f:
        f.write('\n\n')


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

_HELP = f"""
  {_BLU}┌────────────────────────────────────────────────┐{_R}
  {_BLU}│{_R}  {_B}Chat commands{_R}                                  {_BLU}│{_R}
  {_BLU}│{_R}                                                {_BLU}│{_R}
  {_BLU}│{_R}  {_CYN}?{_R}          show this help                   {_BLU}│{_R}
  {_BLU}│{_R}  {_CYN}bye{_R}        exit chat and unload model       {_BLU}│{_R}
  {_BLU}│{_R}  {_CYN}Ctrl+C{_R}     interrupt current reply          {_BLU}│{_R}
  {_BLU}│{_R}                                                {_BLU}│{_R}
  {_BLU}│{_R}  End a line with {_CYN}\\{_R} to continue on the       {_BLU}│{_R}
  {_BLU}│{_R}  next line (multi-line input).               {_BLU}│{_R}
  {_BLU}└────────────────────────────────────────────────┘{_R}
"""

_DEFAULT_SYSTEM = (
    "You are a helpful, knowledgeable assistant. "
    "Answer clearly and concisely. "
    "Use Markdown formatting where it aids readability "
    "(headings, bullet lists, code blocks, bold for key terms)."
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_env()

    _engine_env  = os.environ.get('AI_ENGINE',             'lmstudio').lower()
    _lm_host_env = os.environ.get('LM_STUDIO_HOST',        'http://localhost:1234')
    _lm_fallback = os.environ.get('LM_STUDIO_MODEL',       '')
    _lm_a        = os.environ.get('LM_STUDIO_MODEL_A',     _lm_fallback)
    _lm_a_ctx    = int(os.environ.get('LM_STUDIO_MODEL_A_CTX', '0'))
    _lm_b        = os.environ.get('LM_STUDIO_MODEL_B',     _lm_fallback)
    _lm_b_ctx    = int(os.environ.get('LM_STUDIO_MODEL_B_CTX', '0'))
    _ol_host_env = os.environ.get('OLLAMA_HOST',           'http://localhost:11434')
    _ol_fallback = os.environ.get('OLLAMA_MODEL',          '')
    _ol_a        = os.environ.get('OLLAMA_MODEL_A',        _ol_fallback)
    _ol_a_ctx    = int(os.environ.get('OLLAMA_MODEL_A_CTX', '0'))
    _ol_b        = os.environ.get('OLLAMA_MODEL_B',        _ol_fallback)
    _ol_b_ctx    = int(os.environ.get('OLLAMA_MODEL_B_CTX', '0'))

    _ol_ng_default = parse_num_gpu(os.environ.get('OLLAMA_NUM_GPU', ''))
    _ol_ng_a = (parse_num_gpu(os.environ.get('OLLAMA_NUM_GPU_A', ''))
                if os.environ.get('OLLAMA_NUM_GPU_A') else _ol_ng_default)
    _ol_ng_b = (parse_num_gpu(os.environ.get('OLLAMA_NUM_GPU_B', ''))
                if os.environ.get('OLLAMA_NUM_GPU_B') else _ol_ng_default)

    parser = argparse.ArgumentParser(
        description='Interactive chat with a local AI model — saved to Markdown.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('folder', nargs='?', default=os.getcwd(),
                        help='Working folder for the output file (default: cwd)')
    parser.add_argument('--quality', choices=['a', 'b'], default='b',
                        help='Model quality: a = MODEL_A (best), b = MODEL_B (default)')
    parser.add_argument('-o', '--output', default='',
                        help='Output Markdown file (default: YYYY-MM-DD - Chat.md in folder)')
    parser.add_argument('--engine', default=_engine_env, choices=['lmstudio', 'ollama'])
    parser.add_argument('--host', default='')
    parser.add_argument('--model', default='')
    parser.add_argument('--context-size', type=int, default=0)
    parser.add_argument('--system', default=_DEFAULT_SYSTEM,
                        help='System prompt (default: general helpful assistant)')
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)

    if args.engine == 'ollama':
        host_default  = args.host or _ol_host_env
        quality_model = _ol_a if args.quality == 'a' else _ol_b
        ctx_env       = _ol_a_ctx if args.quality == 'a' else _ol_b_ctx
        model_default = args.model or quality_model
        num_gpu       = _ol_ng_a if args.quality == 'a' else _ol_ng_b
    else:
        host_default  = args.host or _lm_host_env
        quality_model = _lm_a if args.quality == 'a' else _lm_b
        ctx_env       = _lm_a_ctx if args.quality == 'a' else _lm_b_ctx
        model_default = args.model or quality_model
        num_gpu       = None

    out_path = (os.path.abspath(args.output) if args.output
                else os.path.join(folder, datetime.now().strftime('%Y-%m-%d') + ' - Chat.md'))

    resolved_host = resolve_host(host_default)
    backend = create_backend(args.engine, resolved_host)
    client  = backend.create_client()

    model, context_window = get_model_info(backend, model_default)
    if ctx_env:            # .env OLLAMA_MODEL_A_CTX / OLLAMA_MODEL_B_CTX
        context_window = ctx_env
    if args.context_size:  # --context-size CLI flag takes final precedence
        context_window = args.context_size

    quality_label = 'A — best quality' if args.quality == 'a' else 'B — balanced'

    # ── Header ───────────────────────────────────────────────────────────────
    print(_SEP)
    print(f"  {_B}Chat{_R}  ·  {_CYN}{model}{_R}")
    print(f"  engine : {args.engine}  ·  quality: {quality_label}")
    print(f"  output : {os.path.basename(out_path)}")
    print(_SEP)

    # ── Model lifecycle ───────────────────────────────────────────────────────
    ensure_model_loaded(backend, model, context_window, num_gpu=num_gpu)

    # ── Write file header ─────────────────────────────────────────────────────
    _write_header(out_path, folder, model, args.engine, args.quality)

    # ── Conversation state ────────────────────────────────────────────────────
    messages: list[dict] = [{"role": "system", "content": args.system}]

    print()
    print(f"  {_GRN}Ready.{_R}  "
          f"Type {_CYN}?{_R} for help · "
          f"{_CYN}bye{_R} to exit · "
          f"end line with {_CYN}\\{_R} for multi-line")
    print()

    # ── Chat loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            # -- Input --------------------------------------------------------
            print(f"{_BLU}{_B}You ›{_R} ", end='', flush=True)
            try:
                raw = input('')
            except EOFError:
                break

            cmd = raw.strip().lower()
            if cmd == 'bye':
                break
            if cmd == '?':
                print(_HELP)
                continue
            if not raw.strip():
                continue

            # Multi-line continuation: lines ending with backslash
            lines = [raw]
            while lines[-1].rstrip().endswith('\\'):
                lines[-1] = lines[-1].rstrip()[:-1]   # remove backslash
                print(f"{_BLU}  ›{_R} ", end='', flush=True)
                try:
                    lines.append(input(''))
                except EOFError:
                    break

            user_text = '\n'.join(lines).strip()
            if not user_text:
                continue

            # Write prompt to file, add to history
            _write_user(out_path, user_text)
            messages.append({"role": "user", "content": user_text})

            # -- Stream response ----------------------------------------------
            print(f"\n{_DIM}{model}{_R}\n", flush=True)
            _write_assistant_open(out_path)

            t_resp    = time.time()
            full_text = ''
            interrupted = False

            try:
                kwargs: dict = dict(
                    model=model,
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=4096,
                )
                eb = backend.get_inference_extra_body()
                if eb:
                    kwargs['extra_body'] = eb
                stream = client.chat.completions.create(**kwargs)
                for chunk in stream:
                    content = (chunk.choices[0].delta.content or '') if chunk.choices else ''
                    if content:
                        print(content, end='', flush=True)
                        _append(out_path, content)
                        full_text += content

            except KeyboardInterrupt:
                interrupted = True
                note = '\n\n*[reply interrupted by user]*'
                print(f"\n  {_YEL}[interrupted]{_R}")
                _append(out_path, note)
                full_text += note

            except Exception as exc:
                note = f'\n\n*[Error: {exc}]*'
                print(f"\n  {_RED}Error: {exc}{_R}")
                _append(out_path, note)
                full_text += note

            _write_assistant_close(out_path)
            messages.append({"role": "assistant", "content": full_text})
            print(f"\n\n  {_GRY}({_elapsed(time.time() - t_resp)}){_R}\n")

            if interrupted:
                # Single Ctrl+C interrupts the reply; second Ctrl+C (main loop) exits
                pass

    except KeyboardInterrupt:
        pass

    # ── Unload model ──────────────────────────────────────────────────────────
    print()
    t = _step("Unload model")
    backend.unload(model)
    _ok(t, "unloaded")

    # ── Summary ───────────────────────────────────────────────────────────────
    turns = sum(1 for m in messages if m['role'] == 'user')
    print()
    print(_SEP)
    print(f"  Turns:  {turns}")
    print(f"  Saved:  {os.path.basename(out_path)}")
    print(_SEP)


if __name__ == '__main__':
    main()
