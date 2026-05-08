#!/usr/bin/env python3
"""
transform_md.py — AI document transformation CLI.

Applies a transformation template to a Markdown document using a local AI
backend (LMStudio or Ollama). The Map-Reduce-Refine pipeline handles documents
of any size automatically.

Available templates:
  summary  — Concise factual summary preserving key insights
  podcast  — Spoken-word transcript for audio/TTS synthesis, free of footnotes,
              citations, figure references, and other visual-only artefacts

New templates can be added by defining a TransformTemplate in transform_pipeline.py
and registering it in TEMPLATES — no other changes needed.

.env configuration:
  AI_ENGINE=lmstudio          # or 'ollama'

  # LMStudio
  LM_STUDIO_HOST=http://localhost:1234
  LM_STUDIO_MODEL_A=<map model id>
  LM_STUDIO_MODEL_A_CTX=4096
  LM_STUDIO_MODEL_B=<refine model id>
  LM_STUDIO_MODEL_B_CTX=52500

  # Ollama
  OLLAMA_HOST=http://localhost:11434
  OLLAMA_MODEL_A=model:tag
  OLLAMA_MODEL_A_CTX=4096
  OLLAMA_MODEL_B=model:tag
  OLLAMA_MODEL_B_CTX=8192

Usage:
  python transform_md.py doc.md --template summary
  python transform_md.py doc.md --template podcast --target-words 2000
  python transform_md.py doc.md --template podcast --lang French -o out.md
  python transform_md.py doc.md --no-model-swap    # models are already loaded
  python transform_md.py doc.md --force-beta       # always use Map-Reduce path
"""

import sys, os, argparse

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import create_backend, resolve_host       # noqa: E402
from transform_pipeline import (                      # noqa: E402
    TEMPLATES, ModelManager, run, detect_language,
    tok, tok_to_words, log,
)
from ai_common import load_env, parse_num_gpu         # noqa: E402


def main():
    load_env()

    _engine_env = os.environ.get('AI_ENGINE', 'lmstudio').lower()

    # LMStudio env vars
    _lm_model   = os.environ.get('LM_STUDIO_MODEL',        '')
    _lm_host    = os.environ.get('LM_STUDIO_HOST',         'http://localhost:1234')
    _lm_a       = os.environ.get('LM_STUDIO_MODEL_A',      _lm_model)
    _lm_a_ctx   = int(os.environ.get('LM_STUDIO_MODEL_A_CTX', '4096'))
    _lm_b       = os.environ.get('LM_STUDIO_MODEL_B',      _lm_model)
    _lm_b_ctx   = int(os.environ.get('LM_STUDIO_MODEL_B_CTX', '52500'))

    # Ollama env vars
    _ol_model   = os.environ.get('OLLAMA_MODEL',            '')
    _ol_host    = os.environ.get('OLLAMA_HOST',             'http://localhost:11434')
    _ol_a       = os.environ.get('OLLAMA_MODEL_A',          _ol_model)
    _ol_a_ctx   = int(os.environ.get('OLLAMA_MODEL_A_CTX',  '4096'))
    _ol_b       = os.environ.get('OLLAMA_MODEL_B',          _ol_model)
    _ol_b_ctx   = int(os.environ.get('OLLAMA_MODEL_B_CTX',  '8192'))

    _ol_ng_default = parse_num_gpu(os.environ.get('OLLAMA_NUM_GPU', ''))
    _ol_ng_a = (parse_num_gpu(os.environ.get('OLLAMA_NUM_GPU_A', ''))
                if os.environ.get('OLLAMA_NUM_GPU_A') else _ol_ng_default)
    _ol_ng_b = (parse_num_gpu(os.environ.get('OLLAMA_NUM_GPU_B', ''))
                if os.environ.get('OLLAMA_NUM_GPU_B') else _ol_ng_default)

    parser = argparse.ArgumentParser(
        description='AI document transformation — Map-Reduce-Refine pipeline.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join(
            f'  {t.id:<10} {t.description}' for t in TEMPLATES.values()
        ),
    )
    parser.add_argument('md_file')
    parser.add_argument('-o', '--output',
                        help='Output file (default: <name>_<template>.md)')
    parser.add_argument('--template', '-t',
                        default='summary', choices=list(TEMPLATES),
                        help='Transformation template (default: summary)')
    parser.add_argument('--target-words', '-w',
                        type=int, default=500,
                        help='Target word count (default: 500)')
    parser.add_argument('--same-length', action='store_true',
                        help='Set target word count equal to the source document length '
                             '(content-preserving transformation — no compression)')
    parser.add_argument('--prompt', '-p',
                        default='',
                        help='Extra focus instructions passed to all phases')
    parser.add_argument('--lang',
                        default='',
                        help='Output language override (e.g. "French")')
    parser.add_argument('--engine',
                        default=_engine_env, choices=['lmstudio', 'ollama'],
                        help='AI engine backend (default from AI_ENGINE env var)')
    parser.add_argument('--host',        default='', metavar='URL',
                        help='API base URL (default: from engine env vars)')
    parser.add_argument('--model-a',     default='', metavar='ID',
                        help='Model A identifier — map/extraction phase')
    parser.add_argument('--model-a-ctx', type=int, default=0, metavar='N',
                        help='Model A context window in tokens')
    parser.add_argument('--model-b',     default='', metavar='ID',
                        help='Model B identifier — refine/synthesis phase')
    parser.add_argument('--model-b-ctx', type=int, default=0, metavar='N',
                        help='Model B context window in tokens')
    parser.add_argument('--no-model-swap', action='store_true',
                        help='Skip engine model lifecycle management')
    parser.add_argument('--force-beta',    action='store_true',
                        help='Always use Map-Reduce path regardless of document size')
    args = parser.parse_args()

    # Apply engine-specific defaults for args not set on the command line
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
        env_key = 'OLLAMA_MODEL_B' if args.engine == 'ollama' else 'LM_STUDIO_MODEL_B'
        log.error(f"No model specified. Set {env_key} in .env or use --model-b.")
        sys.exit(1)

    # Single-model mode: no Model A → Model B handles all phases, no swap needed
    if not model_a:
        log.info("No Model A configured — using Model B for all phases (no swap).")
        model_a, model_a_ctx, args.no_model_swap = model_b, model_b_ctx, True

    template      = TEMPLATES[args.template]
    resolved_host = resolve_host(host)

    log.info(f"Engine:   {args.engine}  |  Host: {resolved_host}")
    log.info(f"Template: {template.label}")
    log.info(f"Model A:  {model_a} (ctx {model_a_ctx})")
    log.info(f"Model B:  {model_b} (ctx {model_b_ctx})")
    log.info(f"Target:   {args.target_words} words")

    with open(md_file, encoding='utf-8') as f:
        md_text = f.read()

    if args.same_length:
        args.target_words = tok_to_words(tok(md_text))
        log.info(f"Same-length mode: target = {args.target_words} words (source document length)")

    if args.lang:
        lang = args.lang
        log.info(f"Language: {lang} (forced)")
    else:
        code, lang = detect_language(md_text)
        log.info(f"Language detected: {lang} ({code})")

    num_gpu_a = _ol_ng_a if args.engine == 'ollama' else None
    num_gpu_b = _ol_ng_b if args.engine == 'ollama' else None

    backend = create_backend(args.engine, resolved_host)
    manager = ModelManager(
        backend      = backend,
        model_a      = model_a,
        model_a_ctx  = model_a_ctx,
        model_b      = model_b,
        model_b_ctx  = model_b_ctx,
        no_swap      = args.no_model_swap,
        num_gpu_a    = num_gpu_a,
        num_gpu_b    = num_gpu_b,
    )

    try:
        result = run(
            md_text      = md_text,
            manager      = manager,
            target_words = args.target_words,
            lang         = lang,
            extra_prompt = args.prompt,
            template     = template,
            force_beta   = args.force_beta,
        )
    finally:
        manager.release()

    out_path = (os.path.abspath(args.output) if args.output
                else os.path.splitext(md_file)[0] + f'_{template.id}.md')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(result)

    log.info(f"Saved: {out_path}  ({tok_to_words(tok(result))} words)")


if __name__ == '__main__':
    main()
