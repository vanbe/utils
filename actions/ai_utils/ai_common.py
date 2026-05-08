#!/usr/bin/env python3
"""
ai_common.py — Shared utilities for all ai_utils scripts.

Provides:
  load_env()                              load project .env into os.environ
  parse_num_gpu(val)                      parse OLLAMA_NUM_GPU_* env var string
  get_model_info(backend, override)       resolve model id + context window
  ensure_model_loaded(backend, ...)       full model lifecycle with ANSI output

  ANSI constants: SEP, R, B, CYN, GRN, YEL, RED, BLU, GRY, DIM
  Display helpers: step(label), ok(t0, detail), elapsed(t0)
"""

import os
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env() -> None:
    """Load .env from project root into os.environ (setdefault — no overwrite)."""
    env_file = os.path.join(_PROJECT_ROOT, '.env')
    if not os.path.isfile(env_file):
        return
    with open(env_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            os.environ.setdefault(key.strip(), value.strip().strip('"\''))


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

SEP = '─' * 56
R   = '\033[0m'
B   = '\033[1m'
CYN = '\033[36m'
GRN = '\033[32m'
YEL = '\033[33m'
RED = '\033[31m'
BLU = '\033[34m'
GRY = '\033[90m'
DIM = '\033[2m'


def elapsed(t0: float) -> str:
    e = time.time() - t0
    return f"{int(e // 60)}m {int(e % 60):02d}s" if e >= 60 else f"{e:.1f}s"


def step(label: str) -> float:
    """Print a step label and return start time."""
    print(f"  {label:<38}", end='', flush=True)
    return time.time()


def ok(t0: float, detail: str = '') -> None:
    """Print a checkmark with elapsed time and optional detail."""
    d = f"   {detail}" if detail else ''
    print(f"  ✓  {elapsed(t0)}{d}")


# ---------------------------------------------------------------------------
# Env var helpers
# ---------------------------------------------------------------------------

def parse_num_gpu(val: str) -> 'int | None':
    """Parse OLLAMA_NUM_GPU_* env var string to int, or None if empty/invalid."""
    try:
        return int(val) if val and val.strip() else None
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------

def get_model_info(backend, model_override: str) -> 'tuple[str, int]':
    """
    Resolve the active model id and context window size.

    If model_override is given, use it (look it up in the available list for
    the context size). Otherwise, use the first available model.
    Falls back to 8192 tokens if the context size cannot be determined.
    """
    _DEFAULT_CTX = 8192
    models = backend.list_available()
    if not models:
        model_id = model_override or 'unknown'
        ctx = backend.get_ctx(model_id) if model_override else None
        return model_id, ctx or _DEFAULT_CTX
    if model_override:
        chosen = next((m for m in models if m['id'] == model_override), None)
        if chosen is None:
            chosen = {'id': model_override, 'ctx': None}
    else:
        chosen = models[0]
    ctx = chosen.get('ctx') or backend.get_ctx(chosen['id']) or _DEFAULT_CTX
    return chosen['id'], int(ctx)


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def ensure_model_loaded(backend, model_id: str, ctx: int,
                        num_gpu: 'int | None' = None) -> None:
    """
    Ensure model_id is loaded in VRAM with the given context and GPU offload.

    - Already active with matching config → "already active", no reload.
    - Already active but ctx or num_gpu differ → unload + reload with correct config.
    - Different model active → unload it, then load the requested one.
    - Nothing loaded → load directly.
    - Progress is reported via step/ok ANSI helpers.

    num_gpu: GPU layers to offload (Ollama only; None = backend default).
    """
    current = backend.get_loaded()

    if current == model_id:
        # Verify the running model has the required configuration.
        mismatches = []
        loaded_ctx = (backend.get_loaded_ctx()
                      if hasattr(backend, 'get_loaded_ctx') else None)
        loaded_gpu = (backend.get_loaded_num_gpu()
                      if hasattr(backend, 'get_loaded_num_gpu') else None)

        if loaded_ctx is not None and loaded_ctx != ctx:
            mismatches.append(f"ctx {loaded_ctx} → {ctx}")
        if num_gpu is not None and loaded_gpu is not None and loaded_gpu != num_gpu:
            mismatches.append(f"num_gpu {loaded_gpu} → {num_gpu}")

        if not mismatches:
            t = step("Load model")
            ok(t, "already active")
            return

        # Config mismatch — unload and reload with the correct settings.
        mismatch_str = ', '.join(mismatches)
        t = step(f"Unload {current}")
        backend.unload(current)
        ok(t, f"config mismatch ({mismatch_str}) — reloading")

    elif current:
        t = step(f"Unload {current}")
        backend.unload(current)
        ok(t, "unloaded")

    gpu_note = f"  (num_gpu={num_gpu})" if num_gpu is not None else ''
    t = step(f"Load model  ctx={ctx}{gpu_note}")
    backend.load(model_id, ctx, num_gpu=num_gpu)
    backend.wait_ready(model_id)
    ok(t, "ready")
