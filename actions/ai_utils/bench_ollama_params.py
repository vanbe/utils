#!/usr/bin/env python3
"""
bench_ollama_params.py — Sweep Ollama (num_gpu, num_ctx) options to find the
best combination for a given model + workload on this machine.

For each combination the script:
  1. Unloads the model.
  2. Reloads it with the params (keep_alive=-1).
  3. Captures /api/ps (size, size_vram → % on GPU) and nvidia-smi dedicated VRAM.
  4. Runs ONE /api/chat inference against the document, capturing:
       - TTFT          time to first streamed token  (wall clock)
       - prompt t/s    Ollama's prompt_eval_count / prompt_eval_duration
       - decode t/s    Ollama's eval_count / eval_duration  (the metric you
                       usually care about — steady-state generation speed)
  5. Unloads, then proceeds to the next combination.

Why every metric matters together:
  - "100% GPU" in `ollama ps` only means Ollama placed all layers on GPU. If
    dedicated VRAM is exhausted, the driver pages into shared GPU memory
    (Windows DX shared memory / Linux GTT). The run still completes but TTFT
    explodes and decode rate collapses. Always read TTFT and decode rate
    alongside the "% GPU" figure before declaring a config "fits".
  - On WSL2, nvidia-smi shows dedicated VRAM only. Cross-reference Windows
    Task Manager (Performance → GPU → "Shared GPU memory") to confirm paging.

Usage:
  python actions/ai_utils/bench_ollama_params.py "test/pdf to test_merged_mineru.md" \\
      --model gemma4:e4b --num-gpu 38,42,44,46 --num-ctx 8192 --max-tokens 200

Output:
  Live per-config row + a summary table sorted by decode tok/s. Pass --output
  PATH.md to save the table for later comparison.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434').rstrip('/')


# ---------------------------------------------------------------------------
# Tiny Ollama HTTP wrappers (stdlib only — keeps the script self-contained)
# ---------------------------------------------------------------------------

def _post(path: str, payload: dict, timeout: float = 600.0):
    req = urllib.request.Request(
        f'{OLLAMA_HOST}{path}',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _get(path: str, timeout: float = 30.0):
    req = urllib.request.Request(f'{OLLAMA_HOST}{path}', method='GET')
    return urllib.request.urlopen(req, timeout=timeout)


def unload(model: str) -> None:
    try:
        with _post('/api/generate',
                   {'model': model, 'prompt': '', 'keep_alive': 0}) as r:
            r.read()
    except Exception as e:
        print(f'  warn: unload failed: {e}', file=sys.stderr)


def preload(model: str, options: dict, timeout: float = 600.0) -> None:
    payload = {'model': model, 'prompt': '', 'keep_alive': -1, 'options': options}
    with _post('/api/generate', payload, timeout=timeout) as r:
        r.read()


def get_ps() -> list:
    with _get('/api/ps') as r:
        return json.loads(r.read()).get('models', [])


def get_vram_mb():
    """(used_mb, total_mb) for GPU 0 via nvidia-smi, or None if unavailable.

    Note: on WSL2 this is dedicated VRAM only; shared GPU memory paging is
    invisible here — infer it from a TTFT explosion at "100% GPU".
    """
    for nv in ('/usr/lib/wsl/lib/nvidia-smi', 'nvidia-smi'):
        try:
            out = subprocess.check_output(
                [nv, '--query-gpu=memory.used,memory.total',
                 '--format=csv,noheader,nounits', '-i', '0'],
                text=True, timeout=10,
            ).strip().split(',')
            return int(out[0].strip()), int(out[1].strip())
        except (FileNotFoundError, subprocess.SubprocessError, ValueError):
            continue
    return None


def run_inference(model: str, prompt: str, options: dict, max_tokens: int) -> dict:
    """One /api/chat streaming call; returns timing dict (or {'error': ...})."""
    payload = {
        'model':      model,
        'messages':   [{'role': 'user', 'content': prompt}],
        'options':    {**options, 'num_predict': max_tokens},
        'keep_alive': -1,
        'stream':     True,
    }
    t_send = time.time()
    ttft = None
    last: dict = {}
    try:
        with _post('/api/chat', payload, timeout=900.0) as r:
            for raw in r:
                line = raw.decode('utf-8', errors='replace').strip()
                if not line:
                    continue
                obj = json.loads(line)
                if ttft is None and isinstance(obj.get('message'), dict):
                    ttft = time.time() - t_send
                last = obj
                if obj.get('done'):
                    break
    except Exception as e:
        return {'error': str(e), 'ttft': ttft}
    wall  = time.time() - t_send
    pe_ns = last.get('prompt_eval_duration') or 0
    pe_n  = last.get('prompt_eval_count')    or 0
    ev_ns = last.get('eval_duration')        or 0
    ev_n  = last.get('eval_count')           or 0
    return {
        'ttft':        ttft,
        'wall_s':      wall,
        'prompt_tok':  pe_n,
        'prompt_rate': (pe_n / (pe_ns / 1e9)) if pe_ns else 0.0,
        'output_tok':  ev_n,
        'decode_rate': (ev_n / (ev_ns / 1e9)) if ev_ns else 0.0,
    }


# ---------------------------------------------------------------------------
# Sweep + reporting
# ---------------------------------------------------------------------------

def parse_int_list(s: str) -> list:
    return [int(x) for x in s.split(',') if x.strip()]


def _fmt(v, spec: str = '', na: str = '—') -> str:
    if v is None:
        return na
    return f'{v:{spec}}' if spec else str(v)


# ---------------------------------------------------------------------------
# Auto-tune: probe model + adaptive sweep
# ---------------------------------------------------------------------------
# Pure heuristics from "model size + VRAM" don't reliably predict the paging
# cliff (e.g. on this hardware gemma4:e4b at num_gpu=44 looks like it should
# fit by raw math but in fact pages and drops to 2 t/s). So auto-tune still
# measures — it just picks smart starting points and refines around the best.

def probe_model(model: str) -> dict:
    """GET /api/show → architecture details, layer count, model max ctx."""
    try:
        with _post('/api/show', {'name': model}, timeout=30.0) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f'  warn: /api/show failed: {e}', file=sys.stderr)
        return {}
    info = data.get('model_info') or {}
    n_layers = next((int(v) for k, v in info.items()
                     if k.endswith('.block_count')), None)
    max_ctx  = next((int(v) for k, v in info.items()
                     if k.endswith('.context_length')), None)
    return {
        'arch':       info.get('general.architecture'),
        'size_bytes': data.get('size'),
        'num_layers': n_layers,
        'max_ctx':    max_ctx,
    }


def coarse_grid(probe: dict, vram_total_mb: int) -> list:
    """Heuristic starting grid: bracket the predicted "fit" with ±3·stride
    and always include num_layers (to expose paging) plus 0 (CPU baseline)."""
    n_layers   = probe.get('num_layers')  or 32
    size_bytes = probe.get('size_bytes')  or 0
    layer_mb   = (size_bytes / 1024 / 1024) / n_layers if (size_bytes and n_layers) else 0
    # 80 % of dedicated VRAM minus a 700 MB system reserve. The 80 % factor
    # absorbs KV-cache + activation overhead the weights-only math ignores.
    usable = max(0, int(vram_total_mb * 0.80) - 700)
    fit    = int(usable / layer_mb) if layer_mb else n_layers
    fit    = max(0, min(n_layers, fit))
    stride = max(2, n_layers // 12)
    cands = {n_layers, 0, fit}
    for ng in range(max(0, fit - 3 * stride), min(n_layers, fit + 3 * stride) + 1, stride):
        cands.add(ng)
    return sorted(cands)


def fine_grid(best_ng: int, n_layers: int, already: set) -> list:
    """±2 layers around the coarse winner, skipping already-tested values."""
    return [ng for ng in range(max(0, best_ng - 2), min(n_layers, best_ng + 2) + 1)
            if ng not in already]


def _is_viable(r: dict) -> bool:
    """A run is "viable" if it's not paging: TTFT < 5s and decode > 3 t/s."""
    return ('error' not in r
            and (r.get('ttft') or 999) < 5.0
            and r.get('decode_rate', 0) > 3.0)


def _run_one(model: str, doc: str, ng: int, ctx: int, max_tokens: int,
             idx: int, total: int) -> dict:
    """Unload → load(opts) → snapshot ps + nvidia-smi → run inference → row."""
    opts = {'num_gpu': ng, 'num_ctx': ctx}
    print(f'\n[{idx}/{total}]  num_gpu={ng}  num_ctx={ctx}')
    unload(model); time.sleep(1.0)
    t_load = time.time()
    try:
        preload(model, opts)
    except urllib.error.HTTPError as e:
        print(f'  ✗ load HTTP {e.code} — {e.reason}')
        return {'num_gpu': ng, 'num_ctx': ctx, 'error': f'load HTTP {e.code}'}
    except Exception as e:
        print(f'  ✗ load failed: {e}')
        return {'num_gpu': ng, 'num_ctx': ctx, 'error': f'load: {e}'}
    load_s = time.time() - t_load

    time.sleep(0.4)
    ps = get_ps()
    m  = next((x for x in ps
               if x.get('name', '').startswith(model)
               or x.get('model', '').startswith(model)), None)
    size_mb      = (m.get('size', 0)      // (1024 * 1024)) if m else 0
    size_vram_mb = (m.get('size_vram', 0) // (1024 * 1024)) if m else 0
    gpu_pct      = (size_vram_mb / size_mb * 100) if size_mb else 0.0
    loaded_ctx   = (m or {}).get('context_length')
    vram         = get_vram_mb()
    vram_str     = f'{vram[0]} / {vram[1]} MB' if vram else '—'

    print(f'  load: {load_s:.1f}s  ·  ollama ps: {size_mb} MB total, '
          f'{size_vram_mb} MB on GPU ({gpu_pct:.0f}% GPU)'
          + (f', ctx={loaded_ctx}' if loaded_ctx else ''))
    print(f'  nvidia-smi VRAM: {vram_str}')

    bench = run_inference(model, doc, opts, max_tokens)
    base = {
        'num_gpu':       ng,
        'num_ctx':       ctx,
        'gpu_pct':       gpu_pct,
        'size_mb':       size_mb,
        'size_vram_mb':  size_vram_mb,
        'vram_used_mb':  vram[0] if vram else None,
        'vram_total_mb': vram[1] if vram else None,
        'load_s':        load_s,
    }
    if 'error' in bench:
        print(f'  ✗ inference error: {bench["error"]}')
        return {**base, 'error': bench['error']}
    print(f'  TTFT: {_fmt(bench["ttft"], ".2f")}s  ·  '
          f'prompt: {bench["prompt_tok"]} tok @ {bench["prompt_rate"]:.0f} t/s  ·  '
          f'decode: {bench["output_tok"]} tok @ {bench["decode_rate"]:.1f} t/s  ·  '
          f'wall: {bench["wall_s"]:.1f}s')
    return {**base, **bench}


def build_markdown(model: str, doc_path: str, doc_chars: int, max_tokens: int,
                   rows_ok: list, rows_err: list) -> str:
    out = []
    out.append(f'# Ollama param sweep — `{model}`\n')
    out.append(f'- **Document:** `{doc_path}` ({doc_chars} chars)')
    out.append(f'- **Output cap:** {max_tokens} tokens / run')
    out.append(f'- **Host:** {OLLAMA_HOST}\n')
    out.append('## Results (sorted by decode tok/s)\n')
    out.append('| num_gpu | num_ctx | GPU% | model on GPU | nvidia-smi VRAM | load | TTFT | prompt t/s | decode t/s |')
    out.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in rows_ok:
        vt = (f'{r["vram_used_mb"]} / {r["vram_total_mb"]} MB'
              if r.get('vram_used_mb') is not None else '—')
        out.append(
            f'| {r["num_gpu"]} | {r["num_ctx"]} | {r["gpu_pct"]:.0f}% | '
            f'{r["size_vram_mb"]} / {r["size_mb"]} MB | {vt} | '
            f'{r["load_s"]:.1f}s | {_fmt(r.get("ttft"), ".2f")}s | '
            f'{r["prompt_rate"]:.0f} | {r["decode_rate"]:.1f} |'
        )
    if rows_err:
        out.append('\n## Failed configurations\n')
        out.append('| num_gpu | num_ctx | error |')
        out.append('|---:|---:|---|')
        for r in rows_err:
            out.append(f'| {r["num_gpu"]} | {r["num_ctx"]} | {r["error"]} |')
    out.append('\n## How to read this\n')
    out.append('- **decode t/s** is the steady-state generation speed — usually the metric to maximise.')
    out.append('- **TTFT** matters too: a config that is "100% GPU" but with TTFT 10× another '
               'config is paging into shared GPU memory; pick the lower-TTFT config even if its '
               'decode rate is slightly lower.')
    out.append('- **nvidia-smi VRAM** is dedicated only on WSL2 — Windows Task Manager '
               '("Shared GPU memory") confirms whether the rest is being paged.')
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument('document', help='Document used as the user prompt')
    ap.add_argument('--model', required=True, help='Ollama model tag, e.g. gemma4:e4b')
    ap.add_argument('--num-gpu', default='999',
                    help='Comma-separated num_gpu values (default: 999 = all layers)')
    ap.add_argument('--num-ctx', default='8192',
                    help='Comma-separated num_ctx values (default: 8192)')
    ap.add_argument('--max-tokens', type=int, default=200,
                    help='Cap output tokens per run (default: 200)')
    ap.add_argument('--auto-tune', action='store_true',
                    help='Probe model + adaptive sweep; ignores --num-gpu/--num-ctx')
    ap.add_argument('--auto-ctx', type=int, default=8192,
                    help='Context size used during auto-tune (default: 8192). '
                         'Once you have the best num_gpu, re-run with explicit '
                         '--num-ctx to test larger contexts.')
    ap.add_argument('--output', default='', help='Save markdown report to this path')
    args = ap.parse_args()

    if not os.path.isfile(args.document):
        print(f'error: file not found: {args.document}', file=sys.stderr); sys.exit(1)
    with open(args.document, encoding='utf-8') as f:
        doc = f.read()
    if not doc.strip():
        print('error: document is empty', file=sys.stderr); sys.exit(1)

    sep = '─' * 78
    probe = {}
    if args.auto_tune:
        print('Probing model via /api/show…')
        probe = probe_model(args.model)
        size_mb = (probe.get('size_bytes') or 0) // (1024 * 1024)
        print(f'  arch={probe.get("arch")}  layers={probe.get("num_layers")}  '
              f'max_ctx={probe.get("max_ctx")}  size={size_mb} MB')
        vram = get_vram_mb()
        if not vram:
            print('error: nvidia-smi unavailable — auto-tune needs VRAM info', file=sys.stderr)
            sys.exit(1)
        print(f'  dedicated VRAM: {vram[1]} MB')
        ng_vals  = coarse_grid(probe, vram[1])
        ctx_vals = [min(args.auto_ctx, probe.get('max_ctx') or args.auto_ctx)]
        print(f'  coarse num_gpu grid: {ng_vals}  ·  ctx: {ctx_vals[0]}')
    else:
        ng_vals  = parse_int_list(args.num_gpu)
        ctx_vals = parse_int_list(args.num_ctx)

    grid = [(ng, ctx) for ctx in ctx_vals for ng in ng_vals]

    print(sep)
    print(f'  Model:      {args.model}')
    print(f'  Document:   {os.path.basename(args.document)}  ({len(doc)} chars)')
    print(f'  Sweep:      {len(grid)} combinations'
          + ('  (coarse, auto-tune)' if args.auto_tune else ''))
    print(f'  Output cap: {args.max_tokens} tokens / run')
    print(f'  Host:       {OLLAMA_HOST}')
    print(sep)

    rows = []
    for i, (ng, ctx) in enumerate(grid, 1):
        rows.append(_run_one(args.model, doc, ng, ctx, args.max_tokens, i, len(grid)))

    if args.auto_tune:
        viable = [r for r in rows if _is_viable(r)]
        if viable:
            best = max(viable, key=lambda r: r['decode_rate'])
            already = {r['num_gpu'] for r in rows if r['num_ctx'] == best['num_ctx']}
            fine = fine_grid(best['num_gpu'], probe.get('num_layers') or 999, already)
            if fine:
                print(f'\n{sep}')
                print(f'  Fine sweep ±2 around best (num_gpu={best["num_gpu"]}, '
                      f'decode={best["decode_rate"]:.1f} t/s): {fine}')
                print(sep)
                for j, ng in enumerate(fine, 1):
                    rows.append(_run_one(args.model, doc, ng, best['num_ctx'],
                                         args.max_tokens, j, len(fine)))
        else:
            print(f'\n{sep}')
            print('  No viable config found in coarse sweep — every run was either '
                  'paging or failed. Model may not fit at all on this hardware.')
            print(sep)

    unload(args.model)

    rows_ok  = sorted([r for r in rows if 'error' not in r],
                      key=lambda r: -r.get('decode_rate', 0.0))
    rows_err = [r for r in rows if 'error' in r]

    print('\n' + sep)
    print('  Summary (sorted by decode tok/s, fastest first)')
    print(sep)
    print(f'  {"num_gpu":>7}  {"num_ctx":>7}  {"GPU%":>5}  {"VRAM(m)":>9}  '
          f'{"VRAM(total)":>15}  {"load":>5}  {"TTFT":>6}  '
          f'{"prompt t/s":>10}  {"decode t/s":>10}')
    for r in rows_ok:
        vt = (f'{r["vram_used_mb"]}/{r["vram_total_mb"]} MB'
              if r.get('vram_used_mb') is not None else '—')
        print(f'  {r["num_gpu"]:>7}  {r["num_ctx"]:>7}  {r["gpu_pct"]:>4.0f}%  '
              f'{r["size_vram_mb"]:>7}MB  {vt:>15}  {r["load_s"]:>4.1f}s  '
              f'{_fmt(r.get("ttft"), ".2f"):>5}s  '
              f'{r["prompt_rate"]:>10.0f}  {r["decode_rate"]:>10.1f}')
    for r in rows_err:
        print(f'  {r["num_gpu"]:>7}  {r["num_ctx"]:>7}  ✗ {r["error"]}')

    if args.output:
        md = build_markdown(args.model, args.document, len(doc), args.max_tokens,
                            rows_ok, rows_err)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f'\nSaved markdown report: {args.output}')


if __name__ == '__main__':
    main()
