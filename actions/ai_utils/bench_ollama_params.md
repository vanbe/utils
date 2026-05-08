# `bench_ollama_params.py` — Ollama parameter sweeper

Measures how a given Ollama model performs on **this machine** under different
`num_gpu` / `num_ctx` settings, so you can pick values for `.env` that don't
silently page into shared GPU memory.

> **Why this tool exists** — see the [Model parameter benchmarking](../../CLAUDE.md)
> section in `CLAUDE.md` for the rationale and the worked example. TL;DR: pure
> heuristics ("model size + VRAM → num_gpu") do not predict the
> shared-memory paging cliff. You have to measure.

---

## What it does

For each `(num_gpu, num_ctx)` combination in the sweep:

1. Unloads the model.
2. Reloads it with the params (`keep_alive=-1`).
3. Snapshots `/api/ps` (model size, `size_vram` → % on GPU) and `nvidia-smi`
   dedicated VRAM.
4. Runs **one** streaming `/api/chat` inference against the document, capturing:
   - **TTFT** — wall-clock time to first streamed token.
   - **prompt t/s** — `prompt_eval_count / prompt_eval_duration`.
   - **decode t/s** — `eval_count / eval_duration` (steady-state generation
     speed; the metric you usually want to maximise).
5. Unloads, then continues to the next combination.

At the end it prints a console summary sorted by decode tok/s and, with
`--output`, writes the same table as a Markdown report.

---

## Two modes

### `--auto-tune` (recommended for new / unknown models)

Probes the model via `/api/show` (architecture, layer count, max context),
reads dedicated VRAM via `nvidia-smi`, then runs:

1. **Coarse sweep** — `num_gpu` candidates bracketing the predicted "fit"
   plus the all-layers config (to expose paging) and `0` (CPU baseline), all
   at `--auto-ctx` (default `8192`).
2. **Fine sweep** — ±2 layers around the best non-paging coarse result
   (skipping values already tested).

A run is considered **viable** (not paging) if `TTFT < 5 s` and
`decode > 3 t/s`. If no coarse run is viable, the model probably doesn't fit
on this hardware at all and the fine pass is skipped.

Runtime scales with model size: ~5 min for 4 B models, 20–40 min for 26 B+.

### Manual sweep

When you already have specific values to validate or compare:

```bash
python3 actions/ai_utils/bench_ollama_params.py <doc> --model X \
    --num-gpu A,B,C --num-ctx M,N
```

---

## CLI

```
bench_ollama_params.py <document> --model <tag> [options]
```

| Argument | Default | Notes |
|---|---|---|
| `document` | — (required) | Plain-text / Markdown file used as the user prompt. Must be non-empty. |
| `--model` | — (required) | Ollama model tag, e.g. `gemma4:e4b`, `gemma4:26b`. |
| `--num-gpu` | `999` | Comma-separated list. `999` means "all layers on GPU" (Ollama caps to `num_layers`). Ignored under `--auto-tune`. |
| `--num-ctx` | `8192` | Comma-separated list. Ignored under `--auto-tune` (uses `--auto-ctx`). |
| `--max-tokens` | `200` | Cap output tokens per run. Higher = more reliable decode rate, longer wall-clock. |
| `--auto-tune` | off | Probe + adaptive coarse-then-fine sweep. Ignores `--num-gpu` / `--num-ctx`. |
| `--auto-ctx` | `8192` | Context size used during auto-tune. Keep small — `num_gpu` tuning is what auto-tune is for. Validate larger contexts in a follow-up manual run. |
| `--output` | — | If set, writes the summary table to this Markdown path. |

Environment:

- `OLLAMA_HOST` — defaults to `http://localhost:11434`. Override to bench a
  remote/alternate Ollama instance.

---

## Recommended two-step workflow for `.env` values

1. **Find the best `num_gpu`** with `--auto-tune` at the default small ctx:
   ```bash
   python3 actions/ai_utils/bench_ollama_params.py "test/some_doc.md" \
       --model gemma4:e4b --auto-tune --output test/bench_gemma4_e4b.md
   ```
2. **Lock in that `num_gpu`** and validate it at the contexts you actually
   need (VRAM grows with KV cache, so a config that fits at 8 K may page at
   128 K):
   ```bash
   python3 actions/ai_utils/bench_ollama_params.py "test/some_doc.md" \
       --model gemma4:e4b --num-gpu 42 --num-ctx 40000,80000,128000 \
       --output test/bench_gemma4_e4b_largectx.md
   ```

---

## Reading the results

Each row of the summary table looks like:

```
num_gpu  num_ctx  GPU%  VRAM(m)   VRAM(total)   load   TTFT  prompt t/s  decode t/s
     42     8192   33%   1900MB   5700/6144 MB   3.2s  0.45s         220        15.3
```

Selection rules (also in `CLAUDE.md`):

- **Maximise decode tok/s** — that's the steady-state generation speed.
- **Reject any "100 % GPU" config with TTFT > 5 s** — that's the paging
  signature. Pick a lower `num_gpu` even if its decode looks marginally worse.
- **Leave ~1 GB of `nvidia-smi` VRAM headroom** for other GPU consumers
  (browser, Windows DWM, model A↔B swap).
- **For a model larger than dedicated VRAM, `num_gpu = num_layers` is always
  wrong**, even when `ollama ps` reports "100 % GPU" — Ollama reports
  intent, not what actually fits.

### Why every metric, not just "% on GPU"

`ollama ps` "100 % GPU" only means Ollama *placed* all layers on the GPU. If
dedicated VRAM is exhausted, the driver pages into shared GPU memory (Windows
DX shared / Linux GTT). The run still completes — TTFT explodes and decode
collapses. Always read TTFT and decode rate alongside the % figure before
declaring a config "fits".

On WSL2, `nvidia-smi` shows **dedicated VRAM only**. Cross-reference Windows
Task Manager → Performance → GPU → "Shared GPU memory" to confirm paging.

---

## Examples

Coarse + fine auto-tune, save report:

```bash
python3 actions/ai_utils/bench_ollama_params.py \
    "test/Creating Shared Value_mineru.md" \
    --model gemma4:e4b --auto-tune \
    --output test/bench_gemma4_e4b.md
```

Manual sweep over a tight range with a longer output cap (more reliable decode
rate):

```bash
python3 actions/ai_utils/bench_ollama_params.py "test/some_doc.md" \
    --model gemma4:26b --num-gpu 5,6,7,8 --num-ctx 4096 --max-tokens 400
```

Validate `num_gpu` at production context sizes:

```bash
python3 actions/ai_utils/bench_ollama_params.py "test/some_doc.md" \
    --model gemma4:e4b --num-gpu 42 --num-ctx 40000,80000,128000
```

---

## Troubleshooting

**`error: nvidia-smi unavailable` under `--auto-tune`** — auto-tune needs VRAM
info to build its coarse grid. The script tries `/usr/lib/wsl/lib/nvidia-smi`
first (correct path on WSL2), then `nvidia-smi` from `$PATH`. If neither
works, fall back to a manual sweep with explicit `--num-gpu`.

**Load `HTTP 500` on a high `num_gpu`** — Ollama may refuse to load if the
KV cache won't fit. The script records the error and continues with the next
combination, so it's safe to leave aggressive values in the sweep.

**Every run shows the same `num_gpu` in `ollama ps`** — Ollama silently caps
`num_gpu` to the model's actual layer count, so e.g. `999` becomes
`num_layers` for any model. This is expected; the option still serves as
"all layers on GPU".

**`num_gpu` ignored for vision models** — Ollama may ignore `num_gpu` in API
options for some model families (notably Qwen2.5-VL); the vision encoder
(mmproj/SigLIP) also always loads to GPU regardless. For those, build a
Modelfile with `PARAMETER num_gpu N` baked in instead of relying on this
sweep. See the "Recurring problems" section in `CLAUDE.md`.

**TTFT looks fine but decode is suspiciously low** — bump `--max-tokens`. At
the default `200` tokens, a model that warms up slowly can show a depressed
decode rate; `400`–`800` gives a more representative steady-state figure.
