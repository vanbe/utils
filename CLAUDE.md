# Utils Tools — Claude Project Notes

## Project overview

Terminal TUI (`utils_tools.py`) and a collection of actions scripts under `actions/`.
Primary working environment: WSL2 on Windows, with paths under `/mnt/c/` pointing to the Windows filesystem (Nextcloud sync folder).

**Non-interactive CLI for agents**: `utils_run.py` exposes all document conversion actions as a JSON API for use by automated agents (including Claude in other projects).

```bash
# Discover what actions are available for a file
python3 utils_run.py <file>

# Run an action — progress goes to stderr, JSON result to stdout
python3 utils_run.py <file> <action>

# List every registered action with supported extensions
python3 utils_run.py --list-all
```

Key actions for document extraction:
- `pdf-extract` — PDF → Markdown (PyMuPDF, fast, no GPU needed)
- `pdf-mineru` — PDF → Markdown (MinerU, best for layout/tables/formulas)
- `doc-to-md` — DOCX/DOC → Markdown (pandoc)
- `odt-to-docx` — ODT → DOCX (LibreOffice)

Exit code 0 on success, 1 on error. JSON shape:
- Success: `{"status": "ok", "output_file": "/abs/path/to/result.md"}`
- Error: `{"status": "error", "message": "..."}`

---

## Keeping `utils_run.py` and the skill in sync

The skill file at `.claude/skills/utils-convert/SKILL.md` documents every action exposed by `utils_run.py`. **Whenever you make any of the following changes, update both files together:**

| Change | What to update |
|--------|---------------|
| New action added to `_REGISTRY` in `utils_run.py` | Add a row to the action table in the skill |
| New **folder** action added to `_FOLDER_REGISTRY` in `utils_run.py` | Add a row to the "Folder operations" table in the skill **and** wire it into `_actions_for` + `_ACTION_CAT` in `utils_tools.py` (which reads the script path from `_FOLDER_REGISTRY` — single source of truth, no path drift) |
| Action removed or renamed in `utils_run.py` | Remove/rename the corresponding row in the skill |
| Supported extensions change for an action | Update the skill table and the "Choosing the right action" notes |
| Output filename convention changes (e.g. `_extracted.md` suffix) | Update the skill table |
| New action wired up in `_actions_for` in `utils_tools.py` but not yet in `utils_run.py` | Add it to `utils_run.py` first, then update the skill |
| New document type added to `utils_tools.py` (new `*_EXTS` constant + actions) | Mirror it in `utils_run.py` and the skill |

The single source of truth for the registries is `utils_run.py` — `_REGISTRY` (per-file actions) and `_FOLDER_REGISTRY` (folder/batch actions). The skill and `utils_tools.py` (TUI) are mirrors of it — they must stay in sync. Folder-action **script paths live only in `_FOLDER_REGISTRY`**; the TUI imports it, so a path can't drift between TUI and CLI.

---

## UX conventions for `utils_tools.py`

- **Default option always first** in `select_menu` lists — the cursor starts there, Enter confirms immediately without navigating.
- After an action completes, return directly to the file browser (no re-showing the action list).
- No mode-selection prompt on startup — go straight into the file browser.
- Folder actions are accessible via the `[./]` entry inside the browser; no separate mode needed.
- TTS speaking speed: 0.95 is default and must appear first in the speed menu.
- **Actions destructives en raw-mode** : l'écran live de *Record audio* exige une
  **confirmation O/oui** avant d'annuler-supprimer (Esc/Q n'efface pas directement
  le FLAC). En pause, les vumètres retombent à 0 et les aperçus interim « … » sont
  purgés (la capture est gelée par SIGSTOP, aucune donnée fraîche). Le nombre de
  lignes rendues doit rester constant entre les états (REC/PAUSE/confirm) pour que
  le redraw par `\033[NF` ne dérive pas.

---

## Model parameter benchmarking — MANDATORY for every model

**Every Ollama model used by this project must be empirically benched on the
target machine before being committed to `.env`.** Pure heuristics ("model
size + VRAM → num_gpu") do **not** predict the shared-memory paging cliff:
the cliff is a driver-level behaviour that triggers below the threshold raw
weights-and-cache math suggests. Two configs that look equivalent on paper
can differ by 8× in real decode rate.

**Worked example, this hardware** (6 GB dedicated VRAM, gemma4:e4b, 11 GB
total): `num_gpu=42` (33 % on GPU, rest on CPU) → 15 t/s. `num_gpu=46` (the
"100 % GPU" config Ollama recommends and what raw math would pick) → 2 t/s,
TTFT 18 s. The driver was paging the missing ~5 GB through shared GPU memory.
Partial offload that keeps CPU layers in fast system RAM beats paged-GPU.

**Tool**: [actions/ai_utils/bench_ollama_params.py](actions/ai_utils/bench_ollama_params.py)
**Detailed docs**: [actions/ai_utils/bench_ollama_params.md](actions/ai_utils/bench_ollama_params.md) — full CLI reference, modes, examples, troubleshooting.

**Auto-tune** (recommended for new / unknown models):
```
python3 actions/ai_utils/bench_ollama_params.py <doc> --model X --auto-tune
```
Probes the model via `/api/show` (size, layer count, max ctx), reads VRAM via
`nvidia-smi`, runs a coarse `num_gpu` sweep at `auto_ctx=8192`, then a fine
±2-layer sweep around the best non-paging result. Runtime scales with model
size: ~5 min for 4B models, 20–40 min for 26B+.

**Manual sweep** (when you already have a guess to validate):
```
python3 actions/ai_utils/bench_ollama_params.py <doc> --model X \
    --num-gpu A,B,C --num-ctx M,N
```

**Two-step workflow** for picking final `.env` values:
1. `--auto-tune` to find the best `num_gpu` at small ctx (8192).
2. With that `num_gpu` locked in, re-run with explicit `--num-ctx` covering
   the contexts you actually need (e.g. `--num-ctx 40000,80000,128000`) to
   confirm VRAM headroom and decode rate hold at larger contexts.

**Selection rules**:
- Maximise **decode tok/s** (steady-state generation speed).
- **Reject any config with TTFT > 5 s at "100 % GPU"** — that's the paging
  signature; pick a lower `num_gpu` even if its decode looks slightly worse.
- Leave **~1 GB nvidia-smi VRAM headroom** for other GPU consumers (browser,
  Windows DWM, model swap between A and B).
- For a model larger than dedicated VRAM, the all-on-GPU config
  (`num_gpu = num_layers`) is **always wrong**, even when `ollama ps` reports
  "100 % GPU".

**Reference results on this machine** (Lenovo, 6 GB dedicated VRAM):
- **Model B — `gemma4:e4b`** (11 GB total): `OLLAMA_NUM_GPU_B=42`,
  `OLLAMA_MODEL_B_CTX=128000` → ~15 t/s decode, 5.7 GB VRAM total, ctx fully
  pre-allocated. Use `100000` instead of `128000` if you want ~950 MB of
  headroom for other GPU consumers.
- **Model A — `gemma4:26b`** (~17 GB total): not yet empirically benched.
  Current `.env` guess `OLLAMA_NUM_GPU_A=7` / `OLLAMA_MODEL_A_CTX=4096` —
  must be validated with `--auto-tune` before being trusted.

When you change a model tag in `.env` (new quantisation, new minor version),
re-bench. Quantisation changes shift the per-layer footprint and can move
the paging cliff by several layers.

---

## AI actions (`actions/ai_utils/`)

- **Dedicated OCR model**: `pdf_vision_ocr.py` uses `OLLAMA_OCR_MODEL` / `OLLAMA_OCR_MODEL_CTX` / `OLLAMA_OCR_NUM_GPU` — independent of the A/B transform pipeline. The TUI must not ask for model/quality selection for Vision OCR.
- **Tuned-OCR mode**: `OCR_TUNED_PROMPT` env var. When set, `pdf_vision_ocr.py` skips the system prompt and the lang/text-hint augmentations, sending only the literal env value as the user message (e.g. `<image>\nFree OCR.` for DeepSeek-OCR). `\n` is converted to a real newline. Use this for specialised OCR models that ship with a fixed instruction phrase.
- **Inference extra_body**: every Ollama inference call must pass `extra_body=backend.get_inference_extra_body()` to prevent Ollama from silently reloading the model with default context (262144).
- **Context priority** in OCR: explicit env var → `_OCR_DEFAULT_CTX` floor (8192) → `--context-size` CLI override.
- **Model load timeout**: `_LOAD_TIMEOUT = 600` seconds (26B models take several minutes).
- **Unload events** must be logged as a dedicated `step/ok` line, not silently swallowed.

---

## Audio recording (action "Record audio")

Capture multi-sources (micros + sorties système) → **un seul FLAC multicanal** +
sidecar `<name>.channels.json` (map canal→source). Transcription par canal en
direct (`live_transcribe.py`) ou différée (`transcribe_channels.py`). **Action de
dossier, TUI-only.**

- **Moteur** : `actions/audio_utils/recorder.py` — `detect_backend()`,
  `list_sources()`, `class Recorder`. Backend-aware :
  - `linux` : capture locale PulseAudio/PipeWire (`parec` pour les vumètres,
    `ffmpeg -f pulse … amerge` pour le FLAC). Les SORTIES = sources `.monitor`.
  - `wsl` : WSL ne peut pas capter la sortie système Windows nativement. La
    capture est déléguée à Windows via le binaire **`bin/capture.exe`** (loopback
    WASAPI = **API Windows intégrée, rien à installer** : pas de driver, pas de
    filtre DShow/`regsvr32`, pas de VB-CABLE/VoiceMeeter, pas de VC++ redist).
    Lancé en interop ; PCM s16le sur stdout → `ffmpeg` (WSL) encode le FLAC ;
    lignes `LEVEL i rms` sur stderr → vumètres.
- **`capture.exe` n'est PAS versionné** (gitignoré dans `bin/`). On committe la
  source + les scripts : `actions/audio_utils/capture/` (`capture.cpp`,
  `build.sh` cross-compile mingw, `build.bat` Windows, `README.md`). La TUI
  propose de le **construire** s'il manque (`recorder.build_capture()`). Cross-build
  vérifié : `g++-mingw-w64-x86-64`, binaire statique, imports = `kernel32/msvcrt/
  ole32` uniquement.
- **TUI-only assumé** : l'action est câblée dans `_actions_for` (branche dossier)
  + `_ACTION_CAT` de `utils_tools.py`, mais **PAS** dans `_REGISTRY`/
  `_FOLDER_REGISTRY` de `utils_run.py` ni dans `SKILL.md` (enregistrement live =
  interactif). Ne pas la croire « manquante » lors d'une passe de sync registries.
- **Démux + transcription différée par canal** : `transcribe_channels.py` lit le
  sidecar `channels.json`, démux chaque source (`pan=mono|c0=…` = **moyenne** des
  canaux de la source, pas seulement `c0`), transcrit par canal et fusionne →
  `<base>.srt` + `<base>.md`. **Attribution Moi/Système par construction** (pas de
  pyannote). Câblé dans la TUI : `act_transcribe` détecte le sidecar et propose
  « Par canal » (recommandé) vs « Standard — diarisation » (`transcribe_audio.py`,
  qui downmix tout en mono + diarise). ⚠ Ne PAS passer un FLAC multicanal au mode
  standard sans raison : son préprocess `pan=mono|c0=c0` ne garde que le 1er canal.
- **Labels par canal mutualisés** : `whisper_common.channel_labels(sources)` est
  l'unique source des libellés « Moi / Système » — utilisé par `live_transcribe.py`
  ET `transcribe_channels.py` (donc live ≡ différé par canal).

### Transcription live par canal

Option « Transcription live ? Oui » dans *Record audio* → transcrit chaque canal
en direct, affichage défilant (`Moi` / `Système`) + `.srt` aligné sur le FLAC.

- **Moteur** : `actions/audio_utils/live_transcribe.py` — `class LiveTranscriber`.
  Reçoit le PCM via `Recorder(on_pcm=…)` (tee : `capture.exe`/pulse → Python →
  ffmpeg FLAC **et** transcription), désentrelace par canal (`channels.json`),
  downmix mono + resample 16 kHz, **VAD-gate** par énergie, file → **un seul
  worker WhisperModel partagé**.
- **Attribution locuteur = par canal**, pas de pyannote : `input`→« Moi »,
  `output`→« Système ». C'est l'intérêt majeur du multicanal.
- **1 seul modèle partagé** (canaux en série) : sur un GPU unique, dédoubler ne
  gagne rien et **2 modèles turbo ne tiennent pas en 6 Go** + DWM Windows. VAD-gate
  → micro muet = coût nul. faster-whisper n'est pas streaming → **chunk-on-silence**.
- **Sorties** : `.srt` (avec timings) **et** `.md` (sans timings) par défaut.
- **Code mutualisé** : `whisper_common.py` (table modèles, chargement+fallback OOM,
  **`write_srt`/`write_md`**) est utilisé par `live_transcribe.py` ET
  `transcribe_audio.py` → le MD live est **identique** au MD différé (même writer).
  Poids de modèle partagés (`download_root` = `vendor/models/whisper`).
- **Modèles recommandés = défaut « (recommandé) »**, **distincts live vs différé**
  via `whisper_common.recommended_model(role)` : `role='live'` → `LIVE_TRANSCRIBE_MODEL`
  (contrainte temps réel), `role='transcribe'` → `TRANSCRIBE_MODEL` (qualité, sans
  contrainte). Calcul matériel (`setup_env.py`) : GPU 5–10 Go → turbo pour les deux ;
  ≥10 Go → live turbo / différé large ; **CPU → live base (temps réel) / différé small**.
  Clés dans `_AUTO_KEYS`. (Bench espeak CPU 2 cœurs : la qualité dépend surtout de
  l'audio réel ; small > base > tiny ; small n'est pas temps réel sur CPU faible.)
- **Hôte réactif** : `LIVE_TRANSCRIBE_CPU_THREADS` (= cœurs − 1) limite les threads
  CPU du live (sans effet sur GPU).
- **Décodage par canal (piège corrigé)** : comme on VAD-gate déjà nous-mêmes,
  le worker décode avec `vad_filter=False` **et `no_speech_threshold=1.0`**.
  Sans ce dernier, faster-whisper classait certains énoncés courts comme
  « non-parole » (no_speech_prob > 0.6) et **jetait tout le chunk** → un canal
  (souvent la sortie système, plus courte) pouvait disparaître de la
  transcription de façon intermittente. Ne PAS réactiver le VAD/no_speech interne
  sur les chunks déjà découpés. (Le différé `transcribe_audio.py` garde `vad_filter`
  car il traite un fichier entier, pas des énoncés pré-découpés.)
- **Temps réel** : aperçus **interim** pendant qu'on parle
  (`LIVE_TRANSCRIBE_INTERIM_SEC`, défaut 2 s) — affichés grisés « … » puis
  remplacés par le final ; ≤1 interim en vol par source (pas de pile-up).
  `condition_on_previous_text=False` + warmup modèle au démarrage.
- **Config `.env`** : `LIVE_TRANSCRIBE_MODEL` + `TRANSCRIBE_MODEL` / `_DEVICE` /
  `_COMPUTE_TYPE` / `_CPU_THREADS` / `_LANGUAGE` (`fr`|`auto`) / `_WINDOW_SEC` /
  `_INTERIM_SEC`. Aucune dépendance nouvelle (faster-whisper marche en CPU via
  CTranslate2, **sans torch**).

---

## Recurring problems

### WSL2 `/mnt/c/` path disappears mid-session
**Symptom**: `OSError: [Errno 22] Invalid argument` on `os.listdir()` after returning from an action.
**Cause**: Nextcloud (or Windows) renames/moves the folder during sync while the TUI holds the path.
**Fix**: `_browser_entries` retries `os.listdir()` up to 4 times (0.4 s apart) before giving up. On persistent failure it shows a visible warning row instead of silently appearing empty. `browse()` checks `os.path.isdir(current)` at the top of the loop and walks up to parent if the directory is truly gone.

### Ollama reloads model on every inference call
**Symptom**: High CPU / SSD thrashing mid-inference; VRAM shows model loading again.
**Cause**: `/v1/chat/completions` sent without `options` → Ollama sees default `num_ctx=262144`, different from load-time options, and reloads.
**Fix**: `OllamaBackend` tracks `_active_options` set at `load()`; `get_inference_extra_body()` returns them; every inference call injects them via `extra_body`.

### `num_gpu` via Ollama API options not applied for vision models
**Symptom**: `qwen2.5vl:7b` overflows VRAM regardless of `OLLAMA_OCR_NUM_GPU` value.
**Cause**: (1) Ollama may ignore `num_gpu` in API options for some model families. (2) The vision encoder (mmproj/SigLIP) always loads fully to GPU regardless of `num_gpu`.
**Fix**: Use a Modelfile with `PARAMETER num_gpu N` baked in, or switch to a smaller vision model (`qwen2.5vl:3b`). `OLLAMA_NUM_GPU` env var only applies when the Ollama server starts, not per-request.

### Ollama service broken / falls back to CPU / "no models" hang on WSL2
**DO NOT touch the Ollama systemd unit or run `ollama serve` manually without reading this first.** A bad config here cost the user a full reinstall and hours of debugging — treat changes here as risky and confirm before acting.

**Invariants on this machine**:
- Service unit: `/etc/systemd/system/ollama.service` (upstream, runs as user `ollama`).
- Models live at `/usr/share/ollama/.ollama/models/` (owned by the `ollama` user). Do **not** assume `~/.ollama/models/`.
- `nvidia-smi` on WSL2 is at `/usr/lib/wsl/lib/nvidia-smi`, **not** `/usr/bin/nvidia-smi`.
- The upstream service detects the GPU on its own — no `ExecStartPre` GPU check is needed.

**Trap 1 — bad `ExecStartPre` in override.conf**:
- Symptom: `systemctl status ollama` shows `activating (auto-restart)` and `Process: … ExecStartPre=/usr/bin/nvidia-smi -L (code=exited, status=203/EXEC)`. The service never starts.
- Cause: an `/etc/systemd/system/ollama.service.d/override.conf` with `ExecStartPre=/usr/bin/nvidia-smi -L` — wrong path on WSL2.
- Fix: either remove the override (`sudo rm /etc/systemd/system/ollama.service.d/override.conf && sudo systemctl daemon-reload && sudo systemctl restart ollama`) or correct the path to `/usr/lib/wsl/lib/nvidia-smi`.

**Trap 2 — manually-launched `ollama serve` masks the broken service**:
- Symptom: API responds at `127.0.0.1:11434` but `/api/tags` returns `{"models":[]}`; AI transforms hang at "Load model" forever.
- Cause: when systemd ollama fails, running `ollama serve` as your own user starts a second instance that reads from `~/.ollama/models/` (empty) instead of `/usr/share/ollama/.ollama/models/`. It also runs without the render/video group membership and the WSL nvidia lib path, so **CUDA is not detected → everything runs on CPU**.
- Detect: `ps -ef | grep '[o]llama serve'` — if the user is `charles…` (not `ollama`), it's the rogue manual instance.
- Fix: kill the manual `ollama serve`, fix the systemd unit (Trap 1), then `sudo systemctl restart ollama`. Verify with `curl -fsS http://127.0.0.1:11434/api/tags` (should list `gemma4`, `deepseek-ocr`, `qwen2.5vl`, etc.) and check ollama logs for `library=cuda` rather than `library=cpu`.

**Diagnostic checklist before doing anything else when ollama "isn't working"**:
1. `systemctl is-active ollama` — must be `active`, not `activating`.
2. `ps -ef | grep '[o]llama serve'` — exactly one process, owned by `ollama` user.
3. `curl -fsS http://127.0.0.1:11434/api/tags` — must list the expected models.
4. `journalctl -u ollama --no-pager -n 50 | grep -iE 'cuda|cpu|library='` — should show `cuda` runner, not `cpu`.
