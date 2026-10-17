---
name: utils-convert
description: Convert or extract documents (PDF, DOCX, ODT, MD, PPT, XLS, EPUB) and transcribe audio using the utils_run.py CLI tool. Use when the user wants to extract text from PDFs, convert Word documents to Markdown, convert ODT to DOCX, batch-process CVs or job offers, or any other document format conversion. Trigger on phrases like "extract CVs", "convert to Markdown", "extraire les CVs", "convert all PDFs", "transform documents", "extraire toutes les offres".
allowed-tools: Bash
---

# Document conversion via utils_run.py

The conversion tool is `utils_run.py` at the root of the utils repo.
Set `UTILS_RUN` to its absolute path before running any snippet below:
```bash
UTILS_RUN=~/code/utils/utils_run.py   # adjust if cloned elsewhere
```
It uses the utils project's own `.venv` internally — no activation needed.

## Two-step pattern

### 1. Discover available actions for a file
```bash
python3 "$UTILS_RUN" <file>
```
Returns a JSON array:
```json
[{"action": "pdf-extract", "desc": "..."}, {"action": "pdf-mineru", "desc": "..."}]
```
An empty array means no conversion available (e.g. the file is already `.md`).

### 2. Run an action
```bash
python3 "$UTILS_RUN" <file> <action>
```
Returns JSON:
- Success: `{"status": "ok", "output_file": "/abs/path/to/result.md"}`
- Error:   `{"status": "error", "message": "..."}`

Exit code 0 = ok, 1 = error.
Progress and logs from the underlying tool go to stderr — only JSON on stdout.

### Options (any file action)
```bash
python3 "$UTILS_RUN" <file> <action> --out-dir <dir> [script opts...]
```
- `--out-dir <dir>` — write the output (and any side files: images folder,
  temp WAVs…) into `<dir>` (created if needed) instead of beside the input. The
  input folder is left untouched (the input is exposed in `<dir>` through a
  temporary symlink, removed afterwards). `output_file` then points into `<dir>`.
  Fails if `<dir>` already holds a *different* file with the input's name.
  Use it for read-only / synced sources (e.g. Nextcloud).
- Any other trailing options are passed through to the underlying script
  (e.g. `audio-transcribe … --model turbo --max-speakers 2`).
- Folder actions (`--folder`) do not take `--out-dir`.

### `to-md` — default Markdown conversion
`python3 "$UTILS_RUN" <file> to-md [--out-dir <dir>]` picks the default
`*-to-md` action for the extension (`_TO_MD` in `utils_run.py`): `.pdf` →
`pdf-extract`, `.docx/.doc` → `doc-to-md`, `.pptx/.ppt` → `ppt-to-md`,
`.xlsx/.xlsm/.xls` → `xls-to-md`, `.epub` → `epub-to-md`, audio/video →
`audio-transcribe`. Success JSON adds `"action": "<resolved action>"`. Use the
explicit action for heavier alternatives (`pdf-mineru`, `pdf-vision-ocr`).

## All available actions

| Action           | Input             | Output                 | Backend               |
|------------------|-------------------|------------------------|-----------------------|
| `pdf-extract`    | `.pdf`            | `<base>_extracted.md`  | PyMuPDF + OCR fallback |
| `pdf-mineru`     | `.pdf`            | `<base>_mineru.md`     | MinerU (best layout)  |
| `pdf-vision-ocr` | `.pdf`            | `<base>_ocr.md`        | AI vision via Ollama  |
| `doc-to-md`      | `.docx` / `.doc`  | `<base>.md`            | pandoc                |
| `doc-to-pdf`     | `.docx` / `.doc`  | `<base>.pdf`           | LibreOffice           |
| `odt-to-docx`    | `.odt`            | `<base>.docx`          | LibreOffice           |
| `odt-to-pdf`     | `.odt`            | `<base>.pdf`           | LibreOffice           |
| `md-to-docx`     | `.md`             | `<base>.docx`          | pandoc                |
| `md-to-pdf`      | `.md`             | `<base>.pdf`           | pandoc + XeLaTeX      |
| `ppt-to-pdf`     | `.pptx` / `.ppt`  | `<base>.pdf`           | LibreOffice           |
| `xls-to-pdf`     | `.xlsx` / `.xls`  | `<base>.pdf`           | LibreOffice           |
| `ppt-to-md`      | `.pptx` / `.ppt`  | `<base>.md`            | python-pptx (`.ppt` → `.pptx` via LibreOffice first) |
| `xls-to-md`      | `.xlsx` / `.xlsm` / `.xls` | `<base>.md`   | openpyxl (`.xls` → `.xlsx` via LibreOffice first) |
| `epub-to-md`     | `.epub`           | `<base>.md`            | pandoc, per spine document |
| `audio-transcribe` | `.m4a` `.mp3` `.wav` `.mp4` `.mkv` | `<base>_transcription.md` | local faster-whisper + pyannote (GPU) |
| `meeting-report` | audio (`.m4a` `.mp3` `.wav` `.flac` `.ogg` `.opus` `.aac` `.webm` `.mka`) | `Traité/AAAA-MM-JJ - Sujet.md` (+ `_original`, `.srt`, `_brut.md`) | OVH whisper-large-v3 + LLM via LiteLLM (no GPU) |

## Common patterns

### Extract all CVs in a folder (PDF + DOCX → Markdown)
```bash
for f in /path/to/cvs/*.pdf; do
  result=$(python3 "$UTILS_RUN" "$f" pdf-extract)
  echo "$result"
done
for f in /path/to/cvs/*.docx; do
  result=$(python3 "$UTILS_RUN" "$f" doc-to-md)
  echo "$result"
done
```

### Process with error checking
```bash
result=$(python3 "$UTILS_RUN" "$file" pdf-extract)
status=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
if [ "$status" = "ok" ]; then
  out=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['output_file'])")
  echo "Extracted to: $out"
else
  echo "Error: $(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['message'])")"
fi
```

## Choosing the right PDF action

- **`pdf-extract`** — fast, no GPU, best for text-based PDFs with embedded text
- **`pdf-mineru`** — slower, best for scanned docs, complex layouts, tables, formulas
- **`pdf-vision-ocr`** — requires Ollama running; best for scanned/handwritten content

## Locator-preserving Markdown (ppt / xls / epub / audio)

These outputs keep exact locations so notes can cite them:

- **`ppt-to-md`** — per slide, in deck order: `<!-- slide N -->` then
  `## Slide N — <title>` (`(untitled)` if none, `(hidden)` for hidden slides);
  body text in reading order with bullet levels; tables as pipe tables; charts
  as a small data table; slides with pictures but no text → `[image-only slide]`
  (no content at all → `[empty slide]`); other pictures → `*[n image(s) on slide]*`.
  Speaker notes in a delimited block:
  `<!-- notes slide N -->` / `**Speaker notes:**` / … / `<!-- /notes -->`.
- **`xls-to-md`** — per sheet, workbook order: `<!-- sheet N: <name> -->` +
  `## Sheet N — <name>` (`(hidden)` flagged). The used range as a table whose
  header is the column letters and first column the row number (cite `B7`);
  values are the results cached in the file. Then `### Formulas`:
  `` - `B4`: `=B2-B3` → 40 ``. Merged ranges listed. Script options
  `--max-rows N` / `--max-formulas N` (default 2000 per sheet, truncation noted).
- **`epub-to-md`** — `# <book title>`, then per spine document with text:
  `<!-- chapter N -->` + `## <title>` (title from the book's TOC, else its first
  heading); inner headings start at `###`; images → `[image: alt]`.
- **`audio-transcribe`** — `[hh:mm:ss] **Speaker N**: …` paragraphs (consecutive
  segments of one speaker merged, stamped with the start time). Default language
  `en`, model `large`; pass `--language fr|auto`, `--model turbo`, `--max-speakers N`,
  `--no-speaker`, `--enhance`, `--no-improve`. Intermediate `.srt` / WAV are
  removed (`--keep-srt` keeps the SRT). Runs locally on the GPU (several minutes
  per hour of audio); unlike `meeting-report` nothing leaves the machine and the
  source is not moved. Diarization needs `AUDIO_UTILS_HF_TOKEN` in `.env`.
  GPU: `TRANSCRIBE_GPU` in the utils `.env` (or the environment) = nvidia-smi
  index, applied in PCI bus order (CUDA's default order differs on multi-GPU
  machines); a `CUDA_VISIBLE_DEVICES` set by the caller wins.

## MD files need no conversion

`.md` files are already plain text — read them directly with the `Read` tool.

## Folder operations (image / batch)

Besides per-file actions, `utils_run.py` exposes **folder-level** operations that
act on a whole directory (recursively). Same JSON contract.

```bash
python3 "$UTILS_RUN" --list-folders                          # discover folder actions
python3 "$UTILS_RUN" --folder <dir> <action> [extra args]    # run one (args passed to the script)
```

| Action         | Input | Output                        | Notes                                                          |
|----------------|-------|-------------------------------|----------------------------------------------------------------|
| `image-dedup`  | dir   | `<dir>/duplicates.json`       | Duplicate groups. `--method exact` (SHA-256, default) or `perceptual` (imagehash, `--threshold N`, groups are `decision:pending` to review) |
| `image-index`  | dir   | `<dir>/.image_index.json`     | Incremental SHA-256 index of a library (path→hash). `--exclude <subdir>` (repeatable). For O(1) "do I already have this?" lookups |
| `raw-to-jpg`   | dir   | JPEGs in place                | RAW → JPEG (rawpy + exiftool); `--delete-raws`                 |
| `heic-to-jpg`  | dir   | JPEGs in place                | HEIC/HEIF → JPEG (pillow-heif; EXIF + ICC preserved, orientation normalised). `-q/--quality` (default 95, 4:4:4 chroma), `--no-recursive`, `--overwrite`, `--delete-heics` (deletes each source only after a verified conversion) |
| `thumbnails`   | dir   | thumbnails                    | Pillow/ffmpeg; `--mirror` to sync deletions                   |

Success JSON for `image-dedup`: `{"status":"ok","output_file":"/abs/<dir>/duplicates.json"}`.
For in-place actions (raw-to-jpg, heic-to-jpg, thumbnails) there is no single output file: `{"status":"ok"}`.

### `meeting-report` notes
- **Moves the source audio** into `<dir>/Traité/` (renamed `<base>_original.<ext>`), so run it on
  a copy if the original must stay in place. The JSON result carries `final_md`, `subject`,
  `meeting_type`, `cost_eur`, `participants` (output name is chosen by the AI: no fixed suffix).
- No audio enhancement by default (it hurt clean headset audio in an A/B test); run the script
  directly with `--enhance` for far-field / noisy recordings (adds `<base>_enhanced.m4a`).
- `--kind lecture` (run the script directly) = course / conference notes in English instead of a
  meeting report: breaks and off-topic chatter excluded, professor identified, course matched
  against `MEETING_COURSE_CATALOG`. Combine with `--enhance` for in-room recordings.
- A multichannel FLAC from *Record audio* (with `<name>.channels.json`) gets exact Moi/Système
  attribution; any other audio is diarised (`Speaker N`).
- Needs `INFERENCE_BASE_URL` / `INFERENCE_API_KEY` (LiteLLM gateway) and `OVH_AI_API_KEY`
  (diarisation) in the utils `.env`. Audio and transcript are sent to OVH (EU) — not local.

