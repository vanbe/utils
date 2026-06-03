---
name: utils-convert
description: Convert or extract documents (PDF, DOCX, ODT, MD, PPT, XLS) using the utils_run.py CLI tool. Use when the user wants to extract text from PDFs, convert Word documents to Markdown, convert ODT to DOCX, batch-process CVs or job offers, or any other document format conversion. Trigger on phrases like "extract CVs", "convert to Markdown", "extraire les CVs", "convert all PDFs", "transform documents", "extraire toutes les offres".
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
| `thumbnails`   | dir   | thumbnails                    | Pillow/ffmpeg; `--mirror` to sync deletions                   |

Success JSON for `image-dedup`: `{"status":"ok","output_file":"/abs/<dir>/duplicates.json"}`.
For in-place actions (raw-to-jpg, thumbnails) there is no single output file: `{"status":"ok"}`.
