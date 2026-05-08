# utils

An interactive TUI for video, audio, document and image processing — runs inside WSL2 on Windows.

Navigate with **↑↓ arrows**, select with **Enter**, go back with **Esc**.
Browse to a file or folder, then choose from the actions available for it.

---

## Using with Claude Code in other projects

`utils_run.py` exposes all conversion actions as a simple JSON CLI, designed to be called by Claude agents working in any other project. A ready-made **Claude Code skill** is included — copy it once and Claude will automatically convert documents without any extra prompting.

```bash
# Run this from your other project's root
mkdir -p .claude/skills
cp -r ~/code/utils/.claude/skills/utils-convert .claude/skills/
```

That's it. Claude Code in that project will now recognise when you ask to extract, convert, or batch-process documents and will call `utils_run.py` automatically.

> If you cloned utils to a non-standard path, edit `UTILS_RUN=~/code/utils/utils_run.py` at the top of the skill file.

---

## Prerequisites

- **Windows 10/11** with **WSL2** installed ([install guide](https://learn.microsoft.com/en-us/windows/wsl/install)) — Ubuntu or Debian distro recommended
- **NVIDIA GPU** — optional, but needed for AI-heavy actions (MinerU OCR, transcription). CPU-only works for most actions.

---

## Architecture

```
utils/
├── utils_tools.py          ← TUI entry point (file browser + menus)
├── actions/
│   ├── audio_utils/        ← transcription, audio conversion
│   ├── document_utils/     ← PDF, DOCX, Markdown, ODT conversion
│   ├── picture_utils/      ← thumbnails, RAW → JPEG
│   ├── video_utils/        ← split, compress, extract audio
│   └── ai_utils/           ← Ollama/LLM-based transforms
├── install.sh              ← one-shot install (packages + venv + command)
├── setup_env.py            ← hardware detection → writes .env
└── utils_run.py            ← non-interactive CLI for scripting / agents
```

The TUI (`utils_tools.py`) is the user-facing entry point. It runs in WSL and is launched from Windows via a small `.bat` / `.ps1` relay that passes the current Windows directory. All processing happens inside WSL — no Python on the Windows side.

---

## Installation

### 1. Clone into `~/code/utils` (path is required)

```bash
git clone <repo-url> ~/code/utils
cd ~/code/utils
```

> The Windows launchers assume this exact path (`~/code/utils`). If you clone elsewhere the Windows shortcut will not work.

### 2. Run the install script

```bash
bash install.sh
```

This will:
- Install system packages: `ffmpeg`, `pandoc`, `tesseract`, `libreoffice`, `ocrmypdf`, and others
- Create a Python virtual environment (`.venv/`) and install all Python dependencies
- Auto-detect your hardware (CPU, RAM, GPU) and write tuned settings to `.env`
- Optionally prompt for your HuggingFace token (only needed for speaker diarization in transcription — you can skip and add it to `.env` later)
- Install a `utils_tools` command in `~/.local/bin` so you can launch the TUI directly from WSL

After install, the TUI is available in WSL:

```bash
utils_tools          # open TUI in the current directory
utils_tools /path    # open TUI in a specific directory
```

### 3. (Optional) MinerU PDF OCR — download models

The `pdf-mineru` action uses heavy AI models (~3 GB). Download them once:

```bash
.venv/bin/mineru-models-download -s huggingface -m pipeline
```

GPU required. Skip if you do not plan to use MinerU.

### 4. (Optional) Windows launcher — call from any folder in PowerShell / CMD

This sets up a `utils_tools` command that you can run from **any folder** in Windows Terminal without opening WSL manually.

**a) Create a personal bin folder** (skip if you already have one):

```
mkdir C:\Users\%USERNAME%\bin
```

**b) Copy the launchers from WSL:**

```bash
cp ~/code/utils/install/windows/utils_tools.bat /mnt/c/Users/$USER/bin/
cp ~/code/utils/install/windows/utils_tools.ps1 /mnt/c/Users/$USER/bin/
```

Or from Windows Explorer, copy both files from:
```
\\wsl.localhost\Ubuntu\home\<your-wsl-username>\code\utils\install\windows\
```
to `C:\Users\<your-windows-username>\bin\`.

**c) Add the folder to your Windows PATH** — open PowerShell **as Administrator**:

```powershell
[Environment]::SetEnvironmentVariable(
    "PATH",
    $env:PATH + ";C:\Users\$env:USERNAME\bin",
    "User"
)
```

Then **restart your terminal**.

> **PowerShell users**: use `utils_tools.ps1` — it handles accented characters (é, à, ü…) in folder names correctly. If PowerShell blocks scripts, run once as admin: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
>
> **CMD users**: use `utils_tools.bat`.

---

## Usage

From **WSL**:

```bash
utils_tools          # open TUI in the current directory
```

From **Windows** (any folder, any terminal — after step 4):

```
utils_tools
```

The TUI opens with the current directory as your working folder. Navigate to a file, press Enter to see available actions.

---

## Available actions

Actions appear automatically based on the file or folder you select.

### Video files
| Action | Description |
|--------|-------------|
| Transcribe | Speech → Markdown or SRT using Whisper |
| Extract audio | Stream-copy raw + EBU R128 normalized FLAC |
| Split | Cut between two timestamps |
| Compress audio | Re-encode to MP3 / AAC / Opus / OGG |

### Audio files
| Action | Description |
|--------|-------------|
| Transcribe | Speech → Markdown or SRT using Whisper |
| Convert to MP3 | Re-encode at chosen quality |
| Improve quality | EBU R128 loudness normalization |
| Compress | Choose format and bitrate |

### Markdown files
| Action | Description |
|--------|-------------|
| Text to speech | Kokoro TTS → MP3 (multilingual, multiple voices) |
| Export to PDF | pandoc + XeLaTeX |
| Export to DOCX | pandoc |

### PDF files
| Action | Description |
|--------|-------------|
| Extract to Markdown | PyMuPDF text extraction + OCR fallback |
| MinerU OCR | AI layout analysis → structured Markdown (best for tables, formulas) |
| Add OCR layer | OCRmyPDF (makes scanned PDFs searchable) |
| Split pages | One PDF per page |

### Word / Office files
| Action | Description |
|--------|-------------|
| DOC/DOCX → Markdown | pandoc |
| DOC/DOCX → PDF | LibreOffice |
| ODT → DOCX | LibreOffice |
| PPT/PPTX → PDF | LibreOffice |
| XLS/XLSX → PDF | LibreOffice |

### Folders
| Action | Description |
|--------|-------------|
| Merge PDFs | Combine all `.pdf` files alphabetically |
| Merge Markdown | Combine all `.md` files alphabetically |
| Create thumbnails | Batch JPEG thumbnails from images and videos |
| RAW → JPEG | Batch convert camera RAW files |

---

## Configuration

Settings live in `.env` at the project root. The install script generates this file automatically from hardware detection. To update it:

```bash
.venv/bin/python3 setup_env.py           # fill in missing values only
.venv/bin/python3 setup_env.py --dry-run # preview without writing
.venv/bin/python3 setup_env.py --force   # re-detect and overwrite all hardware values
```

Key settings:

| Key | Description |
|-----|-------------|
| `AUDIO_UTILS_HF_TOKEN` | HuggingFace token for speaker diarization (transcription) |
| `OMP_NUM_THREADS` | CPU threads for Whisper / torch inference |
| `THUMBNAIL_MAX_GPU_SESSIONS` | Max concurrent GPU encoding sessions |
| `THUMBNAIL_NUM_CORES` | CPU cores for thumbnail generation |
| `RAW_TO_JPG_NUM_CORES` | CPU cores for RAW → JPEG conversion |
| `NAS_HOST` / `NAS_USER` / `NAS_PASS` | NAS credentials (optional) |
