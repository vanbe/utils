#!/usr/bin/env python3
"""
utils_run.py — Non-interactive CLI for utils_tools document actions.

Designed to be called by automated agents (e.g. Claude in another project).
All progress/logs from underlying scripts go to stderr.
stdout carries only the final JSON result.

Usage:
    utils_run.py <file>                       List available actions for <file>
    utils_run.py <file> <action>              Run a file action on <file>
    utils_run.py --list-all                   List every registered file action
    utils_run.py --list-folders               List every folder action
    utils_run.py --folder <dir> <action> ..   Run a folder action on <dir>
                                              (extra args are passed through to the script)

Exit codes: 0 = ok / list printed, 1 = error, 2 = bad usage

Folder actions (operate on a directory):
    image-dedup      <dir>  → duplicates.json   (exact SHA-256 duplicate groups)
    raw-to-jpg       <dir>  → JPEGs in place     (rawpy + exiftool)
    thumbnails       <dir>  → thumbnails         (Pillow/ffmpeg; supports --mirror)

JSON — list mode:
    [{"action": "doc-to-md", "desc": "..."}, ...]

JSON — run mode (success):
    {"status": "ok", "output_file": "/abs/path/to/result.md"}

JSON — run mode (failure):
    {"status": "error", "message": "..."}

Available actions:
    doc-to-md        .docx/.doc  → Markdown      (pandoc)
    doc-to-pdf       .docx/.doc  → PDF            (LibreOffice)
    odt-to-docx      .odt        → DOCX           (LibreOffice)
    odt-to-pdf       .odt        → PDF            (LibreOffice)
    pdf-extract      .pdf        → Markdown       (PyMuPDF + OCR fallback)
    pdf-mineru       .pdf        → Markdown       (MinerU — layout + formulas)
    pdf-vision-ocr   .pdf        → Markdown       (AI vision model via Ollama)
    md-to-docx       .md         → DOCX           (pandoc)
    md-to-pdf        .md         → PDF            (pandoc + XeLaTeX)
    ppt-to-pdf       .pptx/.ppt  → PDF            (LibreOffice)
    xls-to-pdf       .xlsx/.xls  → PDF            (LibreOffice)

Example — extract all PDFs in a folder to Markdown:
    for f in /path/to/folder/*.pdf; do
        utils_run.py "$f" pdf-extract
    done
"""
import json
import os
import subprocess
import sys

_DIR    = os.path.dirname(os.path.abspath(__file__))
# venv du projet si présent (WSL), sinon le python courant (portable / CI / conteneur).
_PYTHON = os.path.join(_DIR, '.venv', 'bin', 'python3')
if not os.path.exists(_PYTHON):
    _PYTHON = sys.executable
_DOC    = os.path.join(_DIR, 'actions', 'document_utils')
_AI     = os.path.join(_DIR, 'actions', 'ai_utils')
_PIC    = os.path.join(_DIR, 'actions', 'picture_utils')


def _stem(path: str) -> str:
    """Absolute path without extension: /dir/base"""
    return os.path.splitext(os.path.abspath(path))[0]


# Registry: slug → {script, extra (optional), output, ext, desc}
_REGISTRY: dict[str, dict] = {
    'doc-to-md': {
        'script': os.path.join(_DOC, 'doc_to_md.py'),
        'output': lambda p: _stem(p) + '.md',
        'ext':    {'.docx', '.doc'},
        'desc':   'Convert Word document to Markdown (pandoc)',
    },
    'doc-to-pdf': {
        'script': os.path.join(_DOC, 'doc_to_pdf.py'),
        'output': lambda p: _stem(p) + '.pdf',
        'ext':    {'.docx', '.doc', '.odt'},
        'desc':   'Convert Word/ODT document to PDF (LibreOffice)',
    },
    'odt-to-docx': {
        'script': os.path.join(_DOC, 'odt_to_docx.py'),
        'output': lambda p: _stem(p) + '.docx',
        'ext':    {'.odt'},
        'desc':   'Convert ODT to DOCX (LibreOffice)',
    },
    'odt-to-pdf': {
        'script': os.path.join(_DOC, 'doc_to_pdf.py'),
        'output': lambda p: _stem(p) + '.pdf',
        'ext':    {'.odt'},
        'desc':   'Convert ODT to PDF (LibreOffice)',
    },
    'pdf-extract': {
        'script': os.path.join(_DOC, 'pdf_extract_to_md.py'),
        'output': lambda p: _stem(p) + '_extracted.md',
        'ext':    {'.pdf'},
        'desc':   'Extract PDF text to Markdown (PyMuPDF + OCR fallback)',
    },
    'pdf-mineru': {
        'script': os.path.join(_DOC, 'pdf_mineru_to_md.py'),
        'output': lambda p: _stem(p) + '_mineru.md',
        'ext':    {'.pdf'},
        'desc':   'Extract PDF to Markdown (MinerU — layout + formulas + tables)',
    },
    'pdf-vision-ocr': {
        'script': os.path.join(_AI, 'pdf_vision_ocr.py'),
        'extra':  lambda p: ['-o', _stem(p) + '_ocr.md'],
        'output': lambda p: _stem(p) + '_ocr.md',
        'ext':    {'.pdf'},
        'desc':   'OCR PDF pages with AI vision model (requires Ollama)',
    },
    'md-to-docx': {
        'script': os.path.join(_DOC, 'md_to_docx.py'),
        'output': lambda p: _stem(p) + '.docx',
        'ext':    {'.md'},
        'desc':   'Export Markdown to DOCX (pandoc)',
    },
    'md-to-pdf': {
        'script': os.path.join(_DOC, 'md_to_pdf', 'md_to_pdf.py'),
        'output': lambda p: _stem(p) + '.pdf',
        'ext':    {'.md'},
        'desc':   'Export Markdown to PDF (pandoc + XeLaTeX)',
    },
    'ppt-to-pdf': {
        'script': os.path.join(_DOC, 'ppt_to_pdf.py'),
        'output': lambda p: _stem(p) + '.pdf',
        'ext':    {'.pptx', '.ppt'},
        'desc':   'Convert PowerPoint to PDF (LibreOffice)',
    },
    'xls-to-pdf': {
        'script': os.path.join(_DOC, 'xls_to_pdf.py'),
        'output': lambda p: _stem(p) + '.pdf',
        'ext':    {'.xlsx', '.xls'},
        'desc':   'Convert spreadsheet to PDF (LibreOffice)',
    },
}


# Folder-level operations (operate on a directory, not a single file).
# Source of truth for these scripts: both this CLI and the TUI read it here, so a
# script path can't drift in one place and break the other.
# Each entry: slug → {script, output (lambda(dir)->path|None), desc}.
_FOLDER_REGISTRY: dict[str, dict] = {
    'image-dedup': {
        'script': os.path.join(_PIC, 'image_dedup.py'),
        'output': lambda d: os.path.join(d, 'duplicates.json'),
        'desc':   'Duplicate images → JSON of path groups (--method exact|perceptual)',
    },
    'image-index': {
        'script': os.path.join(_PIC, 'image_index.py'),
        'output': lambda d: os.path.join(d, '.image_index.json'),
        'desc':   'Build/refresh a SHA-256 index of a library (incremental, --exclude dirs)',
    },
    'raw-to-jpg': {
        'script': os.path.join(_PIC, 'raw2jpeg.py'),
        'output': None,        # in-place, many outputs
        'desc':   'Convert every RAW image in the folder to JPEG (rawpy + exiftool)',
    },
    'thumbnails': {
        'script': os.path.join(_PIC, 'thumbnailing.py'),
        'output': None,        # thumbnails written beside/into target folder
        'desc':   'Generate JPEG/video thumbnails for the folder (supports --mirror)',
    },
}


def _actions_for(path: str) -> list[dict]:
    ext = os.path.splitext(path)[1].lower()
    return [
        {'action': slug, 'desc': info['desc']}
        for slug, info in _REGISTRY.items()
        if ext in info['ext']
    ]


def _run(path: str, action: str) -> dict:
    path = os.path.abspath(path)

    if not os.path.isfile(path):
        return {'status': 'error', 'message': f'File not found: {path}'}

    if action not in _REGISTRY:
        return {
            'status':  'error',
            'message': f'Unknown action {action!r}. Run with just <file> to list valid actions.',
        }

    info = _REGISTRY[action]
    ext  = os.path.splitext(path)[1].lower()
    if ext not in info['ext']:
        return {
            'status':  'error',
            'message': (
                f'Action {action!r} does not apply to {ext} files. '
                f'Supported extensions: {sorted(info["ext"])}'
            ),
        }

    extra = info.get('extra', lambda p: [])(path)
    cmd   = [_PYTHON, info['script'], path] + extra

    # Route all script output to our stderr so stdout stays clean for JSON.
    proc = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)

    if proc.returncode != 0:
        return {'status': 'error', 'message': f'Script exited with code {proc.returncode}'}

    out_file = info['output'](path)
    if not os.path.isfile(out_file):
        return {
            'status':  'error',
            'message': f'Script succeeded but expected output not found: {out_file}',
        }

    return {'status': 'ok', 'output_file': out_file}


def _run_folder(directory: str, action: str, extra: list = None) -> dict:
    directory = os.path.abspath(directory)

    if not os.path.isdir(directory):
        return {'status': 'error', 'message': f'Folder not found: {directory}'}

    if action not in _FOLDER_REGISTRY:
        return {
            'status':  'error',
            'message': f'Unknown folder action {action!r}. Run --list-folders for valid actions.',
        }

    info = _FOLDER_REGISTRY[action]
    cmd  = [_PYTHON, info['script'], directory] + list(extra or [])

    # Route all script output to our stderr so stdout stays clean for JSON.
    proc = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)

    if proc.returncode != 0:
        return {'status': 'error', 'message': f'Script exited with code {proc.returncode}'}

    out = info['output'](directory) if info.get('output') else None
    if out is not None and not os.path.isfile(out):
        return {
            'status':  'error',
            'message': f'Script succeeded but expected output not found: {out}',
        }

    result = {'status': 'ok'}
    if out is not None:
        result['output_file'] = out
    return result


def _emit(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    argv = sys.argv[1:]

    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__, file=sys.stderr)
        sys.exit(0 if argv and argv[0] in ('-h', '--help') else 2)

    if argv[0] == '--list-all':
        _emit([{'action': slug, 'desc': info['desc'], 'ext': sorted(info['ext'])}
               for slug, info in _REGISTRY.items()])
        return

    if argv[0] == '--list-folders':
        _emit([{'action': slug, 'desc': info['desc']}
               for slug, info in _FOLDER_REGISTRY.items()])
        return

    if argv[0] == '--folder':
        if len(argv) < 3:
            _emit({'status': 'error',
                   'message': 'Usage: utils_run.py --folder <dir> <action> [opts...]'})
            sys.exit(2)
        directory, action, extra = argv[1], argv[2], argv[3:]
        result = _run_folder(directory, action, extra)
        _emit(result)
        sys.exit(0 if result.get('status') == 'ok' else 1)

    file_path = argv[0]

    if len(argv) == 1:
        if not os.path.isfile(file_path):
            _emit({'status': 'error', 'message': f'File not found: {file_path}'})
            sys.exit(1)
        _emit(_actions_for(file_path))
        return

    action = argv[1]
    result = _run(file_path, action)
    _emit(result)
    sys.exit(0 if result.get('status') == 'ok' else 1)


if __name__ == '__main__':
    main()
