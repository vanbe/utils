#!/usr/bin/env python3
"""
pdf_mineru_to_md.py — Convert a PDF to Markdown using MinerU.

Wraps the `mineru` CLI from https://github.com/opendatalab/MinerU.

MinerU runs a pipeline of specialised models — DocLayoutYOLO for layout,
UniMERNet for formulas (rendered as LaTeX), RapidTable for tables,
and PaddleOCR for text — to produce structured Markdown with image,
formula and table fidelity.

Hardware notes
--------------
This wrapper defaults to the `pipeline` backend on CUDA. That fits in
roughly 3 GB VRAM and works on a 6 GB card. The `vlm-auto-engine` and
`hybrid-auto-engine` backends use MinerU's own 2 B-param VLM and need
~8 GB+ VRAM — only enable them via --backend if you know it fits.

Models must be pre-downloaded once (~3 GB). Run:
    mineru-models-download -s huggingface -m pipeline
    mineru-models-download -s huggingface -m all       # also for vlm/hybrid

Output layout
-------------
MinerU's native output is nested:
    <out>/<stem>/<method>/<stem>.md
plus images/, layout PDF, content_list.json, etc.

This wrapper reshapes that into a flat layout next to the PDF, with the
markdown promoted to the parent directory so it shows up alongside the
source PDF in a file browser:
    <pdf_dir>/<name>_mineru.md   ← user-facing markdown (image refs rewritten)
    <pdf_dir>/<name>_mineru/
        images/                  ← extracted figures / equations / tables
        <name>_layout.pdf        ← MinerU's debug overlay
        <name>_content_list.json
        <name>_model.json
        ...

Image references inside the markdown are rewritten from `images/foo.jpg`
to `<name>_mineru/images/foo.jpg` (URL-encoded for spaces) so the file
renders correctly from its parent location.

Install
-------
    pip install -U "mineru[core]"

Usage
-----
    python pdf_mineru_to_md.py document.pdf
    python pdf_mineru_to_md.py document.pdf --device cpu
    python pdf_mineru_to_md.py document.pdf --lang fr --method ocr
"""

import argparse
import glob
import html as _html
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse

_SEP = '─' * 56


def _venv_nvidia_lib_dirs() -> list[str]:
    """
    Return every nvidia/<pkg>/lib directory the active venv ships.

    PyTorch 2.11 + CUDA 13 wheels put runtime libs (incl. the NVRTC
    builtins needed for JIT kernel compilation) under nvidia/cu13/lib/,
    but PyTorch's default loader only searches the old nvidia/cuda_*/lib/
    paths. Without this, GPU inference fails with:
        nvrtc: error: failed to open libnvrtc-builtins.so.13.0
    Prepending these dirs to LD_LIBRARY_PATH restores discovery.
    """
    venv_lib = os.path.dirname(os.path.dirname(sys.executable))   # .venv/lib/..  → .venv
    site_pkgs = glob.glob(os.path.join(venv_lib, 'lib', 'python*', 'site-packages'))
    if not site_pkgs:
        return []
    return sorted(glob.glob(os.path.join(site_pkgs[0], 'nvidia', '*', 'lib')))


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s // 60)}m {int(s % 60):02d}s"


def _check_mineru() -> str:
    # Prefer the binary that ships with the active interpreter's venv —
    # the TUI launches scripts via .venv/bin/python3 without putting
    # .venv/bin on PATH, so shutil.which() alone misses it.
    sibling = os.path.join(os.path.dirname(sys.executable), 'mineru')
    if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
        return sibling
    exe = shutil.which('mineru')
    if exe:
        return exe
    print("Error: `mineru` command not found.")
    print("Install with:  pip install -U \"mineru[core]\"")
    print("(Heavy install: pulls PyTorch + ~3 GB of model weights on first run.)")
    sys.exit(1)


def _find_md(workdir: str, stem: str) -> tuple[str, str] | None:
    """
    Locate MinerU's produced markdown inside workdir.

    Returns (md_path, parent_dir) on success — parent_dir contains the
    images/ folder and auxiliary files. Returns None if nothing found.

    MinerU 2.x layout: <workdir>/<stem>/<method>/<stem>.md
    `method` is one of: auto, txt, ocr (pipeline backend) or vlm (vlm backend).
    Walk the tree to be tolerant of minor changes between MinerU versions.
    """
    target = f"{stem}.md"
    for root, _dirs, files in os.walk(workdir):
        if target in files:
            return os.path.join(root, target), root
    # Fallback: any .md under workdir
    for root, _dirs, files in os.walk(workdir):
        for f in files:
            if f.endswith('.md'):
                return os.path.join(root, f), root
    return None


def _flatten_output(src_dir: str, dst_dir: str, stem: str) -> str:
    """
    Move MinerU's output from src_dir into a flat dst_dir.

    Returns the path to the markdown file inside dst_dir.
    """
    os.makedirs(dst_dir, exist_ok=True)
    md_out = ''

    for entry in os.listdir(src_dir):
        src = os.path.join(src_dir, entry)
        dst = os.path.join(dst_dir, entry)
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        shutil.move(src, dst)
        if entry == f"{stem}.md":
            md_out = dst

    return md_out


# Match the path inside a markdown image link `![alt](images/foo.jpg)` —
# group 1 is the prefix up to and including `(`, group 2 is the relative
# `images/...` part. Optional title in `"..."` is preserved.
_RE_MD_IMG  = re.compile(r'(!\[[^\]]*\]\()(images/[^)\s]+)')
# Same, but for HTML <img src="images/...">  /  src='images/...'.
_RE_HTML_IMG = re.compile(r'(<img\b[^>]*?\bsrc=["\'])(images/[^"\']+)')


_RE_TABLE = re.compile(r'<table\b[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
_RE_TR    = re.compile(r'<tr\b[^>]*>(.*?)</tr>',         re.DOTALL | re.IGNORECASE)
_RE_CELL  = re.compile(r'<t[dh]\b[^>]*>(.*?)</t[dh]>',   re.DOTALL | re.IGNORECASE)
_RE_TAG   = re.compile(r'<[^>]+>')


def _clean_cell(raw: str) -> str:
    """Strip nested tags + collapse whitespace + decode HTML entities."""
    return re.sub(r'\s+', ' ', _html.unescape(_RE_TAG.sub('', raw))).strip()


def _table_to_markdown(table_html: str) -> str:
    """
    Convert one HTML <table> to Markdown.

    MinerU's pipeline often misclassifies multi-column magazine sidebars
    ("Idea in Brief", pull quotes, inset boxes) as tables because the
    layout is grid-shaped. Such "tables" contain prose, not data — so
    a GFM table is unreadable. We pick the format by cell length:

      avg cell length > 60 chars → blockquote (prose sidebar)
      otherwise                  → GFM table (real tabular data)

    Empty cells are tolerated; missing trailing cells are padded.
    """
    rows: list[list[str]] = []
    for tr in _RE_TR.findall(table_html):
        cells = [_clean_cell(c) for c in _RE_CELL.findall(tr)]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ''

    flat = [c for row in rows for c in row if c]
    avg_len = sum(len(c) for c in flat) / max(len(flat), 1)

    if avg_len > 60:
        # Prose-like sidebar → flatten to a blockquote (column structure
        # is meaningless for paragraph text). Pipe-separate cells to
        # preserve some sense of grouping without forcing column reading.
        lines: list[str] = []
        for row in rows:
            joined = ' | '.join(c for c in row if c)
            if joined:
                lines.append(f"> {joined}")
        return '\n'.join(lines)

    # Tabular data → GFM table
    ncols = max(len(r) for r in rows)
    padded = [r + [''] * (ncols - len(r)) for r in rows]
    header = '| ' + ' | '.join(padded[0]) + ' |'
    sep    = '|' + '|'.join([' --- '] * ncols) + '|'
    body   = ['| ' + ' | '.join(r) + ' |' for r in padded[1:]]
    return '\n'.join([header, sep, *body])


def _normalize_html_tables(md_text: str) -> str:
    """Replace every <table>...</table> in md_text with its Markdown form."""
    return _RE_TABLE.sub(lambda m: _table_to_markdown(m.group(0)), md_text)


def _rewrite_image_refs(md_path: str, prefix: str) -> None:
    """
    Prepend `prefix/` to relative `images/...` image refs in `md_path`,
    URL-encoding the prefix so spaces and other special chars survive
    Markdown / HTML attribute parsing.
    """
    enc = urllib.parse.quote(prefix)
    with open(md_path, encoding='utf-8') as f:
        text = f.read()
    new_text = _RE_MD_IMG.sub(rf'\g<1>{enc}/\g<2>', text)
    new_text = _RE_HTML_IMG.sub(rf'\g<1>{enc}/\g<2>', new_text)
    if new_text != text:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(new_text)


def _promote_md_to_parent(asset_dir: str, stem: str, normalize_tables: bool = True) -> str:
    """
    Move <asset_dir>/<stem>.md up to <parent>/<asset_dir_basename>.md,
    rewrite its `images/...` references so they resolve from the parent,
    and (by default) convert HTML tables to Markdown.

    Returns the new public path of the markdown file.
    """
    inner_md = os.path.join(asset_dir, f"{stem}.md")
    if not os.path.isfile(inner_md):
        # Nothing to promote — return whatever is there.
        return inner_md

    parent = os.path.dirname(asset_dir)
    folder_name = os.path.basename(asset_dir)
    public_md = os.path.join(parent, f"{folder_name}.md")

    if os.path.exists(public_md):
        os.remove(public_md)
    shutil.move(inner_md, public_md)
    _rewrite_image_refs(public_md, folder_name)
    if normalize_tables:
        with open(public_md, encoding='utf-8') as f:
            text = f.read()
        new_text = _normalize_html_tables(text)
        if new_text != text:
            with open(public_md, 'w', encoding='utf-8') as f:
                f.write(new_text)
    return public_md


def main():
    parser = argparse.ArgumentParser(
        description='Convert a PDF to Markdown using MinerU (layout + OCR + formulas + tables).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('pdf_file', help='Input PDF file')
    parser.add_argument('--method', default='auto', choices=['auto', 'txt', 'ocr'],
                        help='Pipeline parsing method: auto (detect), txt (text-layer only), '
                             'ocr (force OCR). Default: auto.')
    parser.add_argument('--lang', default='en', metavar='CODE',
                        help='OCR language code: en, ch, ch_server, ch_lite, korean, japan, '
                             'chinese_cht, ta, te, ka, th, el, latin, arabic, east_slavic, '
                             'cyrillic, devanagari. Use `latin` for French / Spanish / Italian / '
                             'German / Dutch. (Default: en)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Inference device (default: cuda)')
    parser.add_argument('--backend', default='pipeline',
                        choices=['pipeline', 'vlm-auto-engine', 'hybrid-auto-engine'],
                        help='MinerU backend. pipeline = low-VRAM specialist models (default, '
                             '~3 GB VRAM). vlm-auto-engine / hybrid-auto-engine = MinerU 2 B-param '
                             'VLM, needs ~8 GB+ VRAM.')
    # MinerU's pipeline backend picks batch sizes from a VRAM tier table:
    #   <6 GB → ratio 1   |   ≥6 → 2   |   ≥8 → 4   |   ≥16 → 8   |   ≥32 → 16
    # Auto-detect on a 6 GB card lands on ratio 2. Lying upward (e.g. 8) doubles
    # batch sizes for layout / OCR detection / formula recognition with usually
    # plenty of headroom left on a 6 GB card. Set to 0 to let MinerU auto-detect.
    parser.add_argument('--virtual-vram', type=int, default=8, metavar='GB',
                        help='Trick MinerU into using larger batches by reporting this VRAM '
                             'budget (in GB). Default: 8 → batch ratio 4 (a 2× bump on a 6 GB '
                             'card, still well within physical VRAM). Use 0 to auto-detect, '
                             '16 for an aggressive ratio of 8 (likely OOM on smaller cards).')
    parser.add_argument('--output-dir', default='', metavar='DIR',
                        help='Output folder (default: <pdf_dir>/<name>_mineru/)')
    parser.add_argument('--keep-nested', action='store_true',
                        help='Keep MinerU\'s nested <stem>/<method>/ subfolders instead of flattening.')
    parser.add_argument('--keep-html-tables', action='store_true',
                        help='Leave HTML <table> blocks as-is. By default they are converted: '
                             'short cells → GFM table, long-prose cells (misclassified sidebars) '
                             '→ Markdown blockquote. Use this if downstream tools rely on the '
                             'raw HTML output.')
    args = parser.parse_args()

    pdf_path = os.path.abspath(args.pdf_file)
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)
    if not pdf_path.lower().endswith('.pdf'):
        print("Input must be a .pdf file.")
        sys.exit(1)

    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = (os.path.abspath(args.output_dir) if args.output_dir
               else os.path.join(os.path.dirname(pdf_path), f"{stem}_mineru"))

    if os.path.exists(out_dir):
        print(f"Output folder already exists: {out_dir}")
        print("Delete it first or pass --output-dir to choose a different location.")
        sys.exit(1)

    mineru = _check_mineru()

    # ── Header ──────────────────────────────────────────────────────────────
    print(_SEP)
    print(f"  {os.path.basename(pdf_path)}")
    print(f"  backend: {args.backend}  ·  device: {args.device}"
          f"  ·  method: {args.method}  ·  lang: {args.lang}")
    if args.virtual_vram > 0 and args.device != 'cpu':
        # Show which batch tier we're requesting so the user can see / tune it.
        v = args.virtual_vram
        if   v >= 32: tier = 16
        elif v >= 16: tier = 8
        elif v >=  8: tier = 4
        elif v >=  6: tier = 2
        else:         tier = 1
        print(f"  vram:    pretend {v} GB  ·  batch ratio {tier}×")
    print(f"  output:  {out_dir}")
    if args.backend != 'pipeline':
        print(f"  warning: {args.backend} needs ~8 GB+ VRAM — may OOM on smaller cards")
    print(_SEP)

    t0 = time.time()

    # MinerU writes into a temp workdir; we relocate after success so a partial
    # / failed run never leaves a half-populated <name>_mineru/ folder behind.
    with tempfile.TemporaryDirectory(prefix='mineru_') as workdir:
        cmd = [
            mineru,
            '-p', pdf_path,
            '-o', workdir,
            '-b', args.backend,
            '-l', args.lang,
        ]
        # --method only applies to pipeline / hybrid-* backends; mineru rejects it for vlm-*.
        if args.backend == 'pipeline' or args.backend.startswith('hybrid-'):
            cmd += ['-m', args.method]

        # Device selection happens via env var, not a CLI flag.
        env = os.environ.copy()
        env['MINERU_DEVICE_MODE'] = args.device

        # Override MinerU's auto-detected VRAM tier to push batch sizes up.
        # Only meaningful on GPU; on CPU batch ratio is irrelevant.
        if args.virtual_vram > 0 and args.device != 'cpu':
            env['MINERU_VIRTUAL_VRAM_SIZE'] = str(args.virtual_vram)

        # Make CUDA 13 NVRTC discoverable for PyTorch 2.11 + cu13 wheels.
        # Harmless on CPU runs; cheap to set unconditionally.
        nv_dirs = _venv_nvidia_lib_dirs()
        if nv_dirs:
            existing = env.get('LD_LIBRARY_PATH', '')
            env['LD_LIBRARY_PATH'] = ':'.join(nv_dirs + ([existing] if existing else []))

        cmd_preview = f"  $ MINERU_DEVICE_MODE={args.device}"
        if 'MINERU_VIRTUAL_VRAM_SIZE' in env:
            cmd_preview += f" MINERU_VIRTUAL_VRAM_SIZE={env['MINERU_VIRTUAL_VRAM_SIZE']}"
        cmd_preview += ' ' + ' '.join(cmd)
        print(cmd_preview)
        print()

        try:
            # cwd must be a Linux path — mineru spawns a fast_api subprocess that
            # inherits CWD, and Python's importlib crashes when it tries to scan a
            # /mnt/c/ path (WSL DrvFS + spaces/dashes → errno 22).
            result = subprocess.run(cmd, env=env, cwd=workdir)
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            sys.exit(130)

        if result.returncode != 0:
            print(f"\n  MinerU exited with code {result.returncode}.")
            sys.exit(result.returncode)

        found = _find_md(workdir, stem)
        if not found:
            print(f"\n  Could not locate produced .md under {workdir}.")
            print("  MinerU finished without error but the output layout is unexpected.")
            sys.exit(1)

        md_src, src_parent = found

        if args.keep_nested:
            shutil.move(workdir, out_dir)
            md_out = os.path.join(out_dir, os.path.relpath(md_src, workdir))
        else:
            inner_md = _flatten_output(src_parent, out_dir, stem)
            # Promote the .md to the parent directory so it sits alongside
            # the source PDF (visible in file browsers). Image refs get
            # rewritten to point into the asset folder; HTML tables get
            # converted to Markdown unless --keep-html-tables is passed.
            md_out = _promote_md_to_parent(
                out_dir, stem, normalize_tables=not args.keep_html_tables,
            )
            if md_out == inner_md:
                # Promotion didn't move anything (file missing); keep what we have.
                pass

    # ── Summary ─────────────────────────────────────────────────────────────
    print()
    print(_SEP)
    print(f"  Total:  {_elapsed(t0)}")
    print(f"  Saved:  {md_out}")
    if not args.keep_nested:
        print(f"  Assets: {out_dir}/")
    print(_SEP)


if __name__ == '__main__':
    main()
