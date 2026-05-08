#!/usr/bin/env python3
"""
doc_to_md.py — Convert a Word document (.docx / .doc) to clean Markdown.

- GitHub-Flavored Markdown: proper pipe tables, fenced code blocks
- Images extracted to <basename>_images/ next to the output .md
- YAML frontmatter with document title when available
- No hard line-wrapping, ATX-style headings (#)

Usage:
  python doc_to_md.py <file.docx>
"""

import json
import os
import re
import shutil
import subprocess
import sys


SEP = '─' * 60
R   = '\033[0m'
B   = '\033[1m'
GRN = '\033[32m'
YEL = '\033[33m'
RED = '\033[31m'
DIM = '\033[2m'
HI  = '\033[36m'


# ---------------------------------------------------------------------------
# Title extraction from pandoc JSON AST
# ---------------------------------------------------------------------------

def _ast_inlines_to_text(nodes: list) -> str:
    parts = []
    for n in nodes:
        t = n.get('t', '')
        if t == 'Str':
            parts.append(n['c'])
        elif t == 'Space':
            parts.append(' ')
        elif t == 'SoftBreak':
            parts.append(' ')
        elif t == 'Emph' or t == 'Strong':
            parts.append(_ast_inlines_to_text(n.get('c', [])))
    return ''.join(parts)


def extract_title(doc_path: str) -> str:
    try:
        result = subprocess.run(
            ['pandoc', doc_path, '-t', 'json'],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return ''
        ast = json.loads(result.stdout)
        title_node = ast.get('meta', {}).get('title', {})
        if title_node.get('t') == 'MetaInlines':
            return _ast_inlines_to_text(title_node.get('c', [])).strip()
    except Exception:
        pass
    return ''


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

_OVER_ESCAPED = re.compile(r'\\([`*_{}[\]()#+\-.!<>|~^])')


def clean_markdown(text: str) -> str:
    # Remove pandoc's over-eager backslash escaping
    text = _OVER_ESCAPED.sub(r'\1', text)

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip trailing whitespace from each line
    lines = [l.rstrip() for l in text.splitlines()]
    return '\n'.join(lines).strip() + '\n'


def flatten_images(img_dir: str, out_md: str) -> int:
    """
    pandoc always creates <img_dir>/media/<files>. Move files up to <img_dir>/
    and fix the references in the markdown. Returns the count of images.
    """
    media_sub = os.path.join(img_dir, 'media')
    if os.path.isdir(media_sub):
        for fname in os.listdir(media_sub):
            src = os.path.join(media_sub, fname)
            dst = os.path.join(img_dir, fname)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        try:
            os.rmdir(media_sub)
        except OSError:
            pass
        # Fix references in the markdown
        img_dir_name = os.path.basename(img_dir)
        with open(out_md, 'r', encoding='utf-8') as f:
            content = f.read()
        content = content.replace(f'{img_dir_name}/media/', f'{img_dir_name}/')
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(content)

    if not os.path.isdir(img_dir):
        return 0
    return len([f for f in os.listdir(img_dir) if not f.startswith('.')])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <file.docx>', file=sys.stderr)
        sys.exit(1)

    doc_file = os.path.abspath(sys.argv[1])
    if not os.path.isfile(doc_file):
        print(f'{RED}File not found:{R} {doc_file}', file=sys.stderr)
        sys.exit(1)

    doc_dir  = os.path.dirname(doc_file)
    base     = os.path.splitext(os.path.basename(doc_file))[0]
    out_md   = os.path.join(doc_dir, f'{base}.md')
    img_dir_name = f'{base}_images'
    img_dir  = os.path.join(doc_dir, img_dir_name)

    print(SEP)
    print(f'  {B}DOC → Markdown{R}')
    print(SEP)
    print(f'  {DIM}Input :{R}  {HI}{os.path.basename(doc_file)}{R}')
    print(f'  {DIM}Output:{R}  {HI}{os.path.basename(out_md)}{R}')
    print(f'  {DIM}Images:{R}  {HI}{img_dir_name}/{R}')
    print()

    # Step 1 — extract title from document metadata
    print(f'  {DIM}Extracting title…{R}')
    title = extract_title(doc_file)
    if title:
        print(f'  {DIM}Title :{R}  {title}')

    # Step 2 — convert with pandoc
    print(f'  {DIM}Converting with pandoc…{R}')
    cmd = [
        'pandoc', doc_file,
        '--from=docx',
        '--to=gfm',                        # GitHub-Flavored Markdown: pipe tables, fenced code
        f'--extract-media={img_dir_name}',  # relative to cwd=doc_dir → lands in doc_dir
        '--wrap=none',                      # no hard mid-paragraph line-wrapping
        '--markdown-headings=atx',          # # H1 style, not underline style
        '-o', out_md,
    ]
    try:
        result = subprocess.run(cmd, cwd=doc_dir, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f'\n{RED}pandoc error:{R}\n{result.stderr}', file=sys.stderr)
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f'\n{RED}Error: pandoc timed out after 120 s{R}', file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f'\n{RED}Error: pandoc not found.{R}  Install with: sudo apt install pandoc', file=sys.stderr)
        sys.exit(1)

    # Step 3 — flatten images subfolder and fix references
    n_images = flatten_images(img_dir, out_md)

    # Step 4 — post-process markdown
    with open(out_md, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    content = clean_markdown(content)

    # Step 5 — prepend YAML frontmatter if title available
    if title:
        safe_title = title.replace('"', "'")
        content = f'---\ntitle: "{safe_title}"\n---\n\n' + content

    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(content)

    # Summary
    n_lines = content.count('\n')
    print()
    print(SEP)
    print(f'  {GRN}✓ Done{R}')
    print(f'  {DIM}Lines :{R}  {n_lines}')
    if n_images:
        print(f'  {DIM}Images:{R}  {n_images} → {img_dir_name}/')
    else:
        print(f'  {DIM}Images:{R}  none')
    print(SEP)


if __name__ == '__main__':
    main()
