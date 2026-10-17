#!/usr/bin/env python3
"""
epub_to_md.py — Convert an EPUB book to Markdown, one section per chapter.

- chapters = the documents of the EPUB spine, in reading order (OPF)
- each chapter: `<!-- chapter N -->` + `## <title>`; the title comes from the
  book's table of contents (nav.xhtml / toc.ncx), else the chapter's first
  heading, else "Chapter N"
- headings inside a chapter are shifted to start at `###`
- each XHTML document is converted with pandoc (GFM, no raw HTML); images are
  replaced by `[image: alt]`; spine documents without text (cover…) are skipped
  and not numbered

Usage:
  python epub_to_md.py <file.epub> [-o OUT.md]
Default output: <basename>.md next to the input.
"""

import argparse
import os
import posixpath
import re
import subprocess
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import unquote


SEP = '─' * 60
R   = '\033[0m'
B   = '\033[1m'
GRN = '\033[32m'
RED = '\033[31m'
DIM = '\033[2m'
HI  = '\033[36m'

_NS = {
    'c':    'urn:oasis:names:tc:opendocument:xmlns:container',
    'opf':  'http://www.idpf.org/2007/opf',
    'dc':   'http://purl.org/dc/elements/1.1/',
    'ncx':  'http://www.daisy.org/z3986/2005/ncx/',
    'xhtml': 'http://www.w3.org/1999/xhtml',
    'epub': 'http://www.idpf.org/2007/ops',
}

_IMG      = re.compile(r'!\[([^\]]*)\]\([^)]*\)')
_HEADING  = re.compile(r'^(#{1,6}) ', re.M)
_EMPTY_LINK_ANCHOR = re.compile(r'\[\]\{#[^}]*\}')


def _text(el) -> str:
    return ' '.join(''.join(el.itertext()).split()) if el is not None else ''


def read_book(zf: zipfile.ZipFile):
    """→ (title, [spine hrefs in order], {href: toc label})."""
    container = ET.fromstring(zf.read('META-INF/container.xml'))
    opf_path = container.find('.//c:rootfile', _NS).get('full-path')
    opf_dir = posixpath.dirname(opf_path)
    opf = ET.fromstring(zf.read(opf_path))

    def resolve(base_dir, href):
        return posixpath.normpath(posixpath.join(base_dir, unquote(href.split('#')[0])))

    title = _text(opf.find('.//dc:title', _NS))
    manifest = {}
    nav_href = None
    for item in opf.findall('.//opf:manifest/opf:item', _NS):
        manifest[item.get('id')] = (resolve(opf_dir, item.get('href')), item.get('media-type'))
        if 'nav' in (item.get('properties') or '').split():
            nav_href = manifest[item.get('id')][0]
    spine_el = opf.find('.//opf:spine', _NS)
    spine = [manifest[i.get('idref')][0] for i in spine_el.findall('opf:itemref', _NS)
             if i.get('idref') in manifest]

    labels = {}
    # EPUB 3 nav document
    if nav_href and nav_href in zf.namelist():
        nav = ET.fromstring(zf.read(nav_href))
        nav_dir = posixpath.dirname(nav_href)
        for navel in nav.iter(f'{{{_NS["xhtml"]}}}nav'):
            if navel.get(f'{{{_NS["epub"]}}}type') not in (None, 'toc'):
                continue
            for a in navel.iter(f'{{{_NS["xhtml"]}}}a'):
                if a.get('href'):
                    labels.setdefault(resolve(nav_dir, a.get('href')), _text(a))
    # EPUB 2 NCX
    toc_id = spine_el.get('toc')
    if not labels and toc_id in manifest and manifest[toc_id][0] in zf.namelist():
        ncx_href = manifest[toc_id][0]
        ncx = ET.fromstring(zf.read(ncx_href))
        for np_ in ncx.iter(f'{{{_NS["ncx"]}}}navPoint'):
            label = _text(np_.find('ncx:navLabel', _NS))
            content = np_.find('ncx:content', _NS)
            if content is not None and content.get('src'):
                labels.setdefault(resolve(posixpath.dirname(ncx_href), content.get('src')), label)
    return title, spine, labels


def xhtml_to_md(data: bytes, tmpdir: str) -> str:
    src = os.path.join(tmpdir, 'chapter.xhtml')
    with open(src, 'wb') as f:
        f.write(data)
    proc = subprocess.run(
        ['pandoc', src, '-f', 'html', '-t', 'gfm-raw_html', '--wrap=none'],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:300])
    md = _IMG.sub(lambda m: f'[image: {m.group(1)}]' if m.group(1).strip() else '[image]', proc.stdout)
    md = _EMPTY_LINK_ANCHOR.sub('', md)
    return re.sub(r'\n{3,}', '\n\n', md).strip()


def _shift_headings(md: str) -> str:
    """Shift headings so the shallowest becomes ### (under the chapter's ##)."""
    levels = [len(m.group(1)) for m in _HEADING.finditer(md)]
    if not levels:
        return md
    delta = 3 - min(levels)
    return _HEADING.sub(lambda m: '#' * max(3, min(6, len(m.group(1)) + delta)) + ' ', md)


def main():
    ap = argparse.ArgumentParser(description='EPUB → Markdown (one ## section per chapter)')
    ap.add_argument('input')
    ap.add_argument('-o', '--output', help='output .md (default: <base>.md beside input)')
    args = ap.parse_args()

    src = os.path.abspath(args.input)
    if not os.path.isfile(src):
        print(f'{RED}File not found:{R} {src}', file=sys.stderr)
        sys.exit(1)
    out_md = os.path.abspath(args.output) if args.output else os.path.splitext(src)[0] + '.md'

    print(SEP)
    print(f'  {B}EPUB → Markdown{R}')
    print(SEP)
    print(f'  {DIM}Input :{R}  {HI}{os.path.basename(src)}{R}')
    print(f'  {DIM}Output:{R}  {HI}{out_md}{R}')

    try:
        with zipfile.ZipFile(src) as zf, tempfile.TemporaryDirectory(prefix='epub_to_md_') as tmp:
            title, spine, labels = read_book(zf)
            names = set(zf.namelist())
            body, n = [], 0
            for href in spine:
                if href not in names:
                    continue
                md = xhtml_to_md(zf.read(href), tmp)
                if not re.sub(r'\[image[^\]]*\]', '', md).strip():
                    continue
                n += 1
                m = re.match(r'\A(#{1,6}) (.+)\n?', md)
                first = m.group(2).strip() if m else ''
                label = labels.get(href) or first.strip('#* ') or f'Chapter {n}'
                if m and first == label:
                    # the chapter's own first heading just repeats the title
                    md = md[m.end():].lstrip()
                md = _shift_headings(md)
                body += [f'<!-- chapter {n} -->', f'## {label}', '', md.strip(), '']
    except Exception as e:
        print(f'  {RED}Error:{R} {e}', file=sys.stderr)
        sys.exit(1)

    title = title or os.path.splitext(os.path.basename(src))[0]
    header = [f'# {title}', '', f'<!-- source: {os.path.basename(src)} | chapters: {n} -->', '']
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(header + body).rstrip() + '\n')

    print(f'  {GRN}✓{R} {n} chapters → {os.path.basename(out_md)}')
    print(SEP)


if __name__ == '__main__':
    main()
