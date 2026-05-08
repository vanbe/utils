#!/usr/bin/env python3
"""
pdf_ocr.py — OCR a PDF and either embed the text layer or extract to Markdown.

Usage:
  python pdf_ocr.py file.pdf                          # add OCR layer → file_ocr.pdf
  python pdf_ocr.py file.pdf --output md              # OCR then extract → file.md
  python pdf_ocr.py file.pdf --mode force             # discard existing text, re-OCR
  python pdf_ocr.py file.pdf --output md --mode redo  # re-OCR then extract
"""

import sys
import os
import argparse
import subprocess
import tempfile
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))


def _check_ocrmypdf():
    if subprocess.run(['which', 'ocrmypdf'], capture_output=True).returncode != 0:
        print("Error: OCRmyPDF is not installed. Run: sudo apt install ocrmypdf")
        sys.exit(1)


def _run_ocrmypdf(input_path: str, output_path: str, lang: str, mode: str):
    """Run ocrmypdf and return exit code."""
    cmd = ['ocrmypdf', '--language', lang]
    if mode == 'skip':
        cmd.append('--skip-text')
    elif mode == 'redo':
        cmd.append('--redo-ocr')
    elif mode == 'force':
        cmd.append('--force-ocr')
    cmd += [input_path, output_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: OCR failed with exit code {result.returncode}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)


def _extract_text_to_md(pdf_path: str, output_md: str):
    """Extract text from a PDF (assumed to have a text layer) into Markdown."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("Error: PyMuPDF not installed. Run: pip install PyMuPDF")
        sys.exit(1)

    doc = fitz.open(pdf_path)
    sections = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            sections.append(f"## Page {i + 1}\n\n{text}")
    doc.close()

    content = "\n\n---\n\n".join(sections) if sections else "(no text extracted)"
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(content + '\n')


def ocr_to_pdf(pdf_path: str, lang: str = 'fra', mode: str = 'skip') -> str:
    """Add OCR text layer to PDF. Returns path to output PDF."""
    _check_ocrmypdf()
    base       = os.path.splitext(pdf_path)[0]
    output_pdf = f"{base}_ocr.pdf"
    print(f"Running OCR ({lang}, mode={mode})…")
    _run_ocrmypdf(pdf_path, output_pdf, lang, mode)
    return output_pdf


def ocr_to_md(pdf_path: str, lang: str = 'fra', mode: str = 'skip') -> str:
    """OCR the PDF then extract text to Markdown. Returns path to output .md."""
    _check_ocrmypdf()
    base       = os.path.splitext(pdf_path)[0]
    output_md  = f"{base}_ocr.md"

    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp_pdf = tmp.name
    try:
        print(f"Running OCR ({lang}, mode={mode})…")
        _run_ocrmypdf(pdf_path, tmp_pdf, lang, mode)
        print("Extracting text to Markdown…")
        _extract_text_to_md(tmp_pdf, output_md)
    finally:
        if os.path.exists(tmp_pdf):
            os.unlink(tmp_pdf)

    return output_md


def main():
    parser = argparse.ArgumentParser(description='OCR a PDF — embed text layer or extract to Markdown')
    parser.add_argument('pdf_file', help='Path to the PDF file')
    parser.add_argument('--language', choices=['fra', 'eng'], default='fra',
                        help='OCR language (default: fra)')
    parser.add_argument('--mode', choices=['skip', 'redo', 'force'], default='skip',
                        help='How to handle pages that already have text: '
                             'skip (default) · redo · force')
    parser.add_argument('--output', choices=['pdf', 'md'], default='pdf',
                        help='Output format: pdf = embed OCR layer (default), '
                             'md = extract text to Markdown')
    args = parser.parse_args()

    if not os.path.exists(args.pdf_file):
        print(f"Error: {args.pdf_file} does not exist")
        sys.exit(1)

    if args.output == 'md':
        out = ocr_to_md(args.pdf_file, args.language, args.mode)
        print(f"Done → {out}")
    else:
        out = ocr_to_pdf(args.pdf_file, args.language, args.mode)
        print(f"Done → {out}")


if __name__ == '__main__':
    main()
