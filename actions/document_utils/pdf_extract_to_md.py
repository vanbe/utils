#!/usr/bin/env python3

import sys
import os
import argparse
import tempfile
import subprocess
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

def extract_text_and_ocr(pdf_path, lang='fra'):
    # First, try to extract text directly from the PDF
    doc = fitz.open(pdf_path)
    text_content = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = str(page.get_text()).strip()
        text_content.append(f"# Page {page_num + 1}\n\n{page_text}\n\n")

    doc.close()
    full_text = "\n".join(text_content).strip()

    # If text is present, return it
    if full_text:
        return full_text

    # If no text, perform OCR
    # Check if ocrmypdf is installed
    if not subprocess.run(['which', 'ocrmypdf'], capture_output=True).returncode == 0:
        print("Error: OCRmyPDF is not installed. Please run the install script: ./pdf_extract_to_md_install.sh")
        sys.exit(1)

    # Create temp file for OCR'd PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        temp_pdf = temp_file.name

    try:
        # Run OCRmyPDF
        cmd = ['ocrmypdf', '--language', lang, pdf_path, temp_pdf]
        subprocess.run(cmd, check=True, capture_output=True)

        # Extract text from OCR'd PDF
        doc = fitz.open(temp_pdf)
        text_content = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = str(page.get_text()).strip()
            text_content.append(f"# Page {page_num + 1}\n\n{page_text}\n\n")

        doc.close()
        return "\n".join(text_content)
    finally:
        if os.path.exists(temp_pdf):
            os.unlink(temp_pdf)

def main():
    parser = argparse.ArgumentParser(description="Extract PDF content to Markdown")
    parser.add_argument("pdf_file", help="Path to the PDF file")
    parser.add_argument("--language", choices=['fra', 'eng'], default='fra', help="Language for OCR (default: fra)")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_file):
        print(f"Error: {args.pdf_file} does not exist")
        sys.exit(1)

    print("Starting PDF extraction...")
    md_content = extract_text_and_ocr(args.pdf_file, args.language)

    base_name = os.path.splitext(args.pdf_file)[0]
    output_md = f"{base_name}_extracted.md"

    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"Extraction complete!")
    print(f"Markdown file: {output_md}")

if __name__ == "__main__":
    main()