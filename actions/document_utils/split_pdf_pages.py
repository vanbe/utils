#!/usr/bin/env python3

import sys
import os
from PyPDF2 import PdfReader, PdfWriter

def split_pdf_pages(input_file):
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist")
        sys.exit(1)

    if not input_file.lower().endswith('.pdf'):
        print("Error: Input file must be a PDF")
        sys.exit(1)

    print(f"Splitting PDF: {input_file}")

    reader = PdfReader(input_file)
    num_pages = len(reader.pages)

    if num_pages == 0:
        print("Error: PDF has no pages")
        sys.exit(1)

    base_name = os.path.splitext(input_file)[0]
    output_dir = os.path.dirname(input_file)

    for page_num in range(num_pages):
        writer = PdfWriter()
        writer.add_page(reader.pages[page_num])
        output_file = f"{base_name}_{page_num + 1}.pdf"
        with open(output_file, 'wb') as out_file:
            writer.write(out_file)
        print(f"Created: {output_file}")

    print(f"Successfully split {input_file} into {num_pages} pages")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_pdf_pages.py <pdf_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    split_pdf_pages(input_file)