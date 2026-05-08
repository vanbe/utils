# -----------------------------------------------------------------------------
# PDF Merger Script
#
# Description: This script finds all .pdf files in a user-specified
#              directory, sorts them alphabetically, and merges them
#              into a single output PDF file.
#
# Author:      Gemini
#
# Prerequisites: Python 3 and the PyPDF2 library.
# To install PyPDF2, run: pip install PyPDF2
# -----------------------------------------------------------------------------

import os
import sys
from PyPDF2 import PdfMerger

def merge_pdfs(pdf_files, output_path):
    """
    Merges the specified PDF files into a single PDF.

    Args:
        pdf_files (list): List of paths to PDF files to merge.
        output_path (str): The path for the output merged PDF file.
    """
    if not pdf_files:
        print("No PDF files provided.")
        return

    print("Merging the following PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf}")

    # --- Merge the PDFs ---
    merger = PdfMerger()
    files_merged = 0

    for pdf_file in pdf_files:
        if not os.path.exists(pdf_file):
            print(f"  [Warning] File '{pdf_file}' does not exist. Skipping.")
            continue
        try:
            merger.append(pdf_file)
            files_merged += 1
        except Exception as e:
            print(f"  [Warning] Could not append '{pdf_file}'. It might be corrupted or password-protected. Skipping. Reason: {e}")

    if files_merged == 0:
        print("Could not merge any files. Please check the warnings above.")
        merger.close()
        return

    # --- Write the merged PDF to a file ---
    try:
        with open(output_path, "wb") as fout:
            merger.write(fout)
        print(f"Success! Merged {files_merged} PDF files into '{output_path}'.")
    except Exception as e:
        print(f"Error: Failed to write the merged PDF. Reason: {e}")
    finally:
        # Clean up the merger object
        merger.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_pdf.py <output_file> <pdf1> <pdf2> ...")
        sys.exit(1)

    output_file = sys.argv[1]
    pdf_files = sys.argv[2:]
    merge_pdfs(pdf_files, output_file)
