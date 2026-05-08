#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert Word document to PDF using LibreOffice")
    parser.add_argument('doc_file', help="Path to the Word document file (.doc, .docx, .odt)")
    args = parser.parse_args()

    doc_file = args.doc_file
    if not os.path.isfile(doc_file):
        print(f"File {doc_file} not found")
        sys.exit(1)

    doc_file_path = os.path.abspath(doc_file)
    doc_file_dir = os.path.dirname(doc_file_path)

    print(f"Converting {doc_file} to PDF using LibreOffice...")

    try:
        # Run LibreOffice to convert to PDF
        # --headless: run without GUI
        # --convert-to pdf: convert to PDF
        # --outdir: output directory
        cmd = ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', doc_file_dir, doc_file_path]
        subprocess.run(cmd, check=True)
        print(f"Successfully converted {doc_file} to PDF")
    except subprocess.CalledProcessError as e:
        print(f"Error running LibreOffice: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()