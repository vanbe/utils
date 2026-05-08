#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert Excel spreadsheet to PDF using LibreOffice")
    parser.add_argument('xls_file', help="Path to the Excel file (.xls, .xlsx, .ods)")
    args = parser.parse_args()

    xls_file = args.xls_file
    if not os.path.isfile(xls_file):
        print(f"File {xls_file} not found")
        sys.exit(1)

    xls_file_path = os.path.abspath(xls_file)
    xls_file_dir = os.path.dirname(xls_file_path)

    print(f"Converting {xls_file} to PDF using LibreOffice...")

    try:
        # Run LibreOffice to convert to PDF
        # --headless: run without GUI
        # --convert-to pdf: convert to PDF
        # --outdir: output directory
        cmd = ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', xls_file_dir, xls_file_path]
        subprocess.run(cmd, check=True)
        print(f"Successfully converted {xls_file} to PDF")
    except subprocess.CalledProcessError as e:
        print(f"Error running LibreOffice: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()