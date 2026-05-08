#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert PowerPoint presentation to PDF using LibreOffice")
    parser.add_argument('ppt_file', help="Path to the PowerPoint file (.ppt, .pptx, .odp)")
    args = parser.parse_args()

    ppt_file = args.ppt_file
    if not os.path.isfile(ppt_file):
        print(f"File {ppt_file} not found")
        sys.exit(1)

    ppt_file_path = os.path.abspath(ppt_file)
    ppt_file_dir = os.path.dirname(ppt_file_path)

    print(f"Converting {ppt_file} to PDF using LibreOffice...")

    try:
        # Run LibreOffice to convert to PDF
        # --headless: run without GUI
        # --convert-to pdf: convert to PDF
        # --outdir: output directory
        cmd = ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', ppt_file_dir, ppt_file_path]
        subprocess.run(cmd, check=True)
        print(f"Successfully converted {ppt_file} to PDF")
    except subprocess.CalledProcessError as e:
        print(f"Error running LibreOffice: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()