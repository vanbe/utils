#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert Markdown to Docx")
    parser.add_argument('md_file', help="Path to the markdown file")
    args = parser.parse_args()

    md_file = args.md_file
    if not os.path.isfile(md_file):
        print(f"File {md_file} not found")
        sys.exit(1)
        
    md_file_path = os.path.abspath(md_file)
    md_file_dir = os.path.dirname(md_file_path)
    
    # Logic for output filename
    md_file_basename = os.path.basename(md_file_path)
    base_name = os.path.splitext(md_file_basename)[0]
    output_docx = os.path.join(md_file_dir, f'{base_name}.docx')

    print(f"Converting {md_file} to {output_docx}...")

    try:
        # Run pandoc
        cmd = ['pandoc', md_file, '-o', output_docx]
        subprocess.run(cmd, check=True)
        print(f"Successfully created: {output_docx}")
    except subprocess.CalledProcessError as e:
        print(f"Error running pandoc: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
