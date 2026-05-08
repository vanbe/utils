#!/usr/bin/env python3
import sys
import os
import subprocess


def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <file.odt>', file=sys.stderr)
        sys.exit(1)

    odt_file = os.path.abspath(sys.argv[1])
    if not os.path.isfile(odt_file):
        print(f'File {odt_file} not found')
        sys.exit(1)

    out_dir = os.path.dirname(odt_file)
    print(f'Converting {os.path.basename(odt_file)} to DOCX using LibreOffice…')

    try:
        cmd = ['libreoffice', '--headless', '--convert-to', 'docx', '--outdir', out_dir, odt_file]
        subprocess.run(cmd, check=True)
        print(f'Done.')
    except subprocess.CalledProcessError as e:
        print(f'Error running LibreOffice: {e}')
        sys.exit(1)
    except FileNotFoundError:
        print('Error: libreoffice not found. Install with: sudo apt install libreoffice')
        sys.exit(1)


if __name__ == '__main__':
    main()
