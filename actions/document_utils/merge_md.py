#!/usr/bin/env python3

import sys
import os
import shlex
import argparse

def collect_md_files(folder, recursive=False):
    files = []
    if recursive:
        for root, dirs, filenames in os.walk(folder):
            for f in filenames:
                if f.endswith('.md'):
                    files.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            if f.endswith('.md'):
                files.append(os.path.join(folder, f))
    return files

def merge_md(files, output):
    files.sort()  # alphabetical order
    with open(output, 'w') as out:
        for f in files:
            filename = os.path.splitext(os.path.basename(f))[0]
            out.write(f"# {filename}\n\n")
            with open(f, 'r') as infile:
                out.write(infile.read())
            out.write('\n\n')  # separator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple .md files into one")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--files', nargs='+', help='List of .md files to merge')
    group.add_argument('--folder', nargs='+', help='Folder path to merge all .md files from')
    parser.add_argument('--recursive', action='store_true', help='Recursively collect .md files from subfolders')
    args = parser.parse_args()

    if args.folder:
        folder = ' '.join(args.folder).strip('"')
        files = collect_md_files(folder, args.recursive)
        if len(files) < 2:
            print("Error: Need at least 2 .md files in the folder")
            sys.exit(1)
        base_dir = os.path.dirname(folder)
        folder_name = os.path.basename(folder)
    else:
        files = [f.strip('"') for f in args.files]
        if len(files) < 2:
            print("Error: Need at least 2 .md files to merge")
            sys.exit(1)
        # Check all are .md
        for f in files:
            if not f.endswith('.md'):
                print(f"Error: {f} is not a .md file")
                sys.exit(1)
        containing_folder = os.path.dirname(files[0])
        base_dir = os.path.dirname(containing_folder)
        folder_name = os.path.basename(containing_folder)

    sorted_files = sorted(files)
    output = os.path.join(base_dir, f"{folder_name}_merged.md")

    n = 1
    while os.path.exists(output):
        output = os.path.join(base_dir, f"{folder_name}_merged{{{n}}}.md")
        n += 1

    merge_md(sorted_files, output)
    print(f"Merged {len(sorted_files)} files into {output}")