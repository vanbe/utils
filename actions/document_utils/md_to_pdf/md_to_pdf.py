#!/usr/bin/env python3
import sys
import os
import re
import subprocess
import tempfile
import shutil
import pypandoc
import argparse

def generate_pdf(content, output_pdf, md_file_dir, temp_dir, title, template):
    latex_header = r'''
        \usepackage{tcolorbox}
        \tcbuselibrary{skins}

        \newtcolorbox{warning}{
            enhanced,
            colback=yellow!10!white,
            colframe=red!75!black,
            fonttitle=\bfseries,
            title={⚠\ Attention},
            attach boxed title to top left={yshift=-2mm, xshift=2mm},
            boxed title style={colback=red!75!black},
            arc=4mm,
            boxshadow=0.5mm
        }
        '''
    if template == 'simple':
        latex_header += r'''
        \usepackage{titlesec}
        \titleformat{\section}{\centering\Large\bfseries}{}{0em}{}
        \titleformat{\subsection}{\large\bfseries}{}{0em}{}
        \titleformat{\subsubsection}{\normalsize\bfseries}{}{0em}{}
        '''
    
    header_file = os.path.join(temp_dir, 'header.tex')
    with open(header_file, 'w', encoding='utf-8') as f:
        f.write(latex_header)

    # Pandoc arguments
    extra_args=[
        '--pdf-engine=xelatex',
        f'--resource-path={md_file_dir}',
        '-V', 'geometry:margin=1in',
        '-V', 'documentclass=article',
        '-V', 'fontsize=11pt',
        '-V', 'mainfont=TeX Gyre Heros',
        '-V', 'sansfont=TeX Gyre Heros',
        '-V', 'monofont=Latin Modern Mono',
        '-H', header_file
    ]

    if template == 'report':
        extra_args.extend([
            '--toc',
            '-V', 'colorlinks=true',
            '-V', 'linkcolor=teal',
            '-V', 'urlcolor=teal'
        ])
        if title:
            extra_args.extend(['-V', f'title:{title}'])
        extra_args.extend(['-V', 'toc-title:Table des Matières'])
    else:  # simple
        # Simple styling, no toc, centered headings
        pass

    pandoc_format = 'markdown+raw_tex+hard_line_breaks'
    pypandoc.convert_text(content, 'pdf', format=pandoc_format, outputfile=output_pdf,
        extra_args=extra_args)

def main():
    parser = argparse.ArgumentParser(description="Convert Markdown to PDF")
    parser.add_argument('md_file', help="Path to the markdown file")
    parser.add_argument('--template', choices=['report', 'simple'], default='simple', help="PDF template to use")
    args = parser.parse_args()

    md_file = args.md_file
    template = args.template
    if not os.path.isfile(md_file):
        print(f"File {md_file} not found")
        sys.exit(1)
        
    md_file_path = os.path.abspath(md_file)
    md_file_dir = os.path.dirname(md_file_path)

    # Read the file content
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Preprocess content for better PDF formatting
    # For transcription files, ensure each line is a separate paragraph
    lines = content.split('\n')
    processed_lines = []
    for line in lines:
        processed_lines.append(line)
        # If line starts with timestamp (transcription format), add blank line for paragraph break
        if line.strip().startswith('[') and ':' in line and ']' in line:
            processed_lines.append('')
    content = '\n'.join(processed_lines)

    # Create a temporary directory for generated assets
    temp_dir = tempfile.mkdtemp()
    
    # Handle Mermaid blocks
    mermaid_pattern = r'```mermaid\s*\n(.*?)\n```'
    matches = re.findall(mermaid_pattern, content, re.DOTALL)
    for i, match in enumerate(matches):
        mermaid_file = os.path.join(temp_dir, f'mermaid_{i}.mmd')
        with open(mermaid_file, 'w', encoding='utf-8') as f:
            f.write(match)

        png_file = os.path.join(temp_dir, f'mermaid_{i}.png')
        try:
            subprocess.run(['mmdc', '-i', mermaid_file, '-o', png_file], check=True)
            old_block = f'```mermaid\n{match}\n```'
            new_block = f'![Mermaid Diagram]({png_file})'
            content = content.replace(old_block, new_block, 1)
        except subprocess.CalledProcessError:
            print(f"Failed to generate image for mermaid block {i}")
            continue

    # Handle warn blocks by converting them to raw LaTeX
    warn_pattern = r'(::: \s*warn(.*?) \s* :::)'
    warn_matches = list(re.finditer(warn_pattern, content, re.DOTALL))
    for match in warn_matches:
        old_block = match.group(0)
        inner_content = match.group(2).strip()
        new_block = f'\\begin{{warning}}\n{inner_content}\n\\end{{warning}}'
        content = content.replace(old_block, new_block, 1)
    
    # Extract the main title (H1) for report template
    title = ""
    if template == 'report':
        h1_pattern = r'^#\s+(.*)'
        h1_match = re.search(h1_pattern, content, re.MULTILINE)
        if h1_match:
            title = h1_match.group(1)
            # Remove the H1 from the content to prevent duplication
            content = content[:h1_match.start()] + content[h1_match.end():]

    # Insert a new page before each H2 for report
    if template == 'report':
        content = re.sub(r'(^##\s+.*)', r'\n\\newpage\n\1', content, flags=re.MULTILINE)

    # --- LOGIC FOR PDF FILENAME ---
    md_file_basename = os.path.basename(md_file_path)
    if md_file_basename.lower() == 'readme.md':
        # If the file is Readme.md, use the parent folder's name for the PDF
        parent_folder_name = os.path.basename(md_file_dir)
        output_pdf = os.path.join(md_file_dir, f'{parent_folder_name}.pdf')
    else:
        # Otherwise, use the original markdown file's name
        output_pdf = md_file.replace('.md', '.pdf')

    try:
        generate_pdf(content, output_pdf, md_file_dir, temp_dir, title, template)
        print(f"PDF created: {output_pdf}")

    except Exception as e:
        print(f"Error during PDF conversion: {e}")
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()