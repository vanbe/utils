import PyPDF2
from PyPDF2.generic import Destination
import os
import argparse
import re
import sys
import csv
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

def get_page_number(reader, page_object):
    """
    Safely extract page number from a page object, handling NullObject issues.
    """
    try:
        # Try the standard method first
        if hasattr(page_object, 'indirect_reference'):
            return reader.get_page_number(page_object)
        
        # Alternative method for problematic PDFs
        for i, page in enumerate(reader.pages):
            if hasattr(page_object, 'get_object') and page == page_object.get_object():
                return i
            elif page == page_object:
                return i
                
        return None
    except Exception as e:
        print(f"Warning: Could not extract page number: {e}")
        return None

def get_pdf_outline_info(outline, reader, level=0, max_level=1) -> List[Dict[str, Any]]:
    """
    Recursively extract bookmark titles and page numbers from PDF outline.
    Only extracts up to the specified max_level to get first-level chapters.
    """
    outline_info = []
    for item in outline:
        try:
            if isinstance(item, Destination):
                title = item.title
                page_index = get_page_number(reader, item.page)
                
                if page_index is not None:
                    outline_info.append({
                        "title": title, 
                        "page_index": page_index,
                        "level": level
                    })
                else:
                    print(f"Warning: Could not get page number for bookmark '{title}'")
                    
            elif isinstance(item, list) and level < max_level:
                # Recursively process nested bookmarks but only up to max_level
                outline_info.extend(get_pdf_outline_info(item, reader, level + 1, max_level))
        except Exception as e:
            print(f"Warning: Error processing outline item: {e}")
            continue
            
    return outline_info

def filter_top_level_chapters(outline_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter to keep only top-level chapters, excluding all subsections.
    """
    # Patterns to identify chapter headings
    chapter_patterns = [
        r'^Chapter\s+\d+',  # Chapter 1, Chapter 2, etc.
        r'^Appendix\s+[A-Z]',  # Appendix A, Appendix B, etc.
    ]
    
    # Also look for patterns that indicate solutions or other content to exclude
    exclude_patterns = [
        r'^\d+\.\d+',  # Subsection patterns like 1.1, 1.2, etc.
        r'^Glossary',
        r'^References',
        r'^Index',
        r'^Solutions',
        r'^Review Questions',
        r'^Critical Thinking Questions',
        r'^Self-Check Questions',
        r'^Key Concepts',
    ]
    
    top_level_chapters = []
    seen_chapters = set()
    
    for item in outline_info:
        title = item["title"]
        
        # Check if this should be excluded (subsections or other non-chapter content)
        if any(re.search(pattern, title, re.IGNORECASE) for pattern in exclude_patterns):
            continue
            
        # Check if this is a top-level chapter
        is_chapter = any(re.search(pattern, title) for pattern in chapter_patterns)
        
        # For chapters that don't match the pattern but are at level 0
        is_top_level = item.get("level", 0) == 0
        
        # Extract chapter number for deduplication
        chapter_num_match = re.search(r'(Chapter|Appendix)\s+([\dA-Z]+)', title)
        chapter_key = None
        if chapter_num_match:
            chapter_key = f"{chapter_num_match.group(1)} {chapter_num_match.group(2)}"
        
        # Add to results if it's a chapter or top-level item, and not a duplicate
        if (is_chapter or is_top_level) and (chapter_key is None or chapter_key not in seen_chapters):
            if chapter_key:
                seen_chapters.add(chapter_key)
            top_level_chapters.append(item)
    
    return top_level_chapters

def calculate_page_ranges(outline_info: List[Dict[str, Any]], total_pages: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Calculate page ranges for each chapter based on outline info and total pages.
    Returns 1-based page numbers and also identifies gaps as unidentified sections.
    """
    sections = []
    unidentified_sections = []
    
    # Sort chapters by page number to ensure correct order
    outline_info.sort(key=lambda x: x["page_index"])
    
    print("\nDetected top-level chapters with page ranges:")
    print("-" * 60)
    
    # First, identify all covered page ranges
    covered_ranges = []
    for i, item in enumerate(outline_info):
        title = item["title"]
        start_index = item["page_index"]

        # Determine end page - this chapter ends where the next chapter begins
        if i + 1 < len(outline_info):
            end_index = outline_info[i+1]["page_index"] - 1
        else:
            end_index = total_pages - 1  # Last chapter goes to end of document

        # Ensure valid page range
        if start_index > end_index:
            print(f"Warning: Invalid page range for '{title}' ({start_index+1}-{end_index+1})")
            continue
            
        # Convert to 1-based for output
        start_page = start_index + 1
        end_page = end_index + 1
        
        sections.append({
            "name": title,
            "start_page": start_page,
            "end_page": end_page,
            "total_pages": end_page - start_page + 1
        })
        
        covered_ranges.append((start_index, end_index))
        
        print(f"{i+1:2d}. {title}")
        print(f"     Pages: {start_page} - {end_page}")

    # Now identify gaps (unidentified pages)
    covered_ranges.sort(key=lambda x: x[0])
    current_page = 0
    
    for start, end in covered_ranges:
        if current_page < start:
            # There's a gap before this section
            gap_start = current_page
            gap_end = start - 1
            total_gap_pages = gap_end - gap_start + 1
            unidentified_sections.append({
                "name": f"Unidentified Pages {gap_start+1}-{gap_end+1}",
                "start_page": gap_start + 1,
                "end_page": gap_end + 1,
                "total_pages": total_gap_pages
            })
        current_page = end + 1
    
    # Check if there's a gap at the end
    if current_page < total_pages:
        total_gap_pages = total_pages - current_page
        unidentified_sections.append({
            "name": f"Unidentified Pages {current_page+1}-{total_pages}",
            "start_page": current_page + 1,
            "end_page": total_pages,
            "total_pages": total_gap_pages
        })

    return sections, unidentified_sections

def generate_chapter_summary(sections: List[Dict[str, Any]], output_dir: str, pdf_basename: str) -> None:
    """
    Generate a CSV file with chapter summary information.
    """
    csv_filename = f"{pdf_basename}_chapter_summary.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    
    try:
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Chapter Number', 'Chapter Name', 'Start Page', 'End Page', 'Total Pages']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for i, section in enumerate(sections):
                # Extract chapter number from name if possible
                chapter_match = re.search(r'(Chapter|Appendix)\s+([\dA-Z]+)', section['name'])
                chapter_number = chapter_match.group(2) if chapter_match else f"Unidentified {i+1}"
                
                writer.writerow({
                    'Chapter Number': chapter_number,
                    'Chapter Name': section['name'],
                    'Start Page': section['start_page'],
                    'End Page': section['end_page'],
                    'Total Pages': section['total_pages']
                })
        
        print(f"Chapter summary saved to: {csv_filename}")
        
    except Exception as e:
        print(f"Error generating chapter summary: {e}")

def perform_pdf_split(reader, sections: List[Dict[str, Any]], output_dir: str, add_sequence: bool = True) -> None:
    """
    Split PDF into multiple files based on calculated page ranges.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving split PDF files to: {output_dir}")

    total_sections = len(sections)
    num_digits = len(str(total_sections))

    for i, section in enumerate(sections):
        writer = PyPDF2.PdfWriter()
        start_page_index = section['start_page'] - 1  # Convert to 0-based
        end_page_index = section['end_page'] - 1      # Convert to 0-based

        # Validate page range
        if (start_page_index < 0 or 
            end_page_index >= len(reader.pages) or 
            start_page_index > end_page_index):
            print(f"Warning: Skipping invalid page range for section '{section['name']}' ({section['start_page']}-{section['end_page']})")
            continue

        # Add pages to writer
        for page_num in range(start_page_index, end_page_index + 1):
            try:
                writer.add_page(reader.pages[page_num])
            except Exception as e:
                print(f"Error adding page {page_num+1} for section '{section['name']}': {e}")
                continue

        # Clean filename by removing invalid characters
        cleaned_name = re.sub(r'[\\/*?:"<>|]', '_', section['name'])
        
        # Add sequence number if requested
        if add_sequence:
            sequence_prefix = f"{i+1:0{num_digits}d}_"
            output_filename = f"{sequence_prefix}{cleaned_name}.pdf"
        else:
            output_filename = f"{cleaned_name}.pdf"
            
        output_filepath = os.path.join(output_dir, output_filename)

        # Write PDF file
        try:
            with open(output_filepath, 'wb') as output_pdf:
                writer.write(output_pdf)
            print(f"Created: {output_filename} (pages {section['start_page']}-{section['end_page']})")
        except Exception as e:
            print(f"Error writing file {output_filename}: {e}")

def split_pdf_by_chapters(pdf_path: str, output_dir: Optional[str] = None, add_sequence: bool = True) -> None:
    """
    Automatically split PDF file based on bookmarks/chapters.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: Input file '{pdf_path}' does not exist.")
        return

    # Get base name of PDF file (without extension)
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]

    # Determine output directory
    if output_dir is None:
        pdf_directory = os.path.dirname(pdf_path)
        final_output_dir = os.path.join(pdf_directory, pdf_basename + "_chapters")
    else:
        final_output_dir = os.path.join(output_dir, pdf_basename + "_chapters")

    try:
        print(f"Processing PDF: {pdf_path}")
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            print(f"PDF total pages: {total_pages}")

            # Check if PDF has outline/bookmarks
            if not reader.outline:
                print("No outline/bookmarks found in PDF. Cannot split by chapters.")
                return

            print("Extracting outline information...")
            # Extract outline information (only level 0 for top-level chapters)
            outline_info = get_pdf_outline_info(reader.outline, reader, max_level=0)
            
            # Filter out bookmarks with None page indices
            outline_info = [item for item in outline_info if item.get("page_index") is not None]

            if not outline_info:
                print("No valid bookmark information found. Cannot split by chapters.")
                return
                
            print(f"Found {len(outline_info)} bookmarks in outline")

            # Filter to keep only top-level chapters
            top_level_chapters = filter_top_level_chapters(outline_info)
            
            if not top_level_chapters:
                print("No top-level chapters found. Trying to use all bookmarks...")
                top_level_chapters = outline_info
                
            if not top_level_chapters:
                print("No valid chapters identified for splitting.")
                return
                
            print(f"Found {len(top_level_chapters)} top-level chapters")

            # Calculate page ranges for each section and identify gaps
            sections, unidentified_sections = calculate_page_ranges(top_level_chapters, total_pages)
            
            if not sections and not unidentified_sections:
                print("No valid sections identified for splitting.")
                return

            # Add unidentified sections to the main sections list
            if unidentified_sections:
                print(f"\nFound {len(unidentified_sections)} sections of unidentified pages:")
                for i, section in enumerate(unidentified_sections):
                    print(f"  {i+1}. {section['name']} (pages {section['start_page']}-{section['end_page']})")
                sections.extend(unidentified_sections)

            # Ask for confirmation before splitting (unless --yes flag is set)
            if not args.yes:
                print("\n" + "-" * 60)
                response = input("Proceed with splitting? (y/n): ")
                if response.lower() != 'y':
                    print("Operation cancelled.")
                    return

            # Perform the actual splitting
            perform_pdf_split(reader, sections, final_output_dir, add_sequence)
            
            # Generate chapter summary CSV
            generate_chapter_summary(sections, final_output_dir, pdf_basename)
            
            print("\nPDF splitting completed successfully!")

    except Exception as e:
        print(f"Error processing PDF file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatically split PDF file based on bookmarks/chapters.")
    parser.add_argument("input_pdf", help="Path to the PDF file to split.")
    parser.add_argument("-o", "--output_dir", default=None,
                        help="Output directory for split PDF files (default: subfolder in same directory as input PDF).")
    parser.add_argument("--no-sequence", action="store_true",
                        help="Do not add sequence number prefixes to output files.")
    parser.add_argument("--include-all", action="store_true",
                        help="Include all bookmarks, not just top-level chapters.")
    parser.add_argument("--yes", action="store_true",
                        help="Automatically proceed without confirmation.")
    
    args = parser.parse_args()
    
    # If include-all flag is set, modify the behavior
    if args.include_all:
        # For include-all, we'll extract more levels
        # This would require modifying the get_pdf_outline_info call
        print("Warning: --include-all flag is not fully implemented in this version.")
    
    split_pdf_by_chapters(args.input_pdf, args.output_dir, add_sequence=not args.no_sequence)
