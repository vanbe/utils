#!/usr/bin/env python3
import os
import sys
import argparse
import rawpy
import imageio
import shutil
import subprocess
import platform
from pathlib import Path
from PIL import Image
import concurrent.futures
import multiprocessing
import threading

# --- CONFIGURATION ---

def load_config():
    """Load configuration from .env file."""
    config = {}
    env_paths = [Path(__file__).parent.parent.parent.parent / '.env']
    for env_path in env_paths:
        if env_path.exists() and env_path.is_file():
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            config[key.strip()] = value.strip()
            except Exception:
                pass
    return config

def get_num_cores():
    """Get the number of cores to use for parallel processing."""
    config = load_config()
    num_cores = config.get('RAW_TO_JPG_NUM_CORES', '')
    if num_cores:
        try:
            num_cores = int(num_cores)
            if num_cores <= 0: num_cores = multiprocessing.cpu_count()
        except ValueError:
            num_cores = multiprocessing.cpu_count()
    else:
        num_cores = multiprocessing.cpu_count()
    return max(1, min(num_cores, multiprocessing.cpu_count()))

# --- METADATA UTILS ---

def run_exiftool(source_path, target_path):
    """
    Use external exiftool to copy all metadata tags from source to target.
    """
    try:
        cmd = [
            'exiftool',
            '-overwrite_original',
            '-TagsFromFile', str(source_path),
            '-all:all>all:all', 
            '-FileModifyDate', 
            str(target_path)
        ]
        # Suppress output unless error
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        # Exiftool not installed
        return False

def extract_metadata(img_path):
    """Extract metadata for comparison/fallback."""
    metadata = {}
    try:
        stat = os.stat(str(img_path))
        metadata['mtime'] = stat.st_mtime
        metadata['atime'] = stat.st_atime
        
        # Try extracting python-accessible metadata
        raw_extensions = {'.arw', '.cr2', '.dng', '.nef', '.raw', '.rw2'}
        if img_path.suffix.lower() in raw_extensions:
            try:
                with rawpy.imread(str(img_path)) as raw:
                    metadata['raw_metadata'] = {'sizes': raw.sizes}
            except: pass
        else:
            try:
                with Image.open(str(img_path)) as img:
                    exif_bytes = img.info.get('exif')
                    if exif_bytes: metadata['exif'] = exif_bytes
            except: pass
    except Exception as e:
        print(f'Warning: Could not extract metadata from {img_path}: {e}')
    return metadata

def copy_metadata_to_file(source_metadata, target_path, source_path=None):
    """
    Copy metadata using exiftool (best) or standard python libs (fallback).
    """
    # 1. Try Exiftool (Copies MakerNotes, Lens info, Correct Dates)
    if source_path:
        if run_exiftool(source_path, target_path):
            return 

    # 2. Fallback: OS Timestamps
    try:
        if 'mtime' in source_metadata and 'atime' in source_metadata:
            os.utime(str(target_path), (source_metadata['atime'], source_metadata['mtime']))
    except Exception as e:
        print(f'Warning: Could not copy metadata to {target_path}: {e}')

def metadata_matches(original_metadata, target_path):
    """
    Check if the target file's modification time roughly matches the source.
    Used to skip updating if not necessary.
    """
    try:
        stat = os.stat(str(target_path))
        # Allow 1 second precision difference
        if abs(stat.st_mtime - original_metadata.get('mtime', 0)) > 1: 
            return False
        return True
    except Exception:
        return False

# --- PROCESSING ---

def convert_raw_to_jpeg_single(task):
    """
    Process a single RAW file conversion task.
    """
    # REMOVED: counter, total, lock from arguments
    raw_path, output_path, quality, move_raws_to, root_path, overwrite, delete_raws = task

    try:
        original_metadata = extract_metadata(raw_path)

        # CHECK EXISTING
        if output_path.exists() and not overwrite:
            status = 'skipped'
            if not metadata_matches(original_metadata, output_path):
                copy_metadata_to_file(original_metadata, output_path, source_path=raw_path)
                status = 'updated_metadata'
            # JPEG déjà présent => conversion considérée faite : purge du RAW si demandé.
            if delete_raws and raw_path.exists():
                os.remove(str(raw_path))
            return (status, raw_path.name)

        # CONVERSION
        with rawpy.imread(str(raw_path)) as raw:
            rgb = raw.postprocess()

        imageio.imsave(str(output_path), rgb, quality=quality, optimize=True)

        # Apply Metadata
        copy_metadata_to_file(original_metadata, output_path, source_path=raw_path)

        # Sur succès : supprimer (prioritaire) ou déplacer le RAW source.
        if delete_raws:
            if raw_path.exists():
                os.remove(str(raw_path))
        elif move_raws_to:
            move_raw_file(raw_path, root_path, move_raws_to)

        return ('success', raw_path.name)

    except Exception as e:
        return ('error', raw_path.name, str(e))

def move_raw_file(raw_path, root_path, move_raws_to):
    try:
        relative_path = raw_path.relative_to(root_path)
        dest_path = Path(move_raws_to) / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(raw_path), str(dest_path))
    except Exception as e:
        print(f'✗ Error moving {raw_path.name}: {e}')

def convert_raw_to_jpeg(root_folder, output_folder=None, quality=90, recursive=True, overwrite=False, move_raws_to=None, num_cores=None, delete_raws=False):
    root_path = Path(root_folder)
    
    if not root_path.exists():
        print(f"Error: Folder '{root_folder}' does not exist")
        return

    raw_extensions = {'.arw', '.cr2', '.dng', '.nef', '.raw', '.rw2'}
    search_pattern = "**/*" if recursive else "*"
    
    raw_files = []
    for file_path in root_path.glob(search_pattern):
        if file_path.is_file() and file_path.suffix.lower() in raw_extensions:
            raw_files.append(file_path)
    
    if not raw_files:
        print(f"No RAW files found in '{root_folder}'")
        return
    
    print(f"Found {len(raw_files)} RAW files")

    try:
        subprocess.run(['exiftool', '-ver'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Exiftool detected: Full metadata copy enabled.")
    except FileNotFoundError:
        print("Warning: Exiftool not found. Timestamps will be synced via OS, but internal EXIF may be incomplete.")

    if num_cores is None:
        num_cores = get_num_cores()
    else:
        num_cores = max(1, min(num_cores, multiprocessing.cpu_count()))
    
    print(f"Using {num_cores} cores for processing")
    sys.stdout.flush()

    tasks = []
    for raw_path in raw_files:
        if output_folder:
            relative_path = raw_path.relative_to(root_path)
            output_path = Path(output_folder) / relative_path.with_suffix('.jpg')
        else:
            output_path = raw_path.with_suffix('.jpg')

        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # REMOVED: counter, total, lock from the tuple
        tasks.append((raw_path, output_path, quality, move_raws_to, root_path, overwrite, delete_raws))

    successful = 0
    skipped = 0
    total_files = len(raw_files)
    
    # Track completed tasks manually in the main process
    completed_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_task = {executor.submit(convert_raw_to_jpeg_single, task): task for task in tasks}

        for future in concurrent.futures.as_completed(future_to_task):
            completed_count += 1
            try:
                # Get the return value from the worker
                status_code, filename, *args = future.result()
                
                progress_str = f" [{completed_count}/{total_files}]"

                if status_code == 'success':
                    successful += 1
                    print(f'✓ Converted {filename}{progress_str}')
                elif status_code == 'updated_metadata':
                    skipped += 1
                    print(f'🔄 Updated metadata {filename}{progress_str}')
                elif status_code == 'skipped':
                    skipped += 1
                    # print(f'⏭️  Skipped {filename}') # Optional
                elif status_code == 'error':
                    error_msg = args[0] if args else "Unknown error"
                    print(f'✗ Error processing {filename}: {error_msg}')
                
                sys.stdout.flush()

            except Exception as e:
                print(f'✗ Critical Error fetching result: {e}')

    print(f"Conversion complete: {successful} converted, {skipped} skipped/updated, {total_files} total files")

def main():
    parser = argparse.ArgumentParser(
        description='Convert RAW images to JPEG preserving metadata.',
        epilog="If output files exist, their metadata will be updated to match the RAW file unless --overwrite is used."
    )
    
    parser.add_argument('input_folder', help='Path to folder containing RAW images')
    parser.add_argument('-o', '--output', dest='output_folder', help='Custom output folder')
    parser.add_argument('-q', '--quality', type=int, default=90, help='JPEG quality (1-100)')
    parser.add_argument('--no-recursive', dest='recursive', action='store_false', default=True, help='Top folder only')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Force re-conversion of existing files')
    parser.add_argument('--move-raws-to', dest='move_raws_to', help='Move RAW files after conversion')
    parser.add_argument('--delete-raws', dest='delete_raws', action='store_true', default=False,
                        help='Delete each source RAW file after a successful conversion (prioritaire sur --move-raws-to)')
    parser.add_argument('--num-cores', type=int, default=None, help='Number of CPU cores')

    args = parser.parse_args()

    convert_raw_to_jpeg(
        root_folder=args.input_folder,
        output_folder=args.output_folder,
        quality=args.quality,
        recursive=args.recursive,
        overwrite=args.overwrite,
        move_raws_to=args.move_raws_to,
        num_cores=args.num_cores,
        delete_raws=args.delete_raws
    )

if __name__ == "__main__":
    main()