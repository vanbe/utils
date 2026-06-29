#!/usr/bin/env python3
"""
heic2jpeg.py — Convert every HEIC/HEIF image in a folder to JPEG.

Decoding via pillow-heif (libheif bundled in the wheel — no apt/libheif system
install needed). Metadata is preserved with as little loss as possible:
  * EXIF block (date, GPS, camera Make/Model, …) and the ICC colour profile are
    carried over to the JPEG.
  * Orientation is normalised once with ImageOps.exif_transpose (pixels rotated
    upright, tag reset to 1) so EXIF-unaware viewers still display correctly and
    no double-rotation can happen.
  * JPEG saved with quality 95 and 4:4:4 chroma (subsampling=0) by default —
    the realistic "minimal loss" point for JPEG (JPEG is inherently lossy and
    8-bit; HDR/10-bit HEIC is tone-mapped to 8-bit).

Folder-action contract (utils_run.py): argv[0] is the directory; progress goes
to stdout (utils_run routes it to stderr); exit 0 on success, non-zero on error.
"""
import os
import sys
import argparse
import concurrent.futures
import multiprocessing
from pathlib import Path

import pillow_heif
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()

HEIC_EXTENSIONS = {'.heic', '.heif'}


def convert_single(task):
    """Convert one HEIC/HEIF file. Returns (status, name[, detail])."""
    src, out, quality, overwrite, delete_src = task
    try:
        if out.exists() and not overwrite:
            # Already converted earlier — honour deletion request, then skip.
            if delete_src and src.exists():
                os.remove(str(src))
            return ('skipped', src.name)

        img = Image.open(str(src))
        # Bake orientation upright and drop the orientation tag (no double-rotate).
        img = ImageOps.exif_transpose(img)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        save_kw = dict(quality=quality, subsampling=0, optimize=True)
        exif = img.info.get('exif')
        icc = img.info.get('icc_profile')
        if exif:
            save_kw['exif'] = exif
        if icc:
            save_kw['icc_profile'] = icc

        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out), 'JPEG', **save_kw)

        # Delete source only after the JPEG is written and re-opens cleanly.
        if delete_src:
            try:
                with Image.open(str(out)) as chk:
                    chk.load()
            except Exception as e:
                return ('error', src.name, f'output verify failed, source kept: {e}')
            if src.exists():
                os.remove(str(src))

        return ('success', src.name)
    except Exception as e:
        return ('error', src.name, str(e))


def convert_folder(root_folder, quality=95, recursive=True, overwrite=False,
                   num_cores=None, delete_heics=False):
    root = Path(root_folder)
    if not root.exists():
        print(f"Error: Folder '{root_folder}' does not exist")
        return 1

    pattern = '**/*' if recursive else '*'
    files = [p for p in root.glob(pattern)
             if p.is_file() and p.suffix.lower() in HEIC_EXTENSIONS]

    if not files:
        print(f"No HEIC/HEIF files found in '{root_folder}'")
        return 0

    print(f"Found {len(files)} HEIC/HEIF files")
    if num_cores is None:
        num_cores = max(1, multiprocessing.cpu_count() - 1)
    else:
        num_cores = max(1, min(num_cores, multiprocessing.cpu_count()))
    print(f"Using {num_cores} cores, quality={quality}, 4:4:4 chroma"
          + (", deleting sources after success" if delete_heics else ""))
    sys.stdout.flush()

    tasks = [(p, p.with_suffix('.jpg'), quality, overwrite, delete_heics) for p in files]

    converted = skipped = failed = 0
    total = len(tasks)
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as ex:
        futures = {ex.submit(convert_single, t): t for t in tasks}
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            tag = f" [{done}/{total}]"
            try:
                status, name, *rest = fut.result()
                if status == 'success':
                    converted += 1
                    print(f'✓ Converted {name}{tag}')
                elif status == 'skipped':
                    skipped += 1
                    print(f'⏭️  Skipped (jpg exists) {name}{tag}')
                elif status == 'error':
                    failed += 1
                    print(f'✗ Error {name}: {rest[0] if rest else "unknown"}{tag}')
            except Exception as e:
                failed += 1
                print(f'✗ Critical error fetching result: {e}')
            sys.stdout.flush()

    print(f"Conversion complete: {converted} converted, {skipped} skipped, "
          f"{failed} failed, {total} total")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(
        description='Convert HEIC/HEIF images in a folder to JPEG (EXIF + ICC preserved).')
    parser.add_argument('input_folder', help='Folder containing HEIC/HEIF images')
    parser.add_argument('-q', '--quality', type=int, default=95,
                        help='JPEG quality 1-100 (default 95 — minimal loss)')
    parser.add_argument('--no-recursive', dest='recursive', action='store_false',
                        default=True, help='Top folder only')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Re-convert even if the .jpg already exists')
    parser.add_argument('--delete-heics', dest='delete_heics', action='store_true',
                        default=False,
                        help='Delete each source HEIC after a verified successful conversion')
    parser.add_argument('--num-cores', type=int, default=None, help='CPU cores to use')
    args = parser.parse_args()

    rc = convert_folder(
        root_folder=args.input_folder,
        quality=args.quality,
        recursive=args.recursive,
        overwrite=args.overwrite,
        num_cores=args.num_cores,
        delete_heics=args.delete_heics,
    )
    sys.exit(rc)


if __name__ == '__main__':
    main()
