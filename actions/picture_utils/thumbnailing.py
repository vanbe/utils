#!/usr/bin/env python3
"""
thumbnailing.py — Batch thumbnail generator for images and videos.

Features
--------
- JPEG/PNG/RAW → JPEG thumbnails with EXIF preservation
- Video → compressed MP4 with metadata copy
- ExifTool daemon (one persistent process instead of per-file subprocess launches)
- GPU (NVENC) acceleration with automatic CPU fallback
- Parallel processing via ThreadPoolExecutor

Usage
-----
  python thumbnailing.py <input_folder> [options]

  -o / --output   PATH    Custom output folder (default: thumbnails/ beside each file)
  --size          W H     Max image thumbnail dimensions (default: 1920 1080)
  --video-size    W H     Max video thumbnail dimensions (default: 640 480)
  -q / --quality  N       JPEG quality 1-100 (default: 85)
  --no-recursive          Top-level folder only
  --overwrite             Regenerate existing thumbnails
  --num-cores     N       Worker threads (default: auto from .env / cpu_count)
  --mirror                Delete output thumbnails whose source is gone (needs -o)
  --dry-run               Preview orphan deletions only; no generation, no deletion
  --allow-empty           Permit mirror pruning even when the source is empty
"""

import os
import sys
import argparse
from pathlib import Path
import concurrent.futures
import multiprocessing
import threading
import subprocess
import platform
from datetime import datetime

from PIL import Image
import rawpy
import imageio


# ---------------------------------------------------------------------------
# Config (read once, cached in module)
# ---------------------------------------------------------------------------

_cfg: dict | None = None


def load_config() -> dict:
    global _cfg
    if _cfg is not None:
        return _cfg
    _cfg = {}
    env = Path(__file__).parent.parent.parent / '.env'
    if env.is_file():
        try:
            with open(env) as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith('#') and '=' in s:
                        k, v = s.split('=', 1)
                        _cfg[k.strip()] = v.strip()
        except Exception:
            pass
    return _cfg


def _int_cfg(key: str, default: int) -> int:
    try:
        return int(load_config().get(key, default))
    except (ValueError, TypeError):
        return default


def get_num_cores() -> int:
    n = _int_cfg('THUMBNAIL_NUM_CORES', 0)
    if n <= 0:
        n = multiprocessing.cpu_count()
    return max(1, min(n, multiprocessing.cpu_count()))


# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------

VIDEO_EXTS = frozenset({'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'})
RAW_EXTS   = frozenset({'.arw', '.cr2', '.cr3', '.dng', '.nef', '.orf', '.pef', '.rw2', '.raw'})
IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'})
MEDIA_EXTS = IMAGE_EXTS | RAW_EXTS | VIDEO_EXTS


# ---------------------------------------------------------------------------
# ExifTool daemon
# ---------------------------------------------------------------------------

class ExifToolDaemon:
    """
    Persistent exiftool process — eliminates per-file Perl startup overhead.

    Without daemon : N files × 2-3 exiftool launches × ~200 ms startup = minutes of overhead.
    With daemon    : one 200 ms startup, then each command costs ~5-10 ms.

    Thread-safe: all stdin/stdout I/O is serialised through a lock.
    Falls back gracefully if exiftool is not installed.
    """

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        try:
            self._proc = subprocess.Popen(
                ['exiftool', '-stay_open', 'True', '-@', '-'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except FileNotFoundError:
            pass  # exiftool not installed — fallback to subprocess per call

    @property
    def ok(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def run(self, args: list) -> str:
        """Execute one exiftool command set; return stdout text."""
        if not self.ok:
            return ''
        with self._lock:
            payload = '\n'.join(str(a) for a in args) + '\n-execute\n'
            self._proc.stdin.write(payload.encode())
            self._proc.stdin.flush()
            lines = []
            while True:
                raw = self._proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode('utf-8', errors='replace').rstrip('\r\n')
                if line == '{ready}':
                    break
                lines.append(line)
            return '\n'.join(lines)

    def close(self):
        if self.ok:
            try:
                self._proc.stdin.write(b'-stay_open\nFalse\n')
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _et_run(etd: ExifToolDaemon | None, args: list) -> str:
    """Run exiftool via daemon if available, otherwise as a one-shot subprocess."""
    if etd and etd.ok:
        return etd.run(args)
    try:
        r = subprocess.run(
            ['exiftool'] + [str(a) for a in args],
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout
    except Exception:
        return ''


# ---------------------------------------------------------------------------
# GPU detection (done once per run, not per video)
# ---------------------------------------------------------------------------

def _probe_nvenc() -> bool:
    """Return True if FFmpeg has h264_nvenc support on this machine."""
    try:
        r = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=5,
        )
        return 'h264_nvenc' in r.stdout
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def extract_metadata(path: Path) -> dict:
    """Collect filesystem timestamps and (for images) the raw EXIF blob."""
    meta: dict = {}
    try:
        st = path.stat()
        meta['mtime'] = st.st_mtime
        meta['atime'] = st.st_atime
        if path.suffix.lower() not in RAW_EXTS:
            try:
                with Image.open(str(path)) as img:
                    exif = img.info.get('exif')
                    if exif:
                        meta['exif'] = exif
            except Exception:
                pass
    except Exception as e:
        print(f'Warning: cannot read metadata for {path.name}: {e}')
    return meta


def get_video_date(path: Path, etd: ExifToolDaemon | None = None) -> float | None:
    """Return the earliest internal date found in a video file, or None."""
    out = _et_run(etd, [
        '-fast',
        '-CreateDate', '-MediaCreateDate', '-DateTimeOriginal',
        '-s3', '-d', '%Y:%m:%d %H:%M:%S',
        str(path),
    ])
    dates = []
    for line in out.splitlines():
        try:
            clean = line.strip().split('+')[0].strip()
            if not clean or '0000' in clean:
                continue
            dt = datetime.strptime(clean, '%Y:%m:%d %H:%M:%S')
            dates.append(dt.timestamp())
        except ValueError:
            continue
    return min(dates) if dates else None


def copy_metadata_to_file(
    src_meta: dict,
    dst: Path,
    src: Path | None = None,
    etd: ExifToolDaemon | None = None,
):
    """Copy EXIF tags + filesystem timestamps from src to dst."""
    dst_s = str(dst)
    src_s = str(src) if src else None

    if src_s:
        # Pass 1 — copy all tags from source
        _et_run(etd, [
            '-overwrite_original', '-P',
            '-TagsFromFile', src_s,
            '-all:all>all:all',
            '-CreateDate<CreateDate',
            '-ModifyDate<ModifyDate',
            '-DateTimeOriginal<DateTimeOriginal',
            '-MediaCreateDate<MediaCreateDate',
            '-TrackCreateDate<TrackCreateDate',
            '-FileModifyDate<FileModifyDate',
            dst_s,
        ])

        # Pass 2 — if DateTimeOriginal is still absent (source had no EXIF date),
        # inject the filesystem mtime so media libraries (Nextcloud, etc.) see a date.
        if 'mtime' in src_meta:
            fs_date = datetime.fromtimestamp(src_meta['mtime']).strftime('%Y:%m:%d %H:%M:%S')
            _et_run(etd, [
                '-overwrite_original', '-P',
                '-if', 'not $DateTimeOriginal',
                f'-DateTimeOriginal={fs_date}',
                f'-CreateDate={fs_date}',
                f'-MediaCreateDate={fs_date}',
                f'-TrackCreateDate={fs_date}',
                dst_s,
            ])

    # Filesystem-level timestamps (always)
    if 'mtime' in src_meta and 'atime' in src_meta:
        try:
            os.utime(dst_s, (src_meta['atime'], src_meta['mtime']))
        except Exception:
            pass
    if src_s and platform.system() != 'Windows':
        try:
            subprocess.run(['touch', '-r', src_s, dst_s], check=False, timeout=5)
        except Exception:
            pass


def metadata_matches(
    src_meta: dict,
    thumb: Path,
    src: Path,
    etd: ExifToolDaemon | None = None,
) -> bool:
    """Return True if the thumbnail is valid and in sync with its source."""
    try:
        st = thumb.stat()

        # 1. Fast: filesystem mtime (2-second tolerance for FAT/SMB)
        if abs(st.st_mtime - src_meta.get('mtime', 0)) > 2:
            return False

        # 2. Image: EXIF presence check
        if thumb.suffix.lower() in {'.jpg', '.jpeg'}:
            try:
                with Image.open(str(thumb)) as img:
                    if not img.info.get('exif'):
                        return False
            except Exception:
                return False

        # 3. Video: internal date check (only when FS timestamps already match)
        if thumb.suffix.lower() in VIDEO_EXTS:
            src_date = get_video_date(src, etd)
            dst_date = get_video_date(thumb, etd)
            if src_date and (not dst_date or abs(src_date - dst_date) > 10):
                return False

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Thumbnail creation
# ---------------------------------------------------------------------------

def create_thumbnail(
    img_path: Path,
    out: Path,
    size: tuple,
    quality: int,
    src_meta: dict,
) -> bool:
    """Resize an image (or RAW) to a JPEG thumbnail."""
    temp: Path | None = None
    try:
        if img_path.suffix.lower() in RAW_EXTS:
            temp = out.with_suffix('.tmp.jpg')
            with rawpy.imread(str(img_path)) as raw:
                rgb = raw.postprocess()
            imageio.imsave(str(temp), rgb, quality=95, optimize=True)
            open_path = temp
        else:
            open_path = img_path

        with Image.open(str(open_path)) as img:
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            img.thumbnail(size, Image.LANCZOS)      # explicit high-quality filter
            kw: dict = {'quality': quality, 'optimize': True}
            if 'exif' in src_meta:
                kw['exif'] = src_meta['exif']
            img.save(str(out), 'JPEG', **kw)
        return True

    except Exception as e:
        print(f'✗ Image error {img_path.name}: {e}')
        return False
    finally:
        if temp and temp.exists():
            try:
                temp.unlink()
            except Exception:
                pass


def create_compressed_video(
    vid: Path,
    out: Path,
    gpu_sem: threading.Semaphore,
    gpu_available: bool,
    video_size: tuple = (640, 480),
) -> bool:
    """Encode a compressed video thumbnail via GPU (NVENC) or CPU (x264)."""
    vw, vh = video_size
    # Scale to fit inside video_size box, keep aspect ratio, round to even pixels.
    vf = (
        f'scale={vw}:{vh}:force_original_aspect_ratio=decrease,'
        'scale=trunc(iw/2)*2:trunc(ih/2)*2'
    )
    base_cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-i', str(vid),
        '-map_metadata', '0',
        '-movflags', '+faststart',   # web-optimised container layout
        '-vf', vf,
        '-y',
    ]
    gpu_enc = ['-c:v', 'h264_nvenc', '-preset', 'p2', '-cq', '32',
               '-c:a', 'aac', '-b:a', '64k']
    cpu_enc = ['-c:v', 'libx264', '-preset', 'fast', '-crf', '28',
               '-c:a', 'aac', '-b:a', '64k', '-threads', '1']

    # --- GPU attempt (only if hardware confirmed available) ---
    if gpu_available:
        with gpu_sem:
            try:
                r = subprocess.run(
                    base_cmd + gpu_enc + [str(out)],
                    capture_output=True, text=True, timeout=300,
                )
                if r.returncode == 0:
                    return True
            except Exception:
                pass  # fall through to CPU

    # --- CPU fallback ---
    try:
        r = subprocess.run(
            base_cmd + cpu_enc + [str(out)],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            print(f'✗ ffmpeg failed for {vid.name}: {r.stderr[-300:]}')
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f'✗ ffmpeg timed out for {vid.name}')
        return False
    except Exception as e:
        print(f'✗ ffmpeg error for {vid.name}: {e}')
        return False


# ---------------------------------------------------------------------------
# Per-file worker (runs inside ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _process_one(task: tuple) -> str:
    (img_path, out_path, size, video_size, quality, overwrite,
     counter, total, lock, gpu_sem, gpu_available, etd) = task

    def log(icon: str, msg: str):
        with lock:
            counter[0] += 1
            n = counter[0]
        pct = n * 100 // total
        print(f'{icon} [{n}/{total} {pct}%]  {msg}')
        sys.stdout.flush()

    try:
        src_meta = extract_metadata(img_path)
        is_video = img_path.suffix.lower() in VIDEO_EXTS

        if out_path.exists():
            if not overwrite:
                if metadata_matches(src_meta, out_path, img_path, etd):
                    # Thumbnail valid — count silently and skip
                    with lock:
                        counter[0] += 1
                    return 'skipped'
                # Metadata drift detected — try a lightweight fix first
                copy_metadata_to_file(src_meta, out_path, img_path, etd)
                if metadata_matches(src_meta, out_path, img_path, etd):
                    log('🔄', f'Fixed metadata: {out_path.name}')
                    return 'fixed'
                # Fix didn't work — fall through to full regeneration

        ok = (
            create_compressed_video(img_path, out_path, gpu_sem, gpu_available, video_size)
            if is_video
            else create_thumbnail(img_path, out_path, size, quality, src_meta)
        )

        if ok:
            copy_metadata_to_file(src_meta, out_path, img_path, etd)
            log('✓', f'{"Vid" if is_video else "Img"}: {out_path.name}')
            return 'success'
        return 'error'

    except Exception as e:
        print(f'✗ Unexpected error on {img_path.name}: {e}')
        return 'error'


# ---------------------------------------------------------------------------
# Mirror / orphan pruning
# ---------------------------------------------------------------------------

def _prune_orphans(out_root: Path, expected: set, dry_run: bool) -> tuple:
    """Delete files under out_root that are not in the expected set, then remove
    empty directories (deepest first). Returns (files_deleted, dirs_removed).

    In dry-run mode nothing is touched; deletions are only logged. Note that
    empty-dir detection in dry-run only catches dirs that are *already* empty,
    since the would-be-deleted files still exist."""
    files_deleted = 0
    dirs_removed = 0

    for f in out_root.rglob('*'):
        try:
            if not f.is_file():
                continue
            if 'thumbnails' in f.relative_to(out_root).parts:
                continue  # never touch nested thumbnails/ dirs (mirrors source skip rule)
            if f.resolve() in expected:
                continue
        except Exception:
            continue
        if dry_run:
            print(f'🗑  would delete orphan: {f}')
            files_deleted += 1
            continue
        try:
            f.unlink()
            print(f'🗑  deleted orphan: {f}')
            files_deleted += 1
        except Exception as e:
            print(f'✗ could not delete {f}: {e}')

    dirs = sorted(
        (p for p in out_root.rglob('*') if p.is_dir()),
        key=lambda p: len(p.parts), reverse=True,
    )
    for d in dirs:
        try:
            if any(d.iterdir()):
                continue
            if dry_run:
                print(f'🗑  would remove empty dir: {d}')
            else:
                d.rmdir()
                print(f'🗑  removed empty dir: {d}')
            dirs_removed += 1
        except Exception:
            pass

    return files_deleted, dirs_removed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def thumbnail_images(
    root_folder,
    output_folder=None,
    size=(1920, 1080),
    video_size=(640, 480),
    quality=85,
    recursive=True,
    overwrite=False,
    num_cores=None,
    mirror=False,
    dry_run=False,
    allow_empty=False,
):
    root = Path(root_folder)
    if not root.exists():
        print(f"Error: '{root_folder}' does not exist")
        return

    out_root = Path(output_folder) if output_folder else None

    if mirror and out_root is None:
        print("Warning: --mirror requires an output folder (-o); mirror disabled.")
        mirror = False

    pattern = '**/*' if recursive else '*'
    media = sorted(
        p for p in root.glob(pattern)
        if p.is_file()
        and p.suffix.lower() in MEDIA_EXTS
        and 'thumbnails' not in p.relative_to(root).parts   # skip existing thumbnail dirs
    )

    if not media:
        print(f"No media found in '{root_folder}'")
        if mirror and out_root and out_root.exists():
            if allow_empty:
                print("⚠  Source empty and --allow-empty set: pruning ALL thumbnails in output.")
                files_deleted, dirs_removed = _prune_orphans(out_root, set(), dry_run)
                print(f"Mirror prune ✓  deleted={files_deleted}  dirs_removed={dirs_removed}"
                      + ("  (dry-run)" if dry_run else ""))
            else:
                print("⚠  Source has no media — aborting mirror prune to avoid wiping the "
                      "destination. Re-run with --allow-empty if this is intentional.")
        return

    n = len(media)
    print(f'Found {n} media files. Processing…')

    def _out_for(p: Path) -> Path:
        ext = '.mp4' if p.suffix.lower() in VIDEO_EXTS else '.jpg'
        if out_root:
            return out_root / p.relative_to(root).with_suffix(ext)
        return p.parent / 'thumbnails' / p.with_suffix(ext).name

    expected = {_out_for(p).resolve() for p in media}

    if dry_run:
        if mirror and out_root:
            print('\n[dry-run] Mirror mode — orphans that WOULD be deleted:')
            files_deleted, dirs_removed = _prune_orphans(out_root, expected, dry_run=True)
            print(f'[dry-run] Mirror prune preview ✓  would_delete={files_deleted}  '
                  f'would_remove_dirs={dirs_removed}')
        else:
            print('[dry-run] No generation performed. Add --mirror to preview orphan deletions.')
        return

    if num_cores is None:
        num_cores = get_num_cores()

    gpu_available = _probe_nvenc()
    gpu_sem = threading.Semaphore(_int_cfg('THUMBNAIL_MAX_GPU_SESSIONS', 3))

    if gpu_available:
        print(f'GPU (NVENC) available — videos will use hardware encoding.')

    counter = [0]
    lock = threading.Lock()

    # Build task list (daemon placeholder = None, filled in below)
    tasks = []
    for p in media:
        out = _out_for(p)
        out.parent.mkdir(parents=True, exist_ok=True)
        tasks.append((p, out, size, video_size, quality, overwrite, counter, n, lock, gpu_sem, gpu_available, None))

    # Launch ExifTool daemon — single Perl process handles ALL metadata calls
    try:
        with ExifToolDaemon() as etd:
            if etd.ok:
                print('ExifTool daemon active (fast metadata mode).')
            # Inject daemon reference into every task tuple
            tasks = [(*t[:-1], etd) for t in tasks]
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as ex:
                results = list(ex.map(_process_one, tasks))
    except Exception as exc:
        print(f'Warning: ExifTool daemon failed ({exc}), falling back to per-call subprocess.')
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as ex:
            results = list(ex.map(_process_one, tasks))

    ok_n    = results.count('success')
    fixed_n = results.count('fixed')
    skip_n  = results.count('skipped')
    err_n   = results.count('error')
    print(f'\nDone ✓  generated={ok_n}  fixed={fixed_n}  skipped={skip_n}  errors={err_n}')

    if mirror and out_root:
        print('\nMirror mode — pruning orphan thumbnails…')
        files_deleted, dirs_removed = _prune_orphans(out_root, expected, dry_run=False)
        print(f'Mirror prune ✓  deleted={files_deleted}  dirs_removed={dirs_removed}')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Batch thumbnail generator with EXIF/metadata preservation.'
    )
    p.add_argument('input_folder', help='Source folder containing media files')
    p.add_argument(
        '-o', '--output', dest='output_folder',
        help='Output folder (default: thumbnails/ subfolder beside each source file)',
    )
    p.add_argument(
        '--size', nargs=2, type=int, default=[1920, 1080], metavar=('W', 'H'),
        help='Max image thumbnail dimensions in pixels (default: 1920 1080)',
    )
    p.add_argument(
        '--video-size', nargs=2, type=int, default=[640, 480], metavar=('W', 'H'),
        dest='video_size',
        help='Max video thumbnail dimensions in pixels (default: 640 480)',
    )
    p.add_argument('-q', '--quality', type=int, default=85,
                   help='JPEG quality 1-100 (default: 85)')
    p.add_argument('--no-recursive', dest='recursive', action='store_false', default=True,
                   help='Process top-level folder only')
    p.add_argument('--overwrite', action='store_true', default=False,
                   help='Regenerate thumbnails even if they already exist')
    p.add_argument('--num-cores', type=int, default=None,
                   help='Number of worker threads (default: auto)')
    p.add_argument('--mirror', action='store_true', default=False,
                   help='Mirror mode: delete output thumbnails whose source no longer '
                        'exists (and prune empty dirs). Requires -o.')
    p.add_argument('--dry-run', dest='dry_run', action='store_true', default=False,
                   help='Preview only: with --mirror, list orphans that would be deleted; '
                        'no thumbnails generated, nothing deleted.')
    p.add_argument('--allow-empty', dest='allow_empty', action='store_true', default=False,
                   help='Allow mirror pruning even when the source has no media '
                        '(otherwise aborts to avoid wiping the destination).')
    args = p.parse_args()

    thumbnail_images(
        root_folder=args.input_folder,
        output_folder=args.output_folder,
        size=tuple(args.size),
        video_size=tuple(args.video_size),
        quality=args.quality,
        recursive=args.recursive,
        overwrite=args.overwrite,
        num_cores=args.num_cores,
        mirror=args.mirror,
        dry_run=args.dry_run,
        allow_empty=args.allow_empty,
    )


if __name__ == '__main__':
    main()
