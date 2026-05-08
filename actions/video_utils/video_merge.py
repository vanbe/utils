#!/usr/bin/env python3
"""Concatenate two videos. video1 first, then video2.

Probes both files; if video codec, resolution, framerate and audio codec/rate
all match, uses ffmpeg concat demuxer with stream copy (fast, no quality loss).
Otherwise falls back to the concat filter with re-encoding so mismatched inputs
still merge cleanly.
"""

import sys
import os
import argparse
import subprocess
import json
import tempfile


def _probe(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-show_streams', '-of', 'json', path],
        capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    v = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), None)
    a = next((s for s in data.get('streams', []) if s.get('codec_type') == 'audio'), None)
    return {
        'v_codec':  v.get('codec_name')      if v else None,
        'width':    v.get('width')           if v else None,
        'height':   v.get('height')          if v else None,
        'fps':      v.get('r_frame_rate')    if v else None,
        'pix_fmt':  v.get('pix_fmt')         if v else None,
        'a_codec':  a.get('codec_name')      if a else None,
        'a_rate':   a.get('sample_rate')     if a else None,
        'a_chan':   a.get('channels')        if a else None,
    }


def _compatible(p1, p2):
    if p1 is None or p2 is None:
        return False
    keys = ['v_codec', 'width', 'height', 'fps', 'pix_fmt',
           'a_codec', 'a_rate', 'a_chan']
    return all(p1.get(k) == p2.get(k) for k in keys)


def _stream_copy_concat(v1, v2, out):
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        list_path = f.name
        f.write(f"file '{os.path.abspath(v1)}'\n")
        f.write(f"file '{os.path.abspath(v2)}'\n")
    try:
        cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
               '-i', list_path, '-c', 'copy', out]
        return subprocess.run(cmd).returncode
    finally:
        try: os.unlink(list_path)
        except OSError: pass


def _filter_concat(v1, v2, out):
    cmd = [
        'ffmpeg', '-y', '-i', v1, '-i', v2,
        '-filter_complex',
        '[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[v][a]',
        '-map', '[v]', '-map', '[a]',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
        '-c:a', 'aac', '-b:a', '192k',
        out,
    ]
    return subprocess.run(cmd).returncode


def merge(v1, v2, out):
    if subprocess.run(['which', 'ffmpeg'], capture_output=True).returncode != 0:
        print('Error: ffmpeg is not installed.')
        sys.exit(1)

    p1, p2 = _probe(v1), _probe(v2)
    if _compatible(p1, p2):
        print(f'Streams match — using fast concat (stream copy).')
        rc = _stream_copy_concat(v1, v2, out)
        if rc == 0:
            return rc
        print('Stream-copy concat failed; falling back to re-encode.')

    print('Streams differ — re-encoding (libx264 / aac).')
    return _filter_concat(v1, v2, out)


def main():
    ap = argparse.ArgumentParser(description='Merge two videos end-to-end.')
    ap.add_argument('video1', help='First video (plays first)')
    ap.add_argument('video2', help='Second video (appended)')
    ap.add_argument('--output', required=True, help='Output path')
    args = ap.parse_args()

    for v in (args.video1, args.video2):
        if not os.path.isfile(v):
            print(f'Error: {v} does not exist'); sys.exit(1)

    rc = merge(args.video1, args.video2, args.output)
    if rc == 0:
        print(f'Merged video: {args.output}')
    sys.exit(rc)


if __name__ == '__main__':
    main()
