#!/usr/bin/env python3
"""
transcribe_to_md.py — audio/video → `<base>_transcription.md` with speaker
labels AND `[hh:mm:ss]` timestamps (citable locations).

Thin wrapper around transcribe_audio.py (local faster-whisper + pyannote): it
runs it with `--output-format srt` (the only format carrying timings), then
rewrites the SRT as Markdown — consecutive segments of the same speaker are
merged into one paragraph stamped with its start time:

    [00:01:23] **Speaker 1**: …

The intermediate `.srt` and the pre-processed WAV are removed unless they
already existed before the run (or `--keep-srt`).

Usage:
  python transcribe_to_md.py <audio> [--keep-srt] [transcribe_audio.py options…]
  (e.g. --language en|fr|auto, --model large|turbo|…, --no-speaker,
   --max-speakers N, --no-improve, --enhance)
"""

import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRT_TIME = re.compile(r'(\d+):(\d+):(\d+)[,.](\d+)\s*-->')
_LABEL = re.compile(r'^(Speaker \d+):\s+(.*)$', re.S)   # labels written by transcribe_audio.py


def _base(path: str) -> str:
    # same derivation as transcribe_audio.py (_job_paths / output_file)
    base = os.path.splitext(path)[0]
    for marker in ('_quality_improved', '_converted', '_enhanced'):
        base = base.replace(marker, '')
    return base


def _gpu_env() -> dict:
    """Pin the GPU from TRANSCRIBE_GPU (env or utils .env) = nvidia-smi index.
    CUDA's default order is fastest-first, which differs from nvidia-smi's on
    multi-GPU machines → force PCI bus order. A CUDA_VISIBLE_DEVICES already set
    by the caller wins; unset key → CUDA default (unchanged behaviour)."""
    env = dict(os.environ)
    gpu = env.get('TRANSCRIBE_GPU')
    if gpu is None:
        try:
            from dotenv import dotenv_values
            gpu = dotenv_values(os.path.join(_HERE, '..', '..', '.env')).get('TRANSCRIBE_GPU')
        except ImportError:
            pass
    if gpu not in (None, '') and 'CUDA_VISIBLE_DEVICES' not in env:
        env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env


def _hms(sec: float) -> str:
    sec = int(sec)
    return f'{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}'


def parse_srt(text: str) -> list:
    segs = []
    for block in re.split(r'\n\s*\n', text.strip()):
        lines = block.strip().splitlines()
        tline = next((i for i, l in enumerate(lines) if '-->' in l), None)
        if tline is None:
            continue
        m = _SRT_TIME.search(lines[tline])
        start = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else 0
        body = ' '.join(l.strip() for l in lines[tline + 1:]).strip()
        if not body:
            continue
        lm = _LABEL.match(body)
        label, body = (lm.group(1).strip(), lm.group(2).strip()) if lm else ('', body)
        segs.append({'start': start, 'label': label, 'text': body})
    return segs


def to_markdown(segs: list, title: str) -> str:
    groups = []
    for s in segs:
        if groups and groups[-1]['label'] == s['label']:
            groups[-1]['text'].append(s['text'])
        else:
            groups.append({'start': s['start'], 'label': s['label'], 'text': [s['text']]})
    lines = [f'# {title}', '']
    for g in groups:
        who = f' **{g["label"]}**:' if g['label'] else ''
        lines += [f'[{_hms(g["start"])}]{who} ' + ' '.join(g['text']), '']
    return '\n'.join(lines).rstrip() + '\n'


def main():
    argv = sys.argv[1:]
    if not argv or argv[0].startswith('-'):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    src, opts = os.path.abspath(argv[0]), argv[1:]
    keep_srt = '--keep-srt' in opts
    opts = [o for o in opts if o != '--keep-srt']
    if '--output-format' in opts:            # we always need srt (timings)
        i = opts.index('--output-format')
        del opts[i:i + 2]

    base = _base(src)
    srt, md = base + '_transcription.srt', base + '_transcription.md'
    temps = [srt] + [base + s for s in ('_quality_improved.wav', '_converted.wav', '_enhanced.wav')]
    preexisting = {p for p in temps if os.path.exists(p)}

    cmd = [sys.executable, os.path.join(_HERE, 'transcribe_audio.py'), src,
           '--output-format', 'srt'] + opts
    proc = subprocess.run(cmd, env=_gpu_env())
    try:
        if proc.returncode != 0 or not os.path.isfile(srt):
            print(f'transcribe_audio.py failed (exit {proc.returncode})', file=sys.stderr)
            sys.exit(1)
        with open(srt, encoding='utf-8') as f:
            segs = parse_srt(f.read())
        with open(md, 'w', encoding='utf-8') as f:
            f.write(to_markdown(segs, os.path.basename(base)))
        print(f'  ✓ {len(segs)} segments → {os.path.basename(md)}', file=sys.stderr)
    finally:
        for p in temps:
            if p in preexisting or (p == srt and keep_srt):
                continue
            if os.path.exists(p):
                os.remove(p)


if __name__ == '__main__':
    main()
