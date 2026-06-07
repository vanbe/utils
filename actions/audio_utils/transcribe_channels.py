#!/usr/bin/env python3
"""
transcribe_channels.py — transcription DIFFÉRÉE *par canal* d'un FLAC multicanal
produit par recorder.py, en s'appuyant sur le sidecar `<name>.channels.json`.

Contrairement à `transcribe_audio.py` (downmix mono + diarisation pyannote), ici
l'attribution du locuteur est connue **par construction** : chaque source occupe
un intervalle de canaux connu (micro → « Moi », sortie → « Système »). On démux
chaque source vers du mono 16 kHz, on transcrit, puis on fusionne tous les
segments triés par timestamp. C'est le pendant différé de `live_transcribe.py`
— mêmes labels (`channel_labels`) et mêmes writers (`write_srt`/`write_md`), donc
sorties STRICTEMENT identiques au live.

Sorties : `<base>.srt` (avec timings) ET `<base>.md` (sans timings).
Exit 0 si OK, 1 sinon (consommé par la TUI via `_show_result`).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

from whisper_common import (load_whisper_model, write_srt, write_md,
                            channel_labels, recommended_model)


def sidecar_path(flac: str) -> str:
    return os.path.splitext(flac)[0] + '.channels.json'


def _demux_channel(flac: str, src: dict, out_wav: str):
    """Démux la source `src` (moyenne de ses canaux) → mono 16 kHz PCM s16le."""
    cs, ce = int(src['channel_start']), int(src['channel_end'])
    n = ce - cs
    terms = '+'.join(f'c{c}' for c in range(cs, ce))
    coef = f'({terms})/{n}' if n > 1 else terms        # moyenne si multi-canaux
    pan = f'pan=mono|c0={coef}'
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', flac,
           '-af', f'{pan},aresample=16000', '-ac', '1',
           '-c:a', 'pcm_s16le', out_wav]
    subprocess.run(cmd, check=True)


def transcribe_channels(flac: str, language: str = 'fr',
                        model_size: str | None = None) -> int:
    sep = '─' * 56
    t_total = time.time()

    if not os.path.exists(flac):
        print(f'\n  Error: file not found: {flac}')
        return 1
    side = sidecar_path(flac)
    if not os.path.exists(side):
        print(f'\n  Error: sidecar introuvable : {os.path.basename(side)}\n'
              "  (ce mode requiert un FLAC produit par « Record audio »).")
        return 1
    try:
        with open(side, encoding='utf-8') as f:
            cmap = json.load(f)
        sources = cmap['sources']
    except (OSError, ValueError, KeyError) as e:
        print(f'\n  Error: channels.json illisible : {e}')
        return 1

    lang = None if (language or '').lower() in ('auto', '') else language
    model_size = model_size or recommended_model('transcribe')
    labels = channel_labels(sources)

    print(sep)
    print(f'  {os.path.basename(flac)}')
    print(f'  {len(sources)} canaux  ·  langue: {language}  ·  modèle: {model_size}')
    print(sep)

    print(f'  [1]  Load Whisper  ({model_size})', end='', flush=True)
    try:
        model, used, device, compute = load_whisper_model(
            model_size,
            download_root=os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))), 'vendor', 'models', 'whisper'))
    except Exception as e:
        print(f'\n  Error: {e}')
        return 1
    detail = used if used == model_size else f'{used} (fallback)'
    print(f'   ✓  {detail} · {device}/{compute}')

    segs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, src in enumerate(sources, 1):
            label = labels.get(src['index'], '')
            print(f'  [{i + 1}]  {label or src.get("name", "?")}', end='', flush=True)
            wav = os.path.join(tmp, f'ch{src["index"]}.wav')
            try:
                _demux_channel(flac, src, wav)
            except subprocess.CalledProcessError as e:
                print(f'   ✗ démux échoué ({e})')
                continue
            try:
                it, _ = model.transcribe(wav, language=lang, beam_size=5,
                                         vad_filter=True,
                                         vad_parameters=dict(min_silence_duration_ms=500),
                                         condition_on_previous_text=True)
                n = 0
                for s in it:
                    txt = (s.text or '').strip()
                    if not txt:
                        continue
                    segs.append({'start': s.start, 'end': s.end,
                                 'label': label, 'text': txt})
                    n += 1
                print(f'   ✓  {n} segments')
            except Exception as e:
                print(f'   ✗ {e}')

    if not segs:
        print('\n  ✗ Aucune parole détectée sur aucun canal.')
        return 1

    base = os.path.splitext(flac)[0]
    srt_out, md_out = base + '.srt', base + '.md'
    write_srt(srt_out, segs)
    write_md(md_out, segs)

    print()
    print(sep)
    print(f'  Total:  {time.time() - t_total:.1f}s  ·  {len(segs)} segments')
    print(f'  Saved:  {os.path.basename(srt_out)}  +  {os.path.basename(md_out)}')
    print(sep)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Transcription par canal (channels.json)')
    p.add_argument('input_file', nargs='+', help='FLAC multicanal (+ sidecar channels.json)')
    p.add_argument('--language', default='fr', help='fr | en | auto')
    p.add_argument('--model', default=None,
                   help='tiny|base|small|medium|large|turbo (défaut: recommandé .env)')
    a = p.parse_args()
    sys.exit(transcribe_channels(' '.join(a.input_file), a.language, a.model))
