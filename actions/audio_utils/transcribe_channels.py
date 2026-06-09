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
import wave

import numpy as np

from whisper_common import (load_whisper_model, write_srt, write_md,
                            channel_labels, recommended_model, is_hallucination,
                            is_echo_duplicate, _norm_phrase)
from audio_utils_common import SPEECH_ENHANCE_FILTERS
import aec as _aec


def sidecar_path(flac: str) -> str:
    return os.path.splitext(flac)[0] + '.channels.json'


def _demux_channel(flac: str, src: dict, out_wav: str, enhance: bool = False):
    """Démux la source `src` (moyenne de ses canaux) → mono 16 kHz PCM s16le.
    Si `enhance`, insère le réhaussement (débruitage + dynaudnorm) avant resample."""
    cs, ce = int(src['channel_start']), int(src['channel_end'])
    n = ce - cs
    # Moyenne des canaux de la source. La syntaxe `pan` n'accepte PAS `(c0+c1)/2`
    # (« Invalid argument ») : il faut des gains explicites `0.5*c0+0.5*c1`.
    coef = '+'.join(f'{1.0 / n:.6f}*c{c}' for c in range(cs, ce))
    pan = f'pan=mono|c0={coef}'
    af = f'{pan},{SPEECH_ENHANCE_FILTERS},aresample=16000' if enhance \
        else f'{pan},aresample=16000'
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', flac,
           '-af', af, '-ac', '1', '-c:a', 'pcm_s16le', out_wav]
    subprocess.run(cmd, check=True)


def _wav_read_f32(path: str) -> np.ndarray:
    """Lit un WAV mono PCM s16le → np.float32 dans [-1, 1]."""
    with wave.open(path, 'rb') as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0


def _wav_write_f32(path: str, x: np.ndarray, sr: int = 16000):
    """Écrit un np.float32 [-1, 1] → WAV mono PCM s16le."""
    pcm = np.clip(x, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _reference_mix(flac: str, outputs: list, tmp: str) -> np.ndarray | None:
    """Référence d'écho = moyenne 16 kHz mono de TOUTES les sorties (ce qui est
    réellement sorti des haut-parleurs). None si aucune sortie."""
    if not outputs:
        return None
    mixes = []
    for s in outputs:
        wav = os.path.join(tmp, f'ref{s["index"]}.wav')
        _demux_channel(flac, s, wav, enhance=False)     # réf brute (pas de réhaussement)
        mixes.append(_wav_read_f32(wav))
    n = min(len(m) for m in mixes)
    return np.mean([m[:n] for m in mixes], axis=0).astype(np.float32)


def transcribe_channels(flac: str, language: str = 'fr',
                        model_size: str | None = None,
                        enhance: bool = False, aec: bool = False) -> int:
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

    # AEC : on ne peut annuler l'écho que si l'on a AU MOINS une sortie (réf) ET
    # une entrée (micro contaminé). Sinon l'option est sans objet.
    outputs = [s for s in sources if s['kind'] == 'output']
    inputs = [s for s in sources if s['kind'] == 'input']
    aec_on = bool(aec and outputs and inputs)

    print(sep)
    print(f'  {os.path.basename(flac)}')
    print(f'  {len(sources)} canaux  ·  langue: {language}  ·  modèle: {model_size}'
          f'{"  ·  réhaussé" if enhance else ""}{"  ·  AEC" if aec_on else ""}')
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
        # Référence d'écho (moyenne des sorties) calculée une seule fois.
        ref_mix = _reference_mix(flac, outputs, tmp) if aec_on else None
        for i, src in enumerate(sources, 1):
            label = labels.get(src['index'], '')
            print(f'  [{i + 1}]  {label or src.get("name", "?")}', end='', flush=True)
            wav = os.path.join(tmp, f'ch{src["index"]}.wav')
            try:
                _demux_channel(flac, src, wav, enhance=enhance)
            except subprocess.CalledProcessError as e:
                print(f'   ✗ démux échoué ({e})')
                continue
            # AEC : sur les canaux MICRO, retire l'écho de la sortie système
            # recaptée par le micro (référence = ce qui est sorti des HP).
            if ref_mix is not None and src['kind'] == 'input':
                try:
                    mic = _wav_read_f32(wav)
                    cleaned = _aec.cancel_array(mic, ref_mix, 16000)
                    _wav_write_f32(wav, cleaned, 16000)
                    print('  (AEC)', end='', flush=True)
                except Exception as e:                  # AEC best-effort : on garde le micro brut
                    print(f'  (AEC ✗ {e})', end='', flush=True)
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
                    # Anti-hallucination (canal quasi-silencieux après AEC, etc.)
                    if is_hallucination(txt, getattr(s, 'avg_logprob', 0.0),
                                        getattr(s, 'no_speech_prob', 0.0)):
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

    # Dé-doublonnage d'écho (mêmes critères que le live) : phrase courte répétée
    # sur un canal ou quasi simultanée sur les deux (résidu d'écho / hallucination).
    segs.sort(key=lambda r: r['start'])
    recent, kept = [], []
    for r in segs:
        if is_echo_duplicate(r['text'], r['label'], r['start'], recent):
            continue
        recent.append((r['start'], r['label'], _norm_phrase(r['text'])))
        recent = [x for x in recent if abs(r['start'] - x[0]) <= 6.0]
        kept.append(r)
    segs = kept

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
    p.add_argument('--enhance', action='store_true',
                   help='réhausser un audio dégradé (capté de loin / bruyant)')
    p.add_argument('--aec', action='store_true',
                   help="annuler l'écho : retire du micro la sortie système recaptée "
                        '(utile si tu écoutes sur haut-parleurs plutôt qu\'au casque)')
    a = p.parse_args()
    sys.exit(transcribe_channels(' '.join(a.input_file), a.language, a.model,
                                 a.enhance, a.aec))
