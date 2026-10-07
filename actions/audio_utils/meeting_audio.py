#!/usr/bin/env python3
"""
meeting_audio.py — préparation audio d'une réunion pour la transcription distante
(OVH via LiteLLM) : métadonnées, extraction par source, anti-écho AUTOMATIQUE,
réhaussement, retrait des blancs (avec table de correspondance des timestamps),
découpage en tranches, fichier d'écoute `_enhanced`.

Tout est en ffmpeg + numpy (aucune dépendance lourde) et en streaming par blocs
pour tenir en RAM sur des captures de plusieurs heures (FLAC 4 canaux 1,5 Go).

Chaîne par source :
  brut 16 kHz mono (pan = moyenne des canaux de la source)
    → [AEC si écho détecté : micro − sortie recaptée, sur le BRUT (filtre linéaire)]
    → réhaussement (highpass + afftdn + dynaudnorm + loudnorm)
    → VAD énergie (sur le brut : le dynaudnorm remonte le bruit de fond)
    → concaténation des zones de parole + table (concat → original)
"""

import bisect
import json
import re
import subprocess
import wave
from datetime import datetime

import numpy as np

from audio_utils_common import SPEECH_ENHANCE_FILTERS

SR = 16000
ENHANCE_CHAIN = f'{SPEECH_ENHANCE_FILTERS},loudnorm=I=-16:TP=-1.5:LRA=11'


# ── Métadonnées : date + sujet depuis le nom de fichier ────────────────────────

_MONTHS_FR = {'janv': 1, 'jan': 1, 'févr': 2, 'fevr': 2, 'fév': 2, 'mars': 3, 'avr': 4,
              'mai': 5, 'juin': 6, 'juil': 7, 'août': 8, 'aout': 8, 'sept': 9, 'oct': 10,
              'nov': 11, 'déc': 12, 'dec': 12}

_GENERIC_SUBJECT = re.compile(
    r'^(capture|enregistrement|nouvel enregistrement|recording|new recording|audio|'
    r'memo|mémo|voice memo|record)?\s*(\(\d+\))?\s*(-\s*copie)?\s*(\(\d+\))?$', re.I)


def _rest_subject(rest: str | None) -> str | None:
    rest = (rest or '').strip(' -_')
    if not rest or _GENERIC_SUBJECT.match(rest):
        return None
    return rest


def parse_name(stem: str, mtime: float, creation_time: str | None = None):
    """→ (datetime local naïf, sujet ou None). Formats reconnus :
    `2026-10-02-09-30 - Sujet` (Record audio), `13.08.2026 13.04.23` (iPhone),
    `2 sept. à 14-41` (Dictaphone iOS FR, année déduite de mtime), `2026-10-02 - Sujet`.
    Sinon : creation_time (tags audio) puis mtime ; sujet = nom entier s'il n'est
    pas générique (« capture », « Nouvel enregistrement (2) »…)."""
    s = stem.strip()
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})[-_ ](\d{2})[-_h.](\d{2})(?:\s*-\s*(.*))?$', s)
    if m:
        y, mo, d, h, mi = map(int, m.groups()[:5])
        return datetime(y, mo, d, h, mi), _rest_subject(m.group(6))
    m = re.match(r'^(\d{2})\.(\d{2})\.(\d{4}) (\d{2})\.(\d{2})\.(\d{2})(?:\s*-\s*(.*))?$', s)
    if m:
        d, mo, y, h, mi, se = map(int, m.groups()[:6])
        return datetime(y, mo, d, h, mi, se), _rest_subject(m.group(7))
    m = re.match(r'^(\d{1,2}) ([a-zéû]+)\.? à (\d{1,2})[-h.:](\d{2})(?:\s*-\s*(.*))?$', s, re.I)
    if m and m.group(2).lower() in _MONTHS_FR:
        ref = datetime.fromtimestamp(mtime)
        mo = _MONTHS_FR[m.group(2).lower()]
        y = ref.year - (1 if mo > ref.month else 0)
        return (datetime(y, mo, int(m.group(1)), int(m.group(3)), int(m.group(4))),
                _rest_subject(m.group(5)))
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})(?:\s*-\s*(.*))?$', s)
    if m:
        y, mo, d = map(int, m.groups()[:3])
        return datetime(y, mo, d), _rest_subject(m.group(4))
    dt = None
    if creation_time:
        try:
            dt = datetime.fromisoformat(creation_time.replace('Z', '+00:00')).astimezone()
            dt = dt.replace(tzinfo=None)
        except ValueError:
            dt = None
    return dt or datetime.fromtimestamp(mtime), _rest_subject(s)


def safe_filename(s: str, max_len: int = 110) -> str:
    """Nom compatible SMB/Windows : retire \\/:*?"<>| et les caractères de contrôle."""
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip(' .')
    return s[:max_len].rstrip(' .') or 'Réunion'


# ── ffprobe / ffmpeg ───────────────────────────────────────────────────────────

def probe(path: str) -> dict:
    out = subprocess.run(['ffprobe', '-v', 'error', '-print_format', 'json',
                          '-show_format', '-show_streams', path],
                         check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    st = next((s for s in info.get('streams', []) if s.get('codec_type') == 'audio'), {})
    tags = {k.lower(): v for k, v in (info.get('format', {}).get('tags') or {}).items()}
    return {'channels': int(st.get('channels') or 1),
            'sample_rate': int(st.get('sample_rate') or 0),
            'codec': st.get('codec_name'),
            'creation_time': tags.get('creation_time')}


def _ffmpeg(*args):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', *args],
                   check=True)


def pan_expr(channel_start: int, channel_end: int) -> str:
    """Moyenne des canaux [start, end) → mono. (`pan` n'accepte pas (c0+c1)/2.)"""
    n = channel_end - channel_start
    return 'pan=mono|c0=' + '+'.join(f'{1.0 / n:.6f}*c{c}'
                                     for c in range(channel_start, channel_end))


def extract_raw(src: str, out_wav: str, channels: tuple[int, int] | None = None):
    """Source → WAV brut 16 kHz mono s16le (channels=None : downmix de tout)."""
    af = (pan_expr(*channels) + ',' if channels else '') + f'aresample={SR}'
    _ffmpeg('-i', src, '-af', af, '-ac', '1', '-c:a', 'pcm_s16le', out_wav)


def enhance(in_wav: str, out_wav: str):
    _ffmpeg('-i', in_wav, '-af', f'{ENHANCE_CHAIN},aresample={SR}', '-ac', '1',
            '-c:a', 'pcm_s16le', out_wav)


def read_wav(path: str) -> np.ndarray:
    """WAV mono s16le → int16 (pas de float : 2 h = 230 Mo au lieu de 460)."""
    with wave.open(path, 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2')


def wav_seconds(path: str) -> float:
    with wave.open(path, 'rb') as w:
        return w.getnframes() / w.getframerate()


def write_wav(path: str, pcm: np.ndarray):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.asarray(pcm, dtype='<i2').tobytes())


def write_flac(path: str, pcm: np.ndarray):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 's16le',
                    '-ar', str(SR), '-ac', '1', '-i', '-', '-c:a', 'flac', path],
                   input=np.asarray(pcm, dtype='<i2').tobytes(), check=True)


# ── Anti-écho automatique ──────────────────────────────────────────────────────

def _frame_db(pcm: np.ndarray, frame: int) -> np.ndarray:
    """dBFS par trame (calcul par blocs pour borner la RAM)."""
    n = pcm.size // frame
    out = np.empty(n, dtype=np.float32)
    blk = 20000
    for i in range(0, n, blk):
        j = min(n, i + blk)
        x = pcm[i * frame:j * frame].astype(np.float32).reshape(j - i, frame) / 32768.0
        out[i:j] = 10.0 * np.log10(np.mean(x * x, axis=1) + 1e-10)
    return out


ECHO_COHERENCE_THRESHOLD = 0.3


def detect_echo(mic: np.ndarray, ref: np.ndarray, n_windows: int = 16,
                win_s: float = 3.0) -> float:
    """Cohérence micro↔sortie (0..1) mesurée sur les fenêtres où la SORTIE est la
    plus forte (quelqu'un parle à distance). Médiane > ECHO_COHERENCE_THRESHOLD ⇒ le
    micro recapte les haut-parleurs (pas de casque). Renvoie la médiane (métadonnée)."""
    import aec
    w = int(win_s * SR)
    n = min(mic.size, ref.size) // w
    if n < 2:
        return 0.0
    db = _frame_db(ref[:n * w], w)
    picks = sorted(np.argsort(db)[::-1][:n_windows])    # les plus fortes, dans l'ordre
    vals = []
    for k in picks:
        if db[k] < -45:                                  # sortie quasi muette : non pertinent
            continue
        m = mic[k * w:(k + 1) * w].astype(np.float64) / 32768.0
        r = ref[k * w:(k + 1) * w].astype(np.float64) / 32768.0
        vals.append(aec.echo_coherence(m, r, SR))
    return float(np.median(vals)) if vals else 0.0


def cancel_echo(mic: np.ndarray, ref: np.ndarray, block_s: float = 30.0) -> np.ndarray:
    """AEC en streaming par blocs (mémoire bornée) ; délai estimé (GCC-PHAT) sur les
    5 premières minutes puis appliqué à toute la référence."""
    import aec
    n = min(mic.size, ref.size)
    probe_n = min(n, 300 * SR)
    d = aec.estimate_delay(mic[:probe_n].astype(np.float64) / 32768.0,
                           ref[:probe_n].astype(np.float64) / 32768.0, SR)
    if d > 0:                                            # écho en retard : avance la réf
        ref = np.concatenate([ref[d:n], np.zeros(d, dtype=ref.dtype)])
    elif d < 0:
        ref = np.concatenate([np.zeros(-d, dtype=ref.dtype), ref[:n + d]])
    ec = aec.EchoCanceller(SR)
    out = np.empty(n, dtype=np.int16)
    pos, b = 0, int(block_s * SR)
    for i in range(0, n, b):
        y = ec.process(mic[i:i + b].astype(np.float64) / 32768.0,
                       ref[i:i + b].astype(np.float64) / 32768.0)
        out[pos:pos + y.size] = np.clip(y * 32767.0, -32768, 32767).astype(np.int16)
        pos += y.size
    y = ec.flush()
    out[pos:pos + y.size] = np.clip(y * 32767.0, -32768, 32767).astype(np.int16)
    return out


# ── Retrait des blancs ─────────────────────────────────────────────────────────

def speech_regions(raw: np.ndarray, frame_ms: int = 30, min_gap_s: float = 0.8,
                   pad_s: float = 0.25, min_len_s: float = 0.3) -> list:
    """VAD énergie → [(start_s, end_s)]. Seuil RELATIF : plancher de bruit (10ᵉ
    percentile) + 12 dB, borné à [-55, -30] dBFS. Volontairement conservateur
    (garder un peu de bruit coûte moins qu'une phrase coupée) ; le VAD serveur
    (`chunking_strategy=auto`) fait la passe fine."""
    frame = SR * frame_ms // 1000
    db = _frame_db(raw, frame)
    if db.size == 0:
        return []
    thr = float(np.clip(np.percentile(db, 10) + 12.0, -55.0, -30.0))
    active = db > thr
    regions, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            regions.append([start, i])
            start = None
    if start is not None:
        regions.append([start, active.size])
    fs = frame_ms / 1000.0
    merged = []
    for s, e in regions:
        s, e = s * fs, e * fs
        if merged and s - merged[-1][1] < min_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    total = raw.size / SR
    out = []
    for s, e in merged:
        if e - s < min_len_s:
            continue
        out.append((max(0.0, s - pad_s), min(total, e + pad_s)))
    # le padding peut faire se chevaucher deux zones : on refusionne
    fused = []
    for s, e in out:
        if fused and s <= fused[-1][1]:
            fused[-1] = (fused[-1][0], max(fused[-1][1], e))
        else:
            fused.append((s, e))
    return fused


class TimeMap:
    """Correspondance temps « concaténé » (parole seule) → temps original."""

    def __init__(self, regions: list):
        self.concat_starts, self.orig_starts, self.lengths = [], [], []
        t = 0.0
        for s, e in regions:
            self.concat_starts.append(t)
            self.orig_starts.append(s)
            self.lengths.append(e - s)
            t += e - s
        self.total = t

    def to_orig(self, t: float) -> float:
        if not self.concat_starts:
            return t
        i = max(0, bisect.bisect_right(self.concat_starts, t) - 1)
        return self.orig_starts[i] + min(max(0.0, t - self.concat_starts[i]), self.lengths[i])


def build_chunks(pcm: np.ndarray, regions: list, max_speech_s: float) -> list:
    """Regroupe les zones de parole en tranches ≤ `max_speech_s` de parole (coupées
    ENTRE deux zones = sur un silence). → [(pcm_concat, TimeMap)]."""
    chunks, cur = [], []
    acc = 0.0
    for s, e in regions:
        if cur and acc + (e - s) > max_speech_s:
            chunks.append(cur)
            cur, acc = [], 0.0
        cur.append((s, e))
        acc += e - s
    if cur:
        chunks.append(cur)
    out = []
    for regs in chunks:
        parts = [pcm[int(s * SR):int(e * SR)] for s, e in regs]
        out.append((np.concatenate(parts) if parts else np.zeros(0, np.int16), TimeMap(regs)))
    return out


# ── Fichier d'écoute ───────────────────────────────────────────────────────────

def write_listening_m4a(out_path: str, left_wav: str, right_wav: str | None = None):
    """Audio réhaussé à télécharger : stéréo (G = Moi, D = les autres) si deux
    pistes, sinon mono. AAC 64 kb/s (voix 16 kHz)."""
    if right_wav:
        _ffmpeg('-i', left_wav, '-i', right_wav, '-filter_complex',
                '[0:a][1:a]join=inputs=2:channel_layout=stereo[a]', '-map', '[a]',
                '-c:a', 'aac', '-b:a', '64k', '-movflags', '+faststart', out_path)
    else:
        _ffmpeg('-i', left_wav, '-c:a', 'aac', '-b:a', '48k', '-movflags', '+faststart',
                out_path)


def mix_wavs(paths: list, out_wav: str):
    """Moyenne de plusieurs WAV mono (≥2 entrées ou ≥2 sorties) → un WAV mono."""
    if len(paths) == 1:
        _ffmpeg('-i', paths[0], '-c:a', 'pcm_s16le', out_wav)
        return
    ins = []
    for p in paths:
        ins += ['-i', p]
    _ffmpeg(*ins, '-filter_complex', f'amix=inputs={len(paths)}:normalize=1', '-ac', '1',
            '-c:a', 'pcm_s16le', out_wav)
