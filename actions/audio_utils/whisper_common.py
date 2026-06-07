#!/usr/bin/env python3
"""
whisper_common.py — code Whisper partagé entre `transcribe_audio.py` (différé,
avec diarisation) et `live_transcribe.py` (live par canal).

Centralise ce qui était dupliqué : table des modèles, formatage SRT, choix
device/compute, chargement modèle (avec fallback OOM), et écriture SRT/MD.

Volontairement SANS import lourd au niveau module (torch / faster_whisper sont
importés paresseusement dans les fonctions) pour que l'import reste gratuit.
"""

import os

# Noms conviviaux → ids faster-whisper. large-v3 = meilleure qualité ;
# turbo = large-v3-turbo (plus rapide, qualité quasi identique).
MODEL_NAME_MAP = {
    'tiny': 'tiny', 'base': 'base', 'small': 'small', 'medium': 'medium',
    'large': 'large-v3', 'turbo': 'large-v3-turbo',
}


def _env_or_dotenv(key: str) -> str:
    """Valeur d'une clé via os.environ, sinon lue dans le .env projet (la TUI
    ne charge pas dotenv)."""
    val = os.environ.get(key)
    if val:
        return val.strip()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        with open(os.path.join(root, '.env'), encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith(key + '='):
                    return line.split('=', 1)[1].split('#')[0].strip()
    except OSError:
        pass
    return ''


def recommended_model(role: str = 'live') -> str:
    """
    Modèle Whisper **recommandé pour cette machine** (calculé par setup_env.py).
    role='live'       → temps réel (RTF<1, hôte réactif) : LIVE_TRANSCRIBE_MODEL
    role='transcribe' → différé/qualité (sans contrainte temps réel) : TRANSCRIBE_MODEL
    Sur GPU les deux sont identiques (turbo) ; sur CPU le live est plus léger.
    """
    if role == 'transcribe':
        return (_env_or_dotenv('TRANSCRIBE_MODEL')
                or _env_or_dotenv('LIVE_TRANSCRIBE_MODEL') or 'turbo')
    return _env_or_dotenv('LIVE_TRANSCRIBE_MODEL') or 'turbo'


def channel_labels(sources: list) -> dict:
    """source_index → libellé locuteur. micro → 'Moi', sortie → 'Système'.
    `sources` = liste de dicts avec au moins {'index', 'kind'} (cf. channel_map).
    Partagé entre live_transcribe.py (live) et transcribe_channels.py (différé)
    pour que l'attribution par canal soit STRICTEMENT identique des deux côtés."""
    labels, n_in, n_out = {}, 0, 0
    for s in sources:
        if s['kind'] == 'input':
            n_in += 1
            labels[s['index']] = 'Moi' if n_in == 1 else f'Moi {n_in}'
        else:
            n_out += 1
            labels[s['index']] = 'Système' if n_out == 1 else f'Système {n_out}'
    return labels


def format_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def _cuda_available() -> bool:
    """True si un GPU CUDA est dispo. Tolère l'absence de torch (faster-whisper
    repose sur CTranslate2, pas torch) → device='cpu' sans planter."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_empty():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def resolve_device_compute(device: str = '', compute: str = '') -> tuple[str, str]:
    """(device, compute_type) — autodétecte si non fourni (sans dépendre de torch)."""
    device = device or ('cuda' if _cuda_available() else 'cpu')
    compute = compute or ('float16' if device == 'cuda' else 'int8')
    return device, compute


def make_whisper_model(model_name: str, device: str, compute_type: str,
                       download_root: str | None = None, cpu_threads: int = 0):
    """Construit un WhisperModel (mappe le nom convivial). Import paresseux.
    cpu_threads>0 limite les threads CPU (garde l'hôte réactif en live)."""
    from faster_whisper import WhisperModel
    mapped = MODEL_NAME_MAP.get(model_name, model_name)
    kw = {}
    if download_root:
        kw['download_root'] = download_root
    if cpu_threads and device == 'cpu':
        kw['cpu_threads'] = int(cpu_threads)
    return WhisperModel(mapped, device=device, compute_type=compute_type, **kw)


def load_whisper_model(model_name: str, device: str = '', compute_type: str = '',
                       fallback_order: list | None = None,
                       download_root: str | None = None, cpu_threads: int = 0):
    """
    Charge un modèle avec **fallback OOM** vers des modèles plus légers.
    Renvoie (model, used_name, device, compute_type). Lève RuntimeError si échec.
    """
    device, compute_type = resolve_device_compute(device, compute_type)
    order = [model_name] + [m for m in (fallback_order
             or ['turbo', 'medium', 'small', 'base', 'tiny']) if m != model_name]
    last = None
    for name in order:
        try:
            _cuda_empty()
            model = make_whisper_model(name, device, compute_type, download_root, cpu_threads)
            return model, name, device, compute_type
        except Exception as e:                     # OOM ou autre → plus léger
            last = e
            _cuda_empty()
    raise RuntimeError(f'aucun modèle Whisper chargeable: {last}')


# ---------------------------------------------------------------------------
# Écriture des sorties — segments = [{start, end, label, text}]
# (label peut être '' ; les segments sont triés par start)
# ---------------------------------------------------------------------------

def write_srt(path: str, segments: list):
    segs = sorted(segments, key=lambda r: r['start'])
    with open(path, 'w', encoding='utf-8') as f:
        for i, r in enumerate(segs, 1):
            lab = (r.get('label') or '').strip()
            text = (f'{lab}: ' if lab else '') + r['text'].strip()
            f.write(f'{i}\n{format_srt_time(r["start"])} --> '
                    f'{format_srt_time(r["end"])}\n{text}\n\n')


def write_md(path: str, segments: list, title: str = 'Transcription'):
    """Markdown SANS timings : blocs `**Label**: …`, fusion des labels consécutifs."""
    segs = sorted(segments, key=lambda r: r['start'])
    groups, cur = [], None
    for r in segs:
        lab = (r.get('label') or '').strip()
        txt = r['text'].strip()
        if not txt:
            continue
        if cur and cur['label'] == lab:
            cur['text'].append(txt)
        else:
            if cur:
                groups.append(cur)
            cur = {'label': lab, 'text': [txt]}
    if cur:
        groups.append(cur)
    lines = [(f'**{g["label"]}**: ' if g['label'] else '') + ' '.join(g['text'])
             for g in groups]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'# {title}\n\n' + '\n\n'.join(lines) + ('\n' if lines else ''))
