#!/usr/bin/env python3
"""
setup_env.py — Detect hardware and write optimised defaults to .env.

- Never overwrites values already present in .env
- Preserves tokens, NAS config, and any user-set values
- Safe to run multiple times

Usage:
  python setup_env.py           # auto-detect and update .env
  python setup_env.py --dry-run # print what would change, write nothing
  python setup_env.py --force   # overwrite ALL auto-detected values even if set
"""

import os
import sys
import re
import argparse
import subprocess
import platform

_ROOT = os.path.dirname(os.path.abspath(__file__))
_ENV  = os.path.join(_ROOT, '.env')

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

class HardwareProfile:
    cpu_cores:     int   = 1
    ram_gb:        float = 4.0
    gpu_name:      str   = ''
    gpu_vram_mb:   int   = 0
    gpu_vendor:    str   = ''   # 'nvidia' | 'amd' | 'intel' | ''
    is_wsl:        bool  = False
    has_cuda:      bool  = False
    has_rocm:      bool  = False


def detect_hardware() -> HardwareProfile:
    h = HardwareProfile()

    # WSL detection
    try:
        with open('/proc/version') as f:
            h.is_wsl = 'microsoft' in f.read().lower()
    except Exception:
        pass

    # CPU cores (physical)
    try:
        r = subprocess.run(['lscpu'], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if re.match(r'CPU\(s\):', line):
                h.cpu_cores = int(line.split(':')[1].strip())
            # If hyperthreading: divide by threads-per-core
            if re.match(r'Thread\(s\) per core:', line):
                tpc = int(line.split(':')[1].strip())
                if tpc > 1:
                    h.cpu_cores = max(1, h.cpu_cores // tpc)
    except Exception:
        h.cpu_cores = os.cpu_count() or 4

    # RAM
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    kb = int(line.split()[1])
                    h.ram_gb = kb / 1024 / 1024
                    break
    except Exception:
        pass

    # NVIDIA GPU
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().split(',')]
            h.gpu_name    = parts[0]
            h.gpu_vram_mb = int(parts[1])
            h.gpu_vendor  = 'nvidia'
            h.has_cuda    = True
    except Exception:
        pass

    # AMD GPU via rocm-smi
    if not h.gpu_vendor:
        try:
            r = subprocess.run(['rocm-smi', '--showproductname'],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                h.gpu_vendor = 'amd'
                h.has_rocm   = True
                for line in r.stdout.splitlines():
                    if 'Card series' in line or 'GPU' in line:
                        h.gpu_name = line.split(':')[-1].strip()
                        break
                # Try to get VRAM via lspci
                r2 = subprocess.run(['lspci', '-v'], capture_output=True, text=True)
                for line in r2.stdout.splitlines():
                    m = re.search(r'(\d+)M prefetchable', line)
                    if m:
                        candidate = int(m.group(1))
                        if candidate > 512:
                            h.gpu_vram_mb = candidate
                            break
        except Exception:
            pass

    # Intel integrated (fallback)
    if not h.gpu_vendor:
        try:
            r = subprocess.run(['lspci'], capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if re.search(r'Intel.*Graphics|Intel.*VGA', line, re.I):
                    h.gpu_vendor = 'intel'
                    h.gpu_name   = line.split(':')[-1].strip()
                    break
        except Exception:
            pass

    return h


# ---------------------------------------------------------------------------
# Compute optimal .env values from hardware
# ---------------------------------------------------------------------------

def _compute_defaults(h: HardwareProfile) -> dict[str, tuple[str, str]]:
    """Returns {key: (value, comment)} for auto-detected keys."""

    d: dict[str, tuple[str, str]] = {}
    cores = h.cpu_cores

    # --- OMP_NUM_THREADS ---
    # GPU present: leave ~4 cores free for GPU data feed + OS
    # CPU-only: use all cores
    if h.has_cuda or h.has_rocm:
        omp = max(1, cores - 4)
    elif h.gpu_vendor == 'intel':
        omp = max(1, cores - 2)
    else:
        omp = cores
    d['OMP_NUM_THREADS'] = (
        str(omp),
        f'CPU threads for torch/whisper  ({cores} physical cores detected,'
        f' {"GPU present" if h.gpu_vendor else "CPU-only mode"})'
    )

    # --- THUMBNAIL_MAX_GPU_SESSIONS ---
    if h.has_cuda:
        # NVENC session limits by VRAM tier
        if   h.gpu_vram_mb >= 12000: sessions = 8
        elif h.gpu_vram_mb >= 8000:  sessions = 5
        elif h.gpu_vram_mb >= 6000:  sessions = 3
        elif h.gpu_vram_mb >= 4000:  sessions = 2
        else:                        sessions = 1
    elif h.has_rocm:
        sessions = 3   # ROCm/VCE encoding, conservative
    else:
        sessions = 0   # CPU-only thumbnailing
    d['THUMBNAIL_MAX_GPU_SESSIONS'] = (
        str(sessions),
        f'Max concurrent GPU encoding sessions  '
        f'({h.gpu_name or "no GPU"}, {h.gpu_vram_mb} MB VRAM)'
    )

    # --- THUMBNAIL_NUM_CORES ---
    # Reserve 2 cores for GPU feed + OS on GPU systems, use 80% on CPU-only
    if h.gpu_vendor:
        thumb_cores = max(1, cores - 2)
    else:
        thumb_cores = max(1, int(cores * 0.8))
    d['THUMBNAIL_NUM_CORES'] = (
        str(thumb_cores),
        f'CPU cores for parallel thumbnail processing  ({cores} total)'
    )

    # --- RAW_TO_JPG_NUM_CORES ---
    # RAW decode is CPU-bound; use all available
    raw_cores = cores
    d['RAW_TO_JPG_NUM_CORES'] = (
        str(raw_cores),
        f'CPU cores for parallel RAW→JPEG conversion'
    )

    # --- Transcription Whisper (live "Record audio" + différé "Transcribe") ---
    # Modèles RECOMMANDÉS calculés selon le matériel :
    #  - live  : doit tenir le TEMPS RÉEL (RTF<1) en gardant l'hôte réactif.
    #  - différé: pas de contrainte temps réel → on peut viser la MEILLEURE qualité.
    # Sur GPU, turbo (≈ qualité large) suffit pour les deux. Sur CPU on allège le
    # live (base = temps réel) et on garde un modèle plus fort pour le différé.
    cuda, vram = h.has_cuda, h.gpu_vram_mb
    if cuda and vram >= 10000:
        live_model, transcribe_model, device, compute = 'turbo', 'large', 'cuda', 'float16'
    elif cuda and vram >= 5000:
        live_model, transcribe_model, device, compute = 'turbo', 'turbo', 'cuda', 'float16'
    elif cuda and vram >= 3000:
        live_model, transcribe_model, device, compute = 'small', 'small', 'cuda', 'int8_float16'
    else:
        live_model, transcribe_model, device, compute = 'base', 'small', 'cpu', 'int8'
    live_threads = max(1, cores - 1)   # laisse 1 cœur à l'hôte (réactivité)

    d['LIVE_TRANSCRIBE_MODEL'] = (
        live_model,
        f'Modèle Whisper RECOMMANDÉ pour le LIVE (temps réel)  '
        f'({h.gpu_name or "CPU"}, {vram} MB VRAM — un seul modèle partagé)'
    )
    d['TRANSCRIBE_MODEL'] = (
        transcribe_model,
        'Modèle RECOMMANDÉ pour la transcription DIFFÉRÉE (qualité, sans contrainte temps réel)'
    )
    d['LIVE_TRANSCRIBE_DEVICE']       = (device,  'cuda | cpu')
    d['LIVE_TRANSCRIBE_COMPUTE_TYPE'] = (compute, 'float16 | int8_float16 | int8')
    d['LIVE_TRANSCRIBE_CPU_THREADS']  = (str(live_threads),
        f'threads CPU du live ({cores} cœurs − 1 → hôte réactif ; ignoré sur GPU)')
    d['LIVE_TRANSCRIBE_LANGUAGE']     = ('fr',  "langue fixe (fiable en live) ; 'auto' = détection")
    d['LIVE_TRANSCRIBE_WINDOW_SEC']   = ('15',  "longueur max d'un énoncé avant flush vers Whisper")
    d['LIVE_TRANSCRIBE_INTERIM_SEC']  = ('2',   'aperçus temps réel toutes les N s (0 = désactivé)')

    # --- Diarisation LIVE des intervenants (canal Système → P1/P2/P3) ---
    # Embedding de voix par énoncé sur CPU + clustering en ligne. À CALIBRER sur
    # de l'audio réel : seuil cosinus plus BAS = fusionne plus (moins de
    # locuteurs), plus HAUT = sépare plus (risque de scinder un même locuteur).
    d['LIVE_DIARIZE_THRESHOLD']    = ('0.5', 'cosinus ≥ seuil → même personne (0.4–0.7 ; à calibrer)')
    d['LIVE_DIARIZE_MAX_SPEAKERS'] = ('8',   'nombre max d’intervenants distincts sur le canal Système')

    # --- Garde anti-écho au niveau énoncé (HP recaptés par le micro) ---
    # Jette un énoncé « Moi » qui est en réalité l'écho de la sortie (cohérence
    # spectrale micro↔sortie ≥ seuil). Sûr au casque (voix décorrélée → coh≈0) et
    # en double-talk (voix ≳ écho → coh sous le seuil). Plus BAS = plus agressif.
    d['LIVE_ECHO_GATE']     = ('1',    'jeter les énoncés Moi dominés par l’écho HP (1 = oui, 0 = non)')
    d['LIVE_ECHO_GATE_COH'] = ('0.65', 'cohérence micro↔sortie ≥ seuil → écho recapté (0.55–0.75)')

    return d


# ---------------------------------------------------------------------------
# .env file read / write helpers
# ---------------------------------------------------------------------------

def _read_env(path: str) -> dict[str, str]:
    """Parse .env into {key: raw_line} preserving order and comments."""
    result: dict[str, str] = {}
    if not os.path.exists(path):
        return result
    with open(path, encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                m = re.match(r'^([A-Z_][A-Z0-9_]*)=', stripped)
                if m:
                    result[m.group(1)] = stripped.split('=', 1)[1]
    return result


def _read_env_lines(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return f.readlines()


def _write_env(path: str, lines: list[str]):
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

_SECTION_HEADER = """\
# =============================================================================
# Auto-detected hardware configuration
# Generated by setup_env.py — edit manually to override
# Machine: {machine}
# =============================================================================
"""


_PLACEHOLDER_TOKEN = 'your_hugging_face_token_here'


def _bootstrap_env():
    """Copy .env.example → .env if .env doesn't exist yet."""
    example = os.path.join(_ROOT, '.env.example')
    if not os.path.exists(_ENV) and os.path.exists(example):
        import shutil
        shutil.copy(example, _ENV)
        print(f'  Created .env from .env.example')
        return True
    return False


def _prompt_hf_token():
    """
    If AUDIO_UTILS_HF_TOKEN is still the placeholder (or missing), ask the user
    to paste their HuggingFace token and write it into .env.
    Skipped in non-interactive mode (e.g. piped stdin).
    """
    existing = _read_env(_ENV)
    current  = existing.get('AUDIO_UTILS_HF_TOKEN', _PLACEHOLDER_TOKEN)

    if current and current != _PLACEHOLDER_TOKEN:
        return  # already set

    # Skip if stdin is not a real terminal (CI, piped, etc.)
    if not sys.stdin.isatty():
        print('  AUDIO_UTILS_HF_TOKEN not set — set it manually in .env')
        print('  (get your token at https://huggingface.co/settings/tokens)')
        return

    print('  ┌─────────────────────────────────────────────────────────────┐')
    print('  │  HuggingFace token required for speaker diarization         │')
    print('  │  Get yours at: https://huggingface.co/settings/tokens       │')
    print('  │  Leave blank to skip — you can add it to .env later         │')
    print('  └─────────────────────────────────────────────────────────────┘')
    try:
        token = input('  Paste token: ').strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not token:
        print('  Skipped — set AUDIO_UTILS_HF_TOKEN in .env when ready.')
        print()
        return

    # Write token in-place (replace the placeholder line)
    lines = _read_env_lines(_ENV)
    new_lines = []
    replaced  = False
    for line in lines:
        m = re.match(r'^(AUDIO_UTILS_HF_TOKEN)=', line.strip())
        if m:
            new_lines.append(f'AUDIO_UTILS_HF_TOKEN={token}\n')
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        # Key not in file yet, append it
        if new_lines and new_lines[-1].strip():
            new_lines.append('\n')
        new_lines.append(f'AUDIO_UTILS_HF_TOKEN={token}\n')

    _write_env(_ENV, new_lines)
    print('  AUDIO_UTILS_HF_TOKEN saved.')
    print()


def run(dry_run: bool = False, force: bool = False):
    print('  Detecting hardware…')
    h = detect_hardware()

    gpu_label = f'{h.gpu_name} ({h.gpu_vram_mb} MB)' if h.gpu_name else 'none detected'
    print(f'  CPU  : {h.cpu_cores} cores  ({platform.processor() or "unknown"})')
    print(f'  RAM  : {h.ram_gb:.1f} GB')
    print(f'  GPU  : {gpu_label}')
    print(f'  WSL  : {"yes" if h.is_wsl else "no"}')
    print()

    if not dry_run:
        _bootstrap_env()
        _prompt_hf_token()

    defaults = _compute_defaults(h)
    existing = _read_env(_ENV)
    lines    = _read_env_lines(_ENV)

    to_set:  dict[str, str] = {}
    to_skip: list[str]      = []

    for key, (value, _comment) in defaults.items():
        if key in existing and not force:
            to_skip.append(key)
        else:
            to_set[key] = value

    if to_skip:
        print(f'  Keeping {len(to_skip)} existing value(s):')
        for k in to_skip:
            print(f'    {k} = {existing[k]}')
        print()

    if not to_set:
        print('  Nothing to change — .env is already up to date.')
        return

    print(f'  {"Would set" if dry_run else "Setting"} {len(to_set)} value(s):')
    for k, v in to_set.items():
        _, comment = defaults[k]
        print(f'    {k} = {v}   # {comment}')
    print()

    if dry_run:
        print('  (dry-run — no file written)')
        return

    machine_str = (
        f'CPU {h.cpu_cores}c  RAM {h.ram_gb:.0f}GB'
        + (f'  GPU {h.gpu_name} {h.gpu_vram_mb}MB' if h.gpu_name else '')
        + ('  WSL2' if h.is_wsl else '')
    )

    # Update keys that already exist in the file in-place
    updated_keys: set[str] = set()
    new_lines: list[str] = []
    for line in lines:
        m = re.match(r'^([A-Z_][A-Z0-9_]*)=', line.strip())
        if m and m.group(1) in to_set:
            key = m.group(1)
            _, comment = defaults[key]
            new_lines.append(f'{key}={to_set[key]}   # {comment}\n')
            updated_keys.add(key)
        else:
            new_lines.append(line)

    # Append keys not yet in the file
    remaining = {k: v for k, v in to_set.items() if k not in updated_keys}
    if remaining:
        if new_lines and new_lines[-1].strip():
            new_lines.append('\n')
        new_lines.append(_SECTION_HEADER.format(machine=machine_str))
        for key, value in remaining.items():
            _, comment = defaults[key]
            new_lines.append(f'# {comment}\n')
            new_lines.append(f'{key}={value}\n')
        new_lines.append('\n')

    _write_env(_ENV, new_lines)
    print(f'  Written → {_ENV}')


def main():
    parser = argparse.ArgumentParser(description='Detect hardware and configure .env')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would change without writing')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite auto-detected values even if already set')
    args = parser.parse_args()
    run(dry_run=args.dry_run, force=args.force)


if __name__ == '__main__':
    main()
