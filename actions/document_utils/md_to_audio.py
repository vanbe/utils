#!/usr/bin/env python3
"""
md_to_audio.py — Convert a Markdown file to a spoken-word audio file (MP3).

Engines:
  kokoro  (default) — near-realtime on CPU, high quality, ~82M params
  coqui             — XTTS v2, slower on CPU but very natural multilingual voice

Usage:
  python md_to_audio.py input.md
  python md_to_audio.py input.md -o output.mp3
  python md_to_audio.py input.md --lang fr
  python md_to_audio.py input.md --lang fr --voice ff_siwis
  python md_to_audio.py input.md --engine coqui --lang fr
  python md_to_audio.py input.md --list-voices
"""

import sys
import os
import re
import argparse
import tempfile
import warnings

# Suppress noisy internal warnings from PyTorch / Kokoro before any imports
warnings.filterwarnings('ignore', message='.*dropout option adds dropout.*')
warnings.filterwarnings('ignore', message='.*weight_norm.*deprecated.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

_PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_VENDOR_MODELS = os.path.join(_PROJECT_ROOT, 'vendor', 'models')

# Redirect HuggingFace Hub cache (Kokoro, pyannote, etc.) to vendor/models/huggingface
os.environ.setdefault('HF_HOME', os.path.join(_VENDOR_MODELS, 'huggingface'))

# ---------------------------------------------------------------------------
# Voice / language catalogue
# ---------------------------------------------------------------------------

# Kokoro single-letter lang codes, with ISO aliases for convenience
_LANG_ALIASES = {
    'en': 'a', 'en-us': 'a', 'en-gb': 'b',
    'fr': 'f', 'fr-fr': 'f',
    'es': 'e', 'es-es': 'e',
    'pt': 'p', 'pt-br': 'p',
    'it': 'i', 'it-it': 'i',
    'ja': 'j', 'jp': 'j',
    'zh': 'z', 'cn': 'z',
    'hi': 'h',
}

# (voice_id, gender, description)
_KOKORO_VOICES: dict[str, list[tuple[str, str, str]]] = {
    'a': [  # American English
        ('af_heart',   'F', 'Warm, expressive — default'),
        ('af_bella',   'F', 'Soft and clear'),
        ('af_nicole',  'F', 'Breathy, intimate'),
        ('af_aoede',   'F', 'Bright and articulate'),
        ('af_alloy',   'F', 'Smooth, balanced'),
        ('af_jessica', 'F', 'Friendly, warm'),
        ('af_kore',    'F', 'Confident, professional'),
        ('af_nova',    'F', 'Crisp, modern'),
        ('af_river',   'F', 'Flowing, natural'),
        ('af_sarah',   'F', 'Natural, conversational'),
        ('af_sky',     'F', 'Young, energetic'),
        ('am_adam',    'M', 'Deep and steady'),
        ('am_echo',    'M', 'Clear, neutral'),
        ('am_eric',    'M', 'Warm, friendly'),
        ('am_fenrir',  'M', 'Strong, authoritative'),
        ('am_liam',    'M', 'Calm, measured'),
        ('am_michael', 'M', 'Rich, full voice'),
        ('am_onyx',    'M', 'Deep, resonant'),
        ('am_puck',    'M', 'Playful, light'),
        ('am_santa',   'M', 'Jolly, warm'),
    ],
    'b': [  # British English
        ('bf_alice',    'F', 'Elegant, crisp'),
        ('bf_emma',     'F', 'Warm, articulate — default'),
        ('bf_isabella', 'F', 'Refined, expressive'),
        ('bf_lily',     'F', 'Soft, gentle'),
        ('bm_daniel',   'M', 'Clear, professional'),
        ('bm_fable',    'M', 'Storytelling tone'),
        ('bm_george',   'M', 'Authoritative, rich'),
        ('bm_lewis',    'M', 'Friendly, modern'),
    ],
    'f': [  # French  — only ff_siwis exists in Kokoro-82M
        ('ff_siwis', 'F', 'Natural French — only available voice'),
    ],
    'e': [  # Spanish
        ('ef_dora',  'F', 'Warm Spanish female — default'),
        ('em_alex',  'M', 'Clear Spanish male'),
        ('em_santa', 'M', 'Expressive Spanish male'),
    ],
    'p': [  # Brazilian Portuguese
        ('pf_dora',  'F', 'Natural BP female — default'),
        ('pm_alex',  'M', 'Clear BP male'),
        ('pm_santa', 'M', 'Expressive BP male'),
    ],
    'i': [  # Italian
        ('if_sara',   'F', 'Warm Italian female — default'),
        ('im_nicola', 'M', 'Smooth Italian male'),
    ],
    'j': [  # Japanese
        ('jf_alpha',      'F', 'Clear female — default'),
        ('jf_gongitsune', 'F', 'Soft, gentle'),
        ('jf_nezumi',     'F', 'Light, nimble'),
        ('jf_tebukuro',   'F', 'Warm, cozy'),
        ('jm_kumo',       'M', 'Calm male'),
    ],
    'z': [  # Mandarin Chinese
        ('zf_xiaobei',  'F', 'Warm female — default'),
        ('zf_xiaoni',   'F', 'Gentle female'),
        ('zf_xiaoxiao', 'F', 'Lively female'),
        ('zf_xiaoyi',   'F', 'Bright female'),
        ('zm_yunjian',  'M', 'Clear male'),
        ('zm_yunxi',    'M', 'Natural male'),
        ('zm_yunxia',   'M', 'Deep male'),
        ('zm_yunyang',  'M', 'Warm male'),
    ],
    'h': [  # Hindi
        ('hf_alpha', 'F', 'Natural female — default'),
        ('hf_beta',  'F', 'Soft female'),
        ('hm_omega', 'M', 'Deep male'),
        ('hm_psi',   'M', 'Clear male'),
    ],
}

# Default voice per lang code
_DEFAULT_VOICE: dict[str, str] = {
    lang: voices[0][0] for lang, voices in _KOKORO_VOICES.items()
}

_LANG_NAMES = {
    'a': 'American English', 'b': 'British English', 'f': 'French',
    'e': 'Spanish', 'p': 'Brazilian Portuguese', 'i': 'Italian',
    'j': 'Japanese', 'z': 'Mandarin Chinese', 'h': 'Hindi',
}


def _resolve_lang(lang_arg: str) -> str:
    """Normalize lang to a Kokoro lang code. 'fr' → 'f', 'en' → 'a', etc."""
    lower = lang_arg.lower()
    if lower in _LANG_ALIASES:
        return _LANG_ALIASES[lower]
    if lower in _KOKORO_VOICES:
        return lower
    return lang_arg  # pass through and let Kokoro raise if invalid


def _print_voices():
    print("Available Kokoro voices by language:\n")
    for code, voices in _KOKORO_VOICES.items():
        lang_name = _LANG_NAMES.get(code, code)
        print(f"  --lang {code}  ({lang_name})")
        for vid, gender, desc in voices:
            print(f"    --voice {vid:<16} [{gender}]  {desc}")
        print()

# ---------------------------------------------------------------------------
# Markdown → clean text with pause annotations
# ---------------------------------------------------------------------------

# Pause durations in milliseconds
_PAUSE_H1    = 900
_PAUSE_H2    = 700
_PAUSE_H3    = 500
_PAUSE_PARA  = 400
_PAUSE_LIST  = 150

def _md_to_segments(md_text: str) -> list[tuple[str, int]]:
    """
    Parse markdown and return a list of (text, pause_after_ms) tuples.
    Tables, code blocks, and raw HTML are skipped.
    """
    segments: list[tuple[str, int]] = []

    # Remove code fences entirely
    md_text = re.sub(r'```[\s\S]*?```', '', md_text)
    md_text = re.sub(r'`[^`]+`', '', md_text)  # inline code

    # Remove HTML tags
    md_text = re.sub(r'<[^>]+>', '', md_text)

    lines = md_text.splitlines()
    in_table = False

    for line in lines:
        stripped = line.strip()

        # Skip table rows and separator lines
        if stripped.startswith('|'):
            in_table = True
            continue
        if in_table and re.match(r'^[\s|:-]+$', stripped):
            continue
        in_table = False

        # Skip blank lines (they contribute pauses via the surrounding context)
        if not stripped:
            continue

        # ATX headings
        m = re.match(r'^(#{1,6})\s+(.*)', stripped)
        if m:
            level = len(m.group(1))
            text = _clean_inline(m.group(2))
            if text:
                pause = _PAUSE_H1 if level == 1 else _PAUSE_H2 if level == 2 else _PAUSE_H3
                segments.append((text, pause))
            continue

        # Horizontal rule — just a pause
        if re.match(r'^[-*_]{3,}$', stripped):
            if segments:
                prev_text, prev_pause = segments[-1]
                segments[-1] = (prev_text, max(prev_pause, _PAUSE_H2))
            continue

        # List items
        m = re.match(r'^[-*+]\s+(.*)', stripped)
        if m:
            text = _clean_inline(m.group(1))
            if text:
                segments.append((text, _PAUSE_LIST))
            continue

        m = re.match(r'^\d+[.)]\s+(.*)', stripped)
        if m:
            text = _clean_inline(m.group(1))
            if text:
                segments.append((text, _PAUSE_LIST))
            continue

        # Normal paragraph text
        text = _clean_inline(stripped)
        if text:
            segments.append((text, _PAUSE_PARA))

    return segments


def _clean_inline(text: str) -> str:
    """Strip inline markdown formatting, keeping readable text."""
    # Bold / italic combinations
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
    text = re.sub(r'___(.+?)___', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'~~(.+?)~~', r'\1', text)

    # Links: keep display text, drop URL
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    # Reference-style images/links
    text = re.sub(r'!\[([^\]]*)\]\[[^\]]*\]', '', text)
    text = re.sub(r'\[([^\]]+)\]\[[^\]]*\]', r'\1', text)

    # LaTeX-style math (dollar signs) — just remove
    text = re.sub(r'\$[^$]+\$', '', text)

    # Remaining special chars
    text = text.replace('\\', '')

    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """
    Split a paragraph into sentences for chunked synthesis.
    Keeps chunks at a comfortable length for TTS.
    """
    # Simple sentence splitter that respects abbreviations somewhat
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text)
    out = []
    for part in parts:
        part = part.strip()
        if part:
            out.append(part)
    return out if out else [text]


# ---------------------------------------------------------------------------
# TTS engines
# ---------------------------------------------------------------------------

def _load_hf_token():
    """Inject HUGGING_FACE_HUB_TOKEN from .env if not already set in environment."""
    if os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN'):
        return
    env_file = os.path.join(_PROJECT_ROOT, '.env')
    if not os.path.isfile(env_file):
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('AUDIO_UTILS_HF_TOKEN='):
                token = line.split('=', 1)[1].strip().strip('"\'')
                if token and token != 'your_hugging_face_token_here':
                    os.environ['HF_TOKEN'] = token
                    os.environ['HUGGING_FACE_HUB_TOKEN'] = token
                return


def _synthesize_kokoro(segments: list[tuple[str, int]], lang: str, voice: str,
                       speed: float, out_path: str):
    try:
        from kokoro import KPipeline
    except ImportError:
        print("Kokoro not installed. Run: pip install kokoro soundfile", file=sys.stderr)
        sys.exit(1)

    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        print("soundfile / numpy not installed. Run: pip install soundfile numpy", file=sys.stderr)
        sys.exit(1)

    # Load HF token from .env to suppress unauthenticated-request warning
    _load_hf_token()

    lang = _resolve_lang(lang)
    if not voice:
        voice = _DEFAULT_VOICE.get(lang, 'af_heart')
    lang_label = _LANG_NAMES.get(lang, lang)
    print(f"Loading Kokoro pipeline — lang: {lang_label}, voice: {voice}…")
    pipeline = KPipeline(lang_code=lang, repo_id='hexgrad/Kokoro-82M')

    sample_rate = 24000
    all_audio: list[np.ndarray] = []

    for idx, (text, pause_ms) in enumerate(segments):
        sentences = _split_sentences(text)
        print(f"  [{idx+1}/{len(segments)}] {text[:60]}{'…' if len(text)>60 else ''}")
        for sentence in sentences:
            generator = pipeline(sentence, voice=voice, speed=speed)
            for _, _, audio in generator:
                all_audio.append(audio)

        # Silence after segment
        silence_samples = int(sample_rate * pause_ms / 1000)
        all_audio.append(np.zeros(silence_samples, dtype=np.float32))

    combined = np.concatenate(all_audio)
    _save_mp3(combined, sample_rate, out_path)


def _synthesize_coqui(segments: list[tuple[str, int]], lang: str, voice: str,
                      speed: float, out_path: str):
    try:
        from TTS.api import TTS
    except ImportError:
        print("Coqui TTS not installed. Run: pip install TTS", file=sys.stderr)
        sys.exit(1)

    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        print("soundfile / numpy not installed. Run: pip install soundfile numpy", file=sys.stderr)
        sys.exit(1)

    models_dir = os.path.join(_VENDOR_MODELS, 'coqui')
    os.makedirs(models_dir, exist_ok=True)

    print("Loading Coqui XTTS v2 (first run downloads ~1.8 GB)…")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    sample_rate = 24000
    all_audio: list[np.ndarray] = []

    with tempfile.TemporaryDirectory() as tmp:
        for idx, (text, pause_ms) in enumerate(segments):
            sentences = _split_sentences(text)
            print(f"  [{idx+1}/{len(segments)}] {text[:60]}{'…' if len(text)>60 else ''}")
            for i, sentence in enumerate(sentences):
                chunk_path = os.path.join(tmp, f'seg_{idx:04d}_{i:03d}.wav')
                tts.tts_to_file(
                    text=sentence,
                    language=lang,
                    speaker=voice or "Ana Florence",
                    file_path=chunk_path,
                )
                data, sr = sf.read(chunk_path)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                all_audio.append(data.astype(np.float32))
                sample_rate = sr

            silence_samples = int(sample_rate * pause_ms / 1000)
            all_audio.append(np.zeros(silence_samples, dtype=np.float32))

    combined = np.concatenate(all_audio)
    _save_mp3(combined, sample_rate, out_path)


def _save_mp3(audio_np, sample_rate: int, out_path: str):
    """Save float32 numpy audio as MP3 via pydub + lame."""
    try:
        from pydub import AudioSegment
        import numpy as np
    except ImportError:
        print("pydub not installed. Run: pip install pydub", file=sys.stderr)
        sys.exit(1)

    # Convert float32 [-1,1] → int16
    import numpy as np
    audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    seg = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    seg.export(out_path, format='mp3', bitrate='128k')
    print(f"\nSaved: {out_path}  ({len(seg)/1000:.1f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Convert a Markdown file to spoken-word audio (MP3).'
    )
    parser.add_argument('md_file', nargs='?', help='Input Markdown file')
    parser.add_argument('-o', '--output', help='Output MP3 path (default: same dir as input)')
    parser.add_argument(
        '--engine', choices=['kokoro', 'coqui'], default='kokoro',
        help='TTS engine: kokoro (fast, CPU-friendly) or coqui (XTTS v2, higher quality)'
    )
    parser.add_argument(
        '--lang', default='a',
        help=(
            'Language. Accepts ISO codes (fr, en, es, pt, it, ja, zh, hi, ko) '
            'or Kokoro codes (a=American EN, b=British EN, f=French, e=Spanish, '
            'p=Portuguese, i=Italian, j=Japanese, z=Chinese, h=Hindi, ko=Korean).'
        )
    )
    parser.add_argument(
        '--voice', default='',
        help='Voice name (default: best voice for the chosen language). '
             'Use --list-voices to see all options.'
    )
    parser.add_argument(
        '--speed', type=float, default=1.0,
        help='Speaking speed multiplier (default 1.0). Try 0.9 for more natural pacing.'
    )
    parser.add_argument(
        '--list-voices', action='store_true',
        help='Print all available Kokoro voices and exit.'
    )
    args = parser.parse_args()

    if args.list_voices:
        _print_voices()
        sys.exit(0)

    if not args.md_file:
        parser.print_help()
        sys.exit(1)

    md_file = os.path.abspath(args.md_file)
    if not os.path.isfile(md_file):
        print(f"File not found: {md_file}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out_path = os.path.abspath(args.output)
    else:
        base = os.path.splitext(md_file)[0]
        out_path = base + '.mp3'

    print(f"Parsing {os.path.basename(md_file)}…")
    with open(md_file, encoding='utf-8') as f:
        md_text = f.read()

    segments = _md_to_segments(md_text)
    print(f"  → {len(segments)} text segments extracted")

    if args.engine == 'coqui':
        _synthesize_coqui(segments, lang=args.lang, voice=args.voice,
                          speed=args.speed, out_path=out_path)
    else:
        _synthesize_kokoro(segments, lang=args.lang, voice=args.voice,
                           speed=args.speed, out_path=out_path)


if __name__ == '__main__':
    main()
