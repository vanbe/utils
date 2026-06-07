#!/usr/bin/env python3
"""
utils_tools.py — Interactive TUI for video, audio, document and image tools.

Navigation: ↑↓ arrows, Enter to select, Esc/← to go back, Q to quit.

Usage (WSL):     python utils_tools.py [--workdir /path/to/files]
Usage (Windows): utils_tools           (via install/windows/utils_tools.bat in PATH)
"""

import os
import sys
import subprocess
import argparse
import tty
import termios
import select as _sel_mod
import threading
import time
import glob
from datetime import datetime

# Source de vérité des scripts d'opérations dossier (chemins partagés avec la CLI
# utils_run.py — évite toute dérive de chemin entre le TUI et la CLI).
from utils_run import _FOLDER_REGISTRY

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PYTHON      = os.path.join(_SCRIPT_DIR, '.venv', 'bin', 'python3')
_ACTIONS     = os.path.join(_SCRIPT_DIR, 'actions')
_AUDIO_UTILS = os.path.join(_ACTIONS, 'audio_utils')
_VIDEO_UTILS = os.path.join(_ACTIONS, 'video_utils')
_DOC_UTILS   = os.path.join(_ACTIONS, 'document_utils')
_PIC_UTILS   = os.path.join(_ACTIONS, 'picture_utils')
_AI_UTILS    = os.path.join(_ACTIONS, 'ai_utils')
_DEV_UTILS   = os.path.join(_ACTIONS, 'dev_utils')

# Moteur de capture audio multi-sources (actions/audio_utils/recorder.py).
sys.path.insert(0, _AUDIO_UTILS)
import recorder  # noqa: E402
from whisper_common import recommended_model  # noqa: E402  (modèle conseillé machine)

# ---------------------------------------------------------------------------
# ANSI
# ---------------------------------------------------------------------------

_R   = '\033[0m'
_B   = '\033[1m'
_CYN = '\033[36m'
_GRN = '\033[32m'
_YEL = '\033[33m'
_RED = '\033[31m'
_BLU = '\033[34m'
_MGT = '\033[35m'
_WHT = '\033[97m'
_GRY = '\033[90m'
_HIDE = '\033[?25l'
_SHOW = '\033[?25h'
_EOL  = '\033[K'        # erase to end of line

def bold(s): return f'{_B}{s}{_R}'
def dim(s):  return f'{_GRY}{s}{_R}'
def hi(s):   return f'{_CYN}{s}{_R}'
def ok(s):   return f'{_GRN}{s}{_R}'
def warn(s): return f'{_YEL}{s}{_R}'
def err(s):  return f'{_RED}{s}{_R}'

def _bar():
    w = min(_term_width() - 4, 58)
    return f'{_BLU}{"─" * w}{_R}'

def _term_width():
    try:    return os.get_terminal_size().columns
    except: return 80

# ---------------------------------------------------------------------------
# Raw keyboard input
# ---------------------------------------------------------------------------

def _read_key(fd: int) -> str:
    """Read one keypress from an already-raw fd.
    Never toggles terminal mode — caller must hold raw mode for the session."""
    ch = os.read(fd, 1)
    if ch == b'\x1b':
        # Check if more bytes are waiting (escape sequence vs bare Esc)
        ready, _, _ = _sel_mod.select([sys.stdin], [], [], 0.05)
        if ready:
            ch2 = os.read(fd, 1)
            if ch2 == b'[':
                ch3 = os.read(fd, 1)
                if ch3 == b'A': return 'up'
                if ch3 == b'B': return 'down'
                if ch3 == b'C': return 'right'
                if ch3 == b'D': return 'left'
                # consume any remaining bytes of unknown sequence
                while _sel_mod.select([sys.stdin], [], [], 0)[0]:
                    os.read(fd, 1)
        return 'esc'
    if ch in (b'\r', b'\n'): return 'enter'
    if ch == b'\x03':        raise KeyboardInterrupt
    if ch == b'\x7f':        return 'backspace'
    try:
        c = ch.decode('utf-8')
        if c.upper() == 'Q': return 'quit'
        return f'char:{c}'
    except Exception:
        return 'unknown'

# ---------------------------------------------------------------------------
# select_menu — core arrow-key widget
# ---------------------------------------------------------------------------

def select_menu(
    items: list,       # str  or  (label, hint)  tuples; label may contain ANSI codes
    title:    str = '',
    subtitle: str = '',
    headers:  set = None,   # indices that are non-selectable group headers
) -> int | None:
    """
    Renders an arrow-key navigable list in-place.
    Returns selected index, or None on Esc/back.
    Raises SystemExit(0) on Q.

    Selected item is marked with a cyan ▶ prefix.
    Non-selected items are indented.
    """
    n = len(items)
    if n == 0:
        return None

    hdrs = headers or set()

    # Start on first selectable item
    idx = 0
    while idx in hdrs and idx < n - 1:
        idx += 1

    def _skip(i, delta):
        j, steps = (i + delta) % n, 0
        while j in hdrs and steps < n:
            j = (j + delta) % n
            steps += 1
        return j

    def _make_lines():
        out = []
        out.append('')
        if title:
            out.append(f'  {bold(title)}')
        if subtitle:
            out.append(f'  {dim(subtitle)}')
        out.append(f'  {_bar()}')
        out.append('')
        for i, item in enumerate(items):
            label = item[0] if isinstance(item, (list, tuple)) else item
            if i in hdrs:
                out.append(f'  {label}')
            else:
                hint = (item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else '') or ''
                if i == idx:
                    row = f'  {_CYN}▶{_R} {label}'
                else:
                    row = f'    {label}'
                if hint:
                    row += f'  {_GRY}{hint}{_R}'
                out.append(row)
        out.append('')
        out.append(f'  {_bar()}')
        out.append(f'  {_GRY}↑↓ navigate · Enter select · Esc/← back · Q quit{_R}')
        out.append('')
        return out

    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    sys.stdout.write(_HIDE)
    sys.stdout.flush()

    lines = _make_lines()
    for line in lines:
        sys.stdout.write('\r' + line + _EOL + '\n')
    sys.stdout.flush()
    count = len(lines)

    result = None
    try:
        tty.setraw(fd)          # raw mode ON — held for the whole menu
        while True:
            key = _read_key(fd)
            if key == 'up':
                idx = _skip(idx, -1)
            elif key == 'down':
                idx = _skip(idx, 1)
            elif key == 'enter':
                result = idx
                break
            elif key in ('esc', 'left'):
                result = None
                break
            elif key == 'quit':
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                _clear_block(count)
                sys.stdout.write(_SHOW)
                sys.stdout.flush()
                sys.exit(0)
            else:
                continue

            # Redraw in-place: \033[{N}F = move up N lines AND go to column 0
            new_lines = _make_lines()
            sys.stdout.write(f'\033[{count}F')
            for line in new_lines:
                sys.stdout.write('\r' + line + _EOL + '\n')
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)   # raw mode OFF
        _clear_block(count)
        sys.stdout.write(_SHOW)
        sys.stdout.flush()

    return result


def _clear_block(count: int):
    sys.stdout.write(f'\033[{count}F')   # up N lines, column 0
    for _ in range(count):
        sys.stdout.write('\r' + _EOL + '\n')
    sys.stdout.write(f'\033[{count}F')   # back to top of cleared block
    sys.stdout.flush()


def multiselect_menu(
    items: list,            # str or (label, hint) ; label may contain ANSI codes
    title:    str = '',
    subtitle: str = '',
    headers:  set = None,   # indices that are non-selectable group headers
    preselected: set = None,
) -> list | None:
    """
    Arrow-key list where Space toggles a checkbox on each selectable row.
    Returns the list of checked indices (Enter), or None on Esc/back.
    Raises SystemExit(0) on Q. Mirrors select_menu's rendering/raw-mode handling.
    """
    n = len(items)
    if n == 0:
        return None

    hdrs = headers or set()
    checked = set(preselected or set())

    idx = 0
    while idx in hdrs and idx < n - 1:
        idx += 1

    def _skip(i, delta):
        j, steps = (i + delta) % n, 0
        while j in hdrs and steps < n:
            j = (j + delta) % n
            steps += 1
        return j

    def _make_lines():
        out = ['']
        if title:    out.append(f'  {bold(title)}')
        if subtitle: out.append(f'  {dim(subtitle)}')
        out.append(f'  {_bar()}')
        out.append('')
        for i, item in enumerate(items):
            label = item[0] if isinstance(item, (list, tuple)) else item
            if i in hdrs:
                out.append(f'  {label}')
                continue
            hint = (item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else '') or ''
            box = f'{_GRN}[x]{_R}' if i in checked else f'{_GRY}[ ]{_R}'
            cursor = f'{_CYN}▶{_R}' if i == idx else ' '
            row = f'  {cursor} {box} {label}'
            if hint:
                row += f'  {_GRY}{hint}{_R}'
            out.append(row)
        out.append('')
        out.append(f'  {_bar()}')
        out.append(f'  {_GRY}↑↓ navigate · Space toggle · Enter confirm · Esc/← back · Q quit{_R}')
        out.append('')
        return out

    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    sys.stdout.write(_HIDE)
    sys.stdout.flush()

    lines = _make_lines()
    for line in lines:
        sys.stdout.write('\r' + line + _EOL + '\n')
    sys.stdout.flush()
    count = len(lines)

    result = None
    try:
        tty.setraw(fd)
        while True:
            key = _read_key(fd)
            if key == 'up':
                idx = _skip(idx, -1)
            elif key == 'down':
                idx = _skip(idx, 1)
            elif key == 'char: ':            # space toggles
                if idx not in hdrs:
                    checked.symmetric_difference_update({idx})
            elif key == 'enter':
                result = sorted(checked)
                break
            elif key in ('esc', 'left'):
                result = None
                break
            elif key == 'quit':
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                _clear_block(count)
                sys.stdout.write(_SHOW)
                sys.stdout.flush()
                sys.exit(0)
            else:
                continue

            new_lines = _make_lines()
            sys.stdout.write(f'\033[{count}F')
            for line in new_lines:
                sys.stdout.write('\r' + line + _EOL + '\n')
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        _clear_block(count)
        sys.stdout.write(_SHOW)
        sys.stdout.flush()

    return result

# ---------------------------------------------------------------------------
# Text input helpers
# ---------------------------------------------------------------------------

def ask(prompt: str, default: str = '') -> str:
    hint = f' [{dim(default)}]' if default else ''
    try:
        val = input(f'  {warn("›")} {prompt}{hint}: ').strip()
    except EOFError:
        return default
    return val if val else default


def confirm(details: list[tuple[str, str]]) -> bool:
    """Show a summary table, ask Y/n. Returns True to proceed."""
    print()
    print(f'  {_bar()}')
    for label, value in details:
        print(f'  {dim(label + ":"): <24} {hi(value)}')
    print(f'  {_bar()}')
    print()
    ans = ask('Run?', 'Y').upper()
    return ans in ('Y', 'YES', '')


def _show_result(rc: int, out_path: str = ''):
    print()
    if rc == 0:
        msg = ok('  ✓ Done.')
        if out_path:
            msg += f'  {dim("→")} {hi(os.path.basename(out_path))}'
        print(msg)
    else:
        print(err(f'  ✗ Failed (exit {rc})'))


def pause():
    print()
    try:
        input(dim('  Press Enter to continue… '))
    except EOFError:
        pass

# ---------------------------------------------------------------------------
# File type helpers
# ---------------------------------------------------------------------------

VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.m4v', '.webm'}
AUDIO_EXTS = {'.mp3', '.aac', '.m4a', '.flac', '.wav', '.ogg', '.opus', '.mka', '.ac3', '.eac3'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.heic'}
RAW_EXTS   = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', '.raw'}
DOC_EXTS   = {'.docx', '.doc'}
ODT_EXTS   = {'.odt'}
SHEET_EXTS = {'.xlsx', '.xls'}
PPT_EXTS   = {'.pptx', '.ppt'}
PDF_EXT    = {'.pdf'}
MD_EXT     = {'.md'}


def _tag(ext: str) -> str:
    if ext in VIDEO_EXTS:  return f'{_MGT}[VID]{_R}'
    if ext in AUDIO_EXTS:  return f'{_CYN}[AUD]{_R}'
    if ext in IMAGE_EXTS:  return f'{_GRN}[IMG]{_R}'
    if ext in RAW_EXTS:    return f'{_GRN}[RAW]{_R}'
    if ext in DOC_EXTS:    return f'{_YEL}[DOC]{_R}'
    if ext in ODT_EXTS:    return f'{_YEL}[ODT]{_R}'
    if ext in SHEET_EXTS:  return f'{_GRN}[XLS]{_R}'
    if ext in PPT_EXTS:    return f'{_YEL}[PPT]{_R}'
    if ext in PDF_EXT:     return f'{_RED}[PDF]{_R}'
    if ext in MD_EXT:      return f'{_BLU}[MD ]{_R}'
    return f'{_GRY}[   ]{_R}'


def _size(path: str) -> str:
    try:
        b = os.path.getsize(path)
        if b < 1 << 10: return f'{b} B'
        if b < 1 << 20: return f'{b >> 10} KB'
        if b < 1 << 30: return f'{b >> 20} MB'
        return f'{b >> 30} GB'
    except:
        return ''


def _count(path: str) -> str:
    try:
        n = len(os.listdir(path))
        return f'{n} item{"s" if n != 1 else ""}'
    except:
        return ''

# ---------------------------------------------------------------------------
# File browser
# ---------------------------------------------------------------------------

_TYPE_UP   = 'up'
_TYPE_DIR  = 'dir'
_TYPE_FILE = 'file'
_TYPE_THIS = 'this'


def _browser_entries(directory: str, *, include_this: bool = True,
                     file_exts: set | None = None) -> list[dict]:
    entries = []

    # Go up
    parent = os.path.dirname(directory)
    if parent != directory:
        entries.append({
            'type':  _TYPE_UP,
            'path':  parent,
            'label': f'{_GRY}[../]  go up{_R}',
            'hint':  os.path.basename(parent) or '/',
        })

    if include_this:
        entries.append({
            'type':  _TYPE_THIS,
            'path':  directory,
            'label': f'{_GRY}[./]   actions for this folder{_R}',
            'hint':  '',
        })

    names = None
    for attempt in range(4):
        try:
            names = sorted(os.listdir(directory), key=lambda x: x.lower())
            break
        except OSError:
            if attempt < 3:
                time.sleep(0.4)
    if names is None:
        entries.append({
            'type':  _TYPE_FILE,
            'path':  directory,
            'label': f'  {_YEL}⚠  directory listing failed (Nextcloud sync?){_R}',
            'hint':  'navigate away and back to retry',
        })
        return entries

    # Directories first
    for name in names:
        full = os.path.join(directory, name)
        if os.path.isdir(full):
            entries.append({
                'type':  _TYPE_DIR,
                'path':  full,
                'label': f'{_BLU}[DIR]  {name}/{_R}',
                'hint':  _count(full),
            })

    # Files
    for name in names:
        full = os.path.join(directory, name)
        if os.path.isfile(full):
            ext = os.path.splitext(name)[1].lower()
            if file_exts is not None and ext not in file_exts:
                continue
            entries.append({
                'type':  _TYPE_FILE,
                'path':  full,
                'label': f'{_tag(ext)}  {name}',
                'hint':  _size(full),
            })

    return entries


def browse(start_dir: str) -> tuple[str, str] | None:
    """
    Interactive file browser — main TUI flow.
    Returns (selected_path, current_dir), or None if user quits.
    ↑↓ navigate · Enter open dir or select file · Esc/← go up · Q quit
    """
    current = start_dir

    while True:
        if not os.path.isdir(current):
            parent = os.path.dirname(current)
            current = parent if parent != current else os.path.expanduser('~')
            continue

        entries = _browser_entries(current)
        items   = [(e['label'], e['hint']) for e in entries]

        idx = select_menu(items, title='Utils Tools', subtitle=current)

        if idx is None:
            parent = os.path.dirname(current)
            if parent == current:
                return None
            current = parent
            continue

        chosen = entries[idx]

        if chosen['type'] in (_TYPE_UP, _TYPE_DIR):
            current = chosen['path']
        else:  # _TYPE_FILE or _TYPE_THIS
            return chosen['path'], current


def pick_file(start_dir: str, title: str,
              file_exts: set | None = None) -> str | None:
    """
    Interactive single-file picker. Same navigation as `browse()` but only
    files are selectable, optionally filtered to `file_exts`.
    Returns the chosen path, or None if the user backs out.
    """
    current = start_dir

    while True:
        if not os.path.isdir(current):
            parent = os.path.dirname(current)
            current = parent if parent != current else os.path.expanduser('~')
            continue

        entries = _browser_entries(current, include_this=False, file_exts=file_exts)
        items   = [(e['label'], e['hint']) for e in entries]

        idx = select_menu(items, title=title, subtitle=current)

        if idx is None:
            parent = os.path.dirname(current)
            if parent == current:
                return None
            current = parent
            continue

        chosen = entries[idx]

        if chosen['type'] in (_TYPE_UP, _TYPE_DIR):
            current = chosen['path']
        elif chosen['type'] == _TYPE_FILE:
            return chosen['path']

# ---------------------------------------------------------------------------
# Script runners
# ---------------------------------------------------------------------------

def _gpu_env() -> dict:
    """Return os.environ with all nvidia/<pkg>/lib dirs prepended to LD_LIBRARY_PATH.

    glibc caches LD_LIBRARY_PATH at process startup, so the fix must be applied
    before the child Python process starts — not inside it.  This is needed for
    cu13 wheels (torch 2.11+) whose libnvrtc-builtins.so.13.0 lives under
    nvidia/cu13/lib/ which PyTorch's loader doesn't search by default.
    """
    import site as _site
    nv_dirs = sorted(glob.glob(
        os.path.join(_site.getsitepackages()[0], 'nvidia', '*', 'lib')
    ))
    env = os.environ.copy()
    if nv_dirs:
        existing = env.get('LD_LIBRARY_PATH', '')
        env['LD_LIBRARY_PATH'] = ':'.join(nv_dirs + ([existing] if existing else []))
    return env


def _py(*args) -> int:
    return subprocess.run([_PYTHON, *args], env=_gpu_env()).returncode


def _cmd(*args) -> int:
    return subprocess.run(args).returncode

# ---------------------------------------------------------------------------
# Actions — Video
# ---------------------------------------------------------------------------

def _act_transcribe_channels(path: str):
    """Transcription différée *par canal* d'un FLAC + sidecar channels.json
    (Moi / Système connus par construction → pas de diarisation). Sorties
    <base>.srt + <base>.md, identiques au live."""
    name = os.path.basename(path)
    lang_idx = select_menu([
        ('FR — Français', 'default'),
        ('EN — English',  ''),
        ('Auto',          'détection automatique'),
    ], title='Langue')
    if lang_idx is None: return
    lang = ['fr', 'en', 'auto'][lang_idx]

    rec = recommended_model('transcribe')
    _models = ['turbo', 'large', 'medium', 'small', 'base', 'tiny']
    order = [rec] + [m for m in _models if m != rec]
    _desc = {'large': 'meilleure qualité, lent', 'turbo': 'rapide, qualité quasi-large',
             'medium': 'compromis', 'small': 'rapide et léger',
             'base': 'très léger', 'tiny': 'minimal'}
    m_idx = select_menu(
        [(m, ('(recommandé) ' if m == rec else '') + _desc.get(m, '')) for m in order],
        title='Modèle Whisper')
    if m_idx is None: return
    model = order[m_idx]

    base = os.path.splitext(name)[0]
    if not confirm([('File', name), ('Mode', 'par canal (Moi / Système)'),
                    ('Langue', lang), ('Modèle', model),
                    ('Output', f'{base}.srt + {base}.md')]):
        return
    print()
    rc = _py(os.path.join(_AUDIO_UTILS, 'transcribe_channels.py'),
             path, '--language', lang, '--model', model)
    _show_result(rc, os.path.join(os.path.dirname(path), base + '.md'))
    pause()


def act_transcribe(path: str):
    name = os.path.basename(path)
    print(f'\n  {bold("Transcribe")}  {dim(name)}\n')

    # FLAC multicanal issu de « Record audio » → propose la transcription par
    # canal (attribution Moi/Système exacte, sans diarisation).
    if os.path.exists(os.path.splitext(path)[0] + '.channels.json'):
        mode_idx = select_menu([
            ('Par canal — Moi / Système', '(recommandé) via channels.json'),
            ('Standard — diarisation',    'downmix + détection des locuteurs'),
        ], title='Mode de transcription')
        if mode_idx is None: return
        if mode_idx == 0:
            return _act_transcribe_channels(path)

    lang_idx = select_menu([
        ('FR — French',  'default'),
        ('EN — English', ''),
        ('Auto',         'language auto-detection'),
    ], title='Language')
    if lang_idx is None: return
    lang = ['fr', 'en', 'auto'][lang_idx]

    rec = recommended_model('transcribe')   # différé → modèle qualité
    _models = ['turbo', 'large', 'medium', 'small', 'base', 'tiny']
    if rec not in _models:
        _models.insert(0, rec)
    order = [rec] + [m for m in _models if m != rec]
    _desc = {'large': 'meilleure qualité, lent', 'turbo': 'rapide, qualité quasi-large',
             'medium': 'compromis', 'small': 'rapide et léger',
             'base': 'très léger', 'tiny': 'minimal'}
    model_idx = select_menu(
        [(m, ('(recommandé) ' if m == rec else '') + _desc.get(m, '')) for m in order],
        title='Whisper model')
    if model_idx is None: return
    model = order[model_idx]

    fmt_idx = select_menu([
        ('Markdown .md',  'with speaker labels  (default)'),
        ('SRT .srt',      'subtitle format'),
    ], title='Output format')
    if fmt_idx is None: return
    fmt = ['md', 'srt'][fmt_idx]

    spk_idx = select_menu([
        ('Yes — identify speakers', 'default'),
        ('No  — skip (faster)',     ''),
    ], title='Speaker detection')
    if spk_idx is None: return

    max_speakers = 0
    if spk_idx == 0:
        spk_count_idx = select_menu([
            ('Auto',     'let the model decide  (default)'),
            ('2',        'interview / podcast'),
            ('3 – 5',    'small meeting'),
            ('6 – 10',   'larger group'),
            ('Custom',   'enter manually'),
        ], title='Max speakers  (speeds up diarization)')
        if spk_count_idx is None: return
        if spk_count_idx == 1:   max_speakers = 2
        elif spk_count_idx == 2: max_speakers = 5
        elif spk_count_idx == 3: max_speakers = 10
        elif spk_count_idx == 4:
            v = ask('Max number of speakers', '4')
            try: max_speakers = int(v)
            except ValueError: max_speakers = 0

    imp_idx = select_menu([
        ('Yes — pre-process audio', 'default'),
        ('No  — skip (faster)',     ''),
    ], title='Audio improvement')
    if imp_idx is None: return

    base = os.path.splitext(name)[0]
    # transcribe_audio.py écrit toujours « <base>_transcription.<fmt> » — refléter
    # le vrai nom (sinon _show_result annonce un fichier qui n'existe pas).
    out  = os.path.join(os.path.dirname(path), f'{base}_transcription.{fmt}')

    details = [
        ('File',              name),
        ('Language',          lang),
        ('Model',             model),
        ('Output format',     fmt),
        ('Speaker detection', 'no' if spk_idx == 1 else 'yes'),
    ]
    if spk_idx == 0:
        details.append(('Max speakers', str(max_speakers) if max_speakers else 'auto'))
    details += [
        ('Audio improve',     'no' if imp_idx == 1 else 'yes'),
        ('Output',            os.path.basename(out)),
    ]
    if not confirm(details): return

    args = [path, '--language', lang, '--model', model, '--output-format', fmt]
    if spk_idx == 1:    args.append('--no-speaker')
    if imp_idx == 1:    args.append('--no-improve')
    if max_speakers > 0: args += ['--max-speakers', str(max_speakers)]
    print()
    rc = _py(os.path.join(_AUDIO_UTILS, 'transcribe_audio.py'), *args)
    _show_result(rc, out)
    pause()


def act_split(path: str):
    name = os.path.basename(path)
    print(f'\n  {bold("Split video")}  {dim(name)}\n')
    start = ask('Start time', '00:00:00')
    end   = ask('End time (blank = end of video)', '')
    base  = os.path.splitext(name)[0]
    ext   = os.path.splitext(name)[1]
    out   = os.path.join(os.path.dirname(path), f'{base}_split{ext}')

    if not confirm([
        ('File',   name),
        ('Start',  start),
        ('End',    end or '(end of video)'),
        ('Output', os.path.basename(out)),
    ]): return

    args = [path, '--start', start]
    if end: args += ['--end', end]
    print()
    rc = _py(os.path.join(_VIDEO_UTILS, 'video_split.py'), *args)
    _show_result(rc, out)
    pause()


def act_merge_videos(path: str):
    name = os.path.basename(path)
    print(f'\n  {bold("Merge videos")}  {dim(name)}\n')
    print(f'  {dim("First video:")}  {hi(name)}')
    print(f'  {dim("Now pick the second video (will be appended).")}\n')
    print(dim('  Press Enter to continue…'), end='', flush=True)
    try: input()
    except EOFError: print()

    second_path = pick_file(
        os.path.dirname(path),
        title='Pick second video  (appended after the first)',
        file_exts=VIDEO_EXTS,
    )
    if second_path is None:
        return

    if os.path.abspath(second_path) == os.path.abspath(path):
        print(warn('\n  Second video must be different from the first.'))
        pause()
        return

    base = os.path.splitext(name)[0]
    ext  = os.path.splitext(name)[1]
    out  = os.path.join(os.path.dirname(path), f'{base}_merged{ext}')

    if not confirm([
        ('First',  name),
        ('Second', os.path.basename(second_path)),
        ('Output', os.path.basename(out)),
    ]): return

    print()
    rc = _py(os.path.join(_VIDEO_UTILS, 'video_merge.py'),
             path, second_path, '--output', out)
    _show_result(rc, out)
    pause()


def act_extract_audio(path: str):
    name = os.path.basename(path)
    print(f'\n  {bold("Extract audio")}  {dim(name)}\n')
    print(dim('  Detecting codec…'), flush=True)

    r = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-select_streams', 'a:0',
         '-show_entries', 'stream=codec_name',
         '-of', 'default=noprint_wrappers=1:nokey=1', path],
        capture_output=True, text=True)
    codec = r.stdout.strip()
    ext_map = {'aac':'aac','mp3':'mp3','ac3':'ac3','eac3':'eac3','opus':'opus',
               'vorbis':'ogg','flac':'flac','pcm_s16le':'wav','pcm_s24le':'wav'}
    ext = ext_map.get(codec, 'mka')
    print(f'  Codec: {hi(codec)}  →  .{hi(ext)}\n')

    base    = os.path.splitext(name)[0]
    dirpath = os.path.dirname(path)
    raw_out = os.path.join(dirpath, f'{base}_audio_raw.{ext}')
    imp_out = os.path.join(dirpath, f'{base}_audio_improved.flac')

    if not confirm([
        ('File',             name),
        ('Codec',            codec),
        ('Raw output',       os.path.basename(raw_out)),
        ('Improved output',  os.path.basename(imp_out)),
    ]): return

    print()
    print(dim('  [1/2] Extracting raw audio (stream copy)…'))
    rc = _cmd('ffmpeg', '-y', '-i', path, '-vn', '-acodec', 'copy', raw_out)
    if rc != 0:
        print(err(f'  ✗ Raw extraction failed (exit {rc})'))
        pause()
        return
    print(ok(f'  → {os.path.basename(raw_out)}'))
    print()
    print(dim('  [2/2] Normalizing (EBU R128 loudnorm → FLAC)…'))
    rc = _cmd('ffmpeg', '-y', '-i', raw_out,
              '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11,highpass=f=80,lowpass=f=15000',
              imp_out)
    _show_result(rc, imp_out)
    pause()


# ---------------------------------------------------------------------------
# Record audio — multi-source capture (micros + system output) → FLAC multicanal
# ---------------------------------------------------------------------------

def _fmt_dur(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    if h: return f'{h}h {m:02d}m {s:02d}s'
    if m: return f'{m}m {s:02d}s'
    return f'{s}s'


def _mmss(sec: float) -> str:
    sec = int(sec)
    return f'{sec // 60:02d}:{sec % 60:02d}'


_TRANSCRIPT_H = 10   # hauteur fixe de la zone transcript (lignes constantes)


def _fmt_seg(seg, partial: bool = False) -> str:
    width = max(20, _term_width() - 4)
    prefix = f'[{_mmss(seg["start"])}] {seg["label"]}: '
    avail = max(8, width - len(prefix) - 2)
    text = seg['text']
    if len(text) > avail:
        text = text[:avail - 1] + '…'
    if partial:   # aperçu en cours → tout grisé + « … »
        return f'  {_GRY}[{_mmss(seg["start"])}] {seg["label"]}: {text} …{_R}'
    color = _GRN if seg['kind'] == 'input' else _MGT
    return f'  {_GRY}[{_mmss(seg["start"])}]{_R} {color}{seg["label"]}{_R}: {text}'


def _transcript_lines(transcript) -> list:
    """Zone transcript (hauteur fixe). `transcript` = (state, lock) ;
    state = {'final': [segs], 'partial': {source_idx: seg}}."""
    out = [f'  {_bar()}', f'  {bold("Transcript (live)")}  {dim("· … = en cours")}']
    state, lock = transcript
    with lock:
        finals = list(state['final'])
        partials = list(state['partial'].values())
    display = [_fmt_seg(s, False) for s in finals]
    display += [_fmt_seg(s, True)
                for s in sorted(partials, key=lambda s: s.get('source_idx', 0))]
    display = display[-_TRANSCRIPT_H:]
    out += display
    out += [''] * (_TRANSCRIPT_H - len(display))
    return out


def _meter(level: float, width: int = 28) -> str:
    # RMS → sqrt scaling pour rendre les niveaux faibles visibles
    disp = max(0.0, min(1.0, level ** 0.5))
    filled = int(disp * width)
    color = _GRN if disp < 0.7 else (_YEL if disp < 0.9 else _RED)
    return f'{color}{"█" * filled}{_R}{_GRY}{"░" * (width - filled)}{_R}'


def _record_lines(rec, sources, out_path, transcript=None, confirming=False) -> list:
    lines = ['']
    status = warn('⏸ PAUSE') if rec.paused else err('● REC')
    lines.append(f'  {bold("Recording")}   {status}')
    fname = os.path.basename(out_path)
    if len(fname) > 44:
        fname = fname[:43] + '…'
    lines.append(f'  {dim("File:")}  {hi(fname)}')
    lines.append(f'  {dim("Time:")}  {bold(_fmt_dur(rec.elapsed()))}'
                 f'    {dim("Size:")} {_size(out_path) or "—"}'
                 f'    {dim("Format:")} {rec.rate} Hz · {rec.total_channels}ch FLAC')
    lines.append(f'  {_bar()}')
    # En pause, la capture est gelée (SIGSTOP) → pas de nouveaux niveaux : on
    # affiche 0 plutôt que de laisser les barres figées sur la dernière valeur.
    levels = [0.0] * len(sources) if rec.paused else rec.levels()
    for i, s in enumerate(sources):
        lvl = levels[i] if i < len(levels) else 0.0
        tag = f'{_GRN}IN {_R}' if s['kind'] == 'input' else f'{_MGT}OUT{_R}'
        name = s['name'] if len(s['name']) <= 26 else s['name'][:25] + '…'
        lines.append(f'  [{tag}] {name:<26} {_meter(lvl)}')
    if transcript is not None:
        lines += _transcript_lines(transcript)
    lines.append(f'  {_bar()}')
    if confirming:
        lines.append(f'  {err("Supprimer cet enregistrement ?")}  '
                     f'{bold("O")}=oui (supprime)  ·  {bold("N")}=non (continue)')
    else:
        lines.append(f'  {_GRY}Space pause/resume · S/Enter stop & save · '
                     f'Esc/Q annuler{_R}')
    lines.append('')
    return lines


def _record_live(rec, sources, out_path, transcript=None) -> str:
    """Écran live raw-mode. Retourne 'stopped' ou 'cancelled'."""
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    sys.stdout.write(_HIDE)
    sys.stdout.flush()
    count = 0
    result = 'stopped'
    confirming = False        # Esc/Q armé → on demande confirmation avant suppression
    try:
        tty.setraw(fd)
        rec.start()
        first = True
        while True:
            lines = _record_lines(rec, sources, out_path, transcript,
                                  confirming=confirming)
            if not first:
                sys.stdout.write(f'\033[{count}F')
            for line in lines:
                sys.stdout.write('\r' + line + _EOL + '\n')
            sys.stdout.flush()
            count = len(lines)
            first = False

            if not rec.is_alive():           # ffmpeg/capture terminé tout seul
                break

            ready, _, _ = _sel_mod.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            key = _read_key(fd)
            if confirming:
                # Suppression destructive → exige un O/oui explicite ; tout le
                # reste (N, Esc, Espace…) annule la demande et reprend l'enreg.
                if key in ('char:o', 'char:O', 'char:y', 'char:Y'):
                    result = 'cancelled'
                    break
                confirming = False
                continue
            if key == 'enter' or key in ('char:s', 'char:S'):
                break
            if key == 'char: ':
                if rec.paused:
                    rec.resume()
                else:
                    rec.pause()
                    if transcript is not None:   # purge les aperçus « … » figés
                        _st, _lk = transcript
                        with _lk:
                            _st['partial'].clear()
            elif key in ('esc', 'quit'):
                confirming = True       # ne supprime pas encore : demande d'abord
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        _clear_block(count)
        sys.stdout.write(_SHOW)
        sys.stdout.flush()

    if result == 'cancelled':
        rec.cancel()
    else:
        rec.stop()
    return result


def _build_capture_exe() -> bool:
    print(dim('\n  Construction de capture.exe…'))
    okb, log = recorder.build_capture()
    print()
    if okb:
        print(ok('  ✓ capture.exe construit et épinglé (SHA-256).'))
        return True
    print(err('  ✗ Build échoué.'))
    if log:
        print(dim('  ' + log.replace('\n', '\n  ')))
    print(dim('  → installer le toolchain : sudo apt-get install g++-mingw-w64-x86-64'))
    print(dim('    (ou compiler sur Windows : actions/audio_utils/capture/build.bat)'))
    pause()
    return False


def _ensure_capture_exe(backend: str) -> bool:
    """Sur WSL, s'assure que capture.exe existe ET n'a pas été altéré.
    Build à la demande + vérification d'intégrité (épinglage SHA-256) :
    un binaire substitué pourrait exfiltrer l'audio → on le refuse."""
    if backend != 'wsl':
        return True
    if not recorder.capture_exe_present():
        print(warn("  Le capteur Windows (capture.exe) est absent — il n'est pas versionné."))
        idx = select_menu([
            ('Construire maintenant', 'cross-compile via mingw (build.sh)'),
            ('Annuler', ''),
        ], title='capture.exe manquant')
        if idx != 0:
            return False
        return _build_capture_exe()

    status, cur = recorder.verify_capture()
    if status == 'ok':
        return True
    if status == 'mismatch':
        print(err('  ⚠ capture.exe a été MODIFIÉ depuis sa compilation '
                  '(hash ≠ épinglé).'))
        print(dim('    Un binaire substitué pourrait capter/exfiltrer l\'audio.'))
        idx = select_menu([
            ('Reconstruire depuis la source', '(recommandé) build.sh + ré-épinglage'),
            ('Ré-épingler (je fais confiance)', 'accepter ce binaire tel quel'),
            ('Annuler', ''),
        ], title='Intégrité capture.exe')
        if idx == 0:
            return _build_capture_exe()
        if idx == 1:
            recorder.pin_capture()
            print(ok('  ✓ Binaire ré-épinglé.'))
            return True
        return False
    # 'unpinned' : binaire présent mais jamais épinglé (ex. build Windows manuel)
    print(warn('  capture.exe présent mais non épinglé (intégrité non vérifiable).'))
    idx = select_menu([
        ('Épingler maintenant', '(recommandé) fige le hash de confiance'),
        ('Continuer sans épingler', ''),
        ('Annuler', ''),
    ], title='capture.exe non épinglé')
    if idx == 0:
        recorder.pin_capture()
        print(ok('  ✓ capture.exe épinglé (SHA-256).'))
        return True
    return idx == 1


def act_record_audio(dirpath: str):
    print(f'\n  {bold("Record audio")}  {dim(dirpath)}\n')
    backend = recorder.detect_backend()

    if not _ensure_capture_exe(backend):
        return

    sources = recorder.list_sources(backend)
    if not sources:
        hint = ('capture.exe --list vide ?' if backend == 'wsl'
                else 'serveur PulseAudio absent ?')
        print(warn(f'  Aucune source audio détectée ({hint})'))
        pause()
        return

    # Multi-sélection groupée Entrées / Sorties
    inputs  = [s for s in sources if s['kind'] == 'input']
    outputs = [s for s in sources if s['kind'] == 'output']
    items, headers, meta = [], set(), []
    if inputs:
        headers.add(len(items)); items.append((f'{_GRN}ENTRÉES (micros){_R}', '')); meta.append(None)
        for s in inputs:
            items.append((s['name'], f"{s['channels']}ch")); meta.append(s)
    if outputs:
        headers.add(len(items)); items.append((f'{_MGT}SORTIES (système){_R}', '')); meta.append(None)
        for s in outputs:
            items.append((s['name'], f"{s['channels']}ch")); meta.append(s)

    sel = multiselect_menu(items, title='Sources à capturer',
                           subtitle='Espace pour (dé)cocher · Entrée pour valider',
                           headers=headers)
    if not sel:
        return
    chosen = [meta[i] for i in sel if meta[i] is not None]
    if not chosen:
        print(warn('  Aucune source sélectionnée.'))
        pause()
        return

    # Transcription live (optionnelle)
    t_idx = select_menu([
        ('Non', 'enregistrement seul (défaut)'),
        ('Oui', 'transcription live par canal (Moi / Système)'),
    ], title='Transcription live ?')
    if t_idx is None:
        return
    want_trans = (t_idx == 1)
    language, model = 'fr', None
    if want_trans:
        l_idx = select_menu([
            ('FR — Français', 'default'),
            ('EN — English', ''),
            ('Auto', 'détection automatique'),
        ], title='Langue')
        if l_idx is None:
            return
        language = ['fr', 'en', 'auto'][l_idx]
        rec = recommended_model('live')      # live → modèle temps réel
        m_idx = select_menu([
            (f'Recommandé — {rec}', '(recommandé) · live temps réel (.env)'),
            ('turbo', 'rapide, qualité quasi-large'),
            ('medium', 'compromis'),
            ('small', 'léger'),
            ('large', 'meilleure qualité, plus lent'),
        ], title='Modèle Whisper')
        if m_idx is None:
            return
        # None → LiveTranscriber lit LIVE_TRANSCRIBE_MODEL (= le recommandé)
        model = [None, 'turbo', 'medium', 'small', 'large'][m_idx]

    default_base = recorder.default_basename()
    name = ask('Nom du fichier', default_base) or default_base
    if name.lower().endswith('.flac'):
        name = name[:-5]
    out_path = recorder.unique_path(dirpath, name, 'flac')

    details = [
        ('Folder',   dirpath),
        ('Sources',  ', '.join(s['name'] for s in chosen)),
        ('Channels', str(sum(int(s['channels']) for s in chosen))),
        ('File',     os.path.basename(out_path)),
    ]
    if want_trans:
        details.append(('Transcription', f'live · {language} · {model or "défaut (.env)"}'))
    if not confirm(details):
        return

    try:
        rec = recorder.Recorder(chosen, out_path, backend=backend)
    except Exception as e:
        print(err(f'  ✗ {e}'))
        pause()
        return

    transcriber = None
    transcript = None
    if want_trans:
        try:
            import live_transcribe
        except Exception as e:
            print(err(f'  ✗ Module transcription indisponible : {e}'))
            pause()
            return
        t_state = {'final': [], 'partial': {}}
        t_lock = threading.Lock()
        transcript = (t_state, t_lock)

        def _on_seg(seg, _st=t_state, _lk=t_lock):
            with _lk:
                if seg.get('interim'):
                    _st['partial'][seg['source_idx']] = seg
                else:
                    _st['final'].append(seg)
                    _st['partial'].pop(seg['source_idx'], None)

        srt_path = os.path.splitext(out_path)[0] + '.srt'
        try:
            transcriber = live_transcribe.LiveTranscriber(
                rec.channel_map(), _on_seg, srt_path,
                language=language, model_name=model)
        except Exception as e:
            print(err(f'  ✗ {e}'))
            pause()
            return
        rec.on_pcm = transcriber.feed_bytes
        print(dim(f'  Chargement du modèle Whisper ({transcriber.model_name})…'))
        try:
            transcriber.start()
        except Exception as e:
            print(err(f'  ✗ Modèle : {e}'))
            pause()
            return

    try:
        result = _record_live(rec, chosen, out_path, transcript)
    except Exception as e:
        rec.cancel()
        if transcriber:
            transcriber.finalize()
        print(err(f"  ✗ Erreur durant l'enregistrement : {e}"))
        pause()
        return

    if transcriber:
        print(dim('\n  Finalisation de la transcription…'))
        transcriber.finalize()

    print()
    stem = os.path.splitext(os.path.basename(out_path))[0]
    if result == 'cancelled':
        print(warn('  Enregistrement annulé (fichier supprimé).'))
        if transcriber:
            for ext in ('.srt', '.md'):
                try:
                    os.remove(os.path.splitext(out_path)[0] + ext)
                except OSError:
                    pass
    elif os.path.exists(out_path):
        print(ok('  ✓ Enregistré.') + f'  {dim("→")} {hi(os.path.basename(out_path))}')
        side = stem + '.channels.json'
        extra = ''
        if transcriber:
            n = len(transcriber.segments_snapshot())
            extra = (f' · transcript {stem}.srt + {stem}.md ({n} segments)' if n
                     else ' · transcription vide (aucune parole détectée)')
        print(dim(f'  Durée {_fmt_dur(rec.elapsed())} · {_size(out_path)} · '
                  f'{rec.total_channels} canaux · sidecar {side}{extra}'))
    else:
        print(err('  ✗ Aucun fichier produit (capture/ffmpeg en échec ?).'))
    pause()


def act_compress_audio(path: str):
    name = os.path.basename(path)
    print(f'\n  {bold("Compress audio")}  {dim(name)}\n')

    fmt_idx = select_menu([
        ('MP3  .mp3',  'widely compatible'),
        ('AAC  .m4a',  'good quality — Apple / mobile'),
        ('Opus .opus', 'best quality at low bitrate'),
        ('OGG  .ogg',  'open format'),
    ], title='Output format')
    if fmt_idx is None: return
    fmt_label, ext, codec = [
        ('MP3',  'mp3',  'libmp3lame'),
        ('AAC',  'm4a',  'aac'),
        ('Opus', 'opus', 'libopus'),
        ('OGG',  'ogg',  'libvorbis'),
    ][fmt_idx]

    br_idx = select_menu([
        (' 64 kbps', 'very light — voice / podcast'),
        (' 96 kbps', 'light — good voice quality'),
        ('128 kbps', 'standard                       (default)'),
        ('192 kbps', 'high quality'),
        ('320 kbps', 'maximum'),
    ], title='Bitrate')
    if br_idx is None: return
    bitrate = ['64k', '96k', '128k', '192k', '320k'][br_idx]

    base = os.path.splitext(name)[0]
    out  = os.path.join(os.path.dirname(path), f'{base}_compressed.{ext}')

    if not confirm([
        ('File',    name),
        ('Format',  fmt_label),
        ('Bitrate', bitrate),
        ('Output',  os.path.basename(out)),
    ]): return

    print()
    rc = _cmd('ffmpeg', '-y', '-i', path, '-acodec', codec, '-b:a', bitrate, out)
    _show_result(rc, out)
    pause()

# ---------------------------------------------------------------------------
# Actions — Audio
# ---------------------------------------------------------------------------

def act_convert_mp3(path: str):
    name = os.path.basename(path)
    print(f'\n  {bold("Convert to MP3")}  {dim(name)}\n')

    br_idx = select_menu([
        (' 64 kbps', 'very light — voice / podcast'),
        (' 96 kbps', 'light — good voice quality'),
        ('128 kbps', 'standard'),
        ('192 kbps', 'high quality                   (default)'),
        ('320 kbps', 'maximum'),
    ], title='Bitrate')
    if br_idx is None: return
    bitrate = ['64k', '96k', '128k', '192k', '320k'][br_idx]

    base = os.path.splitext(name)[0]
    out  = os.path.join(os.path.dirname(path), f'{base}.mp3')

    if not confirm([('File', name), ('Bitrate', bitrate), ('Output', os.path.basename(out))]):
        return
    print()
    rc = _cmd('ffmpeg', '-y', '-i', path, '-acodec', 'libmp3lame', '-b:a', bitrate, out)
    _show_result(rc, out)
    pause()


def act_improve_audio(path: str):
    name     = os.path.basename(path)
    base, ex = os.path.splitext(name)
    out      = os.path.join(os.path.dirname(path), f'{base}_quality_improved{ex}')
    if not confirm([('File', name), ('Output', os.path.basename(out))]):
        return
    print()
    rc = _py(os.path.join(_AUDIO_UTILS, 'improve_audio_quality.py'), path)
    _show_result(rc, out)
    pause()

# ---------------------------------------------------------------------------
# Actions — Document / Markdown
# ---------------------------------------------------------------------------

def act_transform_md(path: str):
    name = os.path.basename(path)
    base = os.path.splitext(name)[0]
    print(f'\n  {bold("AI Document Transformation")}  {dim(name)}\n')

    tmpl_idx = select_menu([
        ('Summary',         'concise summary of key facts and insights'),
        ('Podcast script',  'spoken-word transcript for audio / TTS synthesis'),
        ('Post-OCR cleanup','restore reading flow + fix OCR artefacts (1:1, no compression)'),
    ], title='Transformation template')
    if tmpl_idx is None: return

    # Post-OCR cleanup is content-preserving: skip target-length / extra-prompt
    # menus and route to the dedicated cleanup script.
    if tmpl_idx == 2:
        return _act_cleanup_ocr(path, name, base)

    template    = ['summary', 'podcast'][tmpl_idx]
    tmpl_labels = ['Summary', 'Podcast script']

    words_idx = select_menu([
        (' 500 words',  'short'),
        ('1000 words',  'medium'),
        ('2000 words',  'long'),
        ('3000 words',  'very long  (recommended for podcast)'),
        ('Same length', 'same content volume as source  (no compression)'),
        ('Custom',      'enter manually'),
    ], title='Target length')
    if words_idx is None: return
    same_length  = (words_idx == 4)
    target_words = None
    if words_idx == 5:
        target_words = ask('Target word count', '1000')
    elif not same_length:
        target_words = [500, 1000, 2000, 3000][words_idx]

    extra_prompt = ask('Extra instructions (Enter to skip)', '')

    lang_idx = select_menu([
        ('Auto-detect', 'detect language from document  (default)'),
        ('French',      ''),
        ('English',     ''),
        ('Other',       'enter manually'),
    ], title='Output language')
    if lang_idx is None: return
    if lang_idx == 0:
        lang = ''
    elif lang_idx == 3:
        lang = ask('Language')
    else:
        lang = ['', 'French', 'English'][lang_idx]

    swap_idx = select_menu([
        ('Auto   — script manages model load/unload', 'default'),
        ('Manual — model is already loaded',          'skip lifecycle management'),
    ], title='Model lifecycle')
    if swap_idx is None: return
    no_swap = swap_idx == 1

    out_path = os.path.join(os.path.dirname(path), f'{base}_{template}.md')

    details = [
        ('File',      name),
        ('Template',  tmpl_labels[tmpl_idx]),
        ('Target',    'same length as source' if same_length else f'{target_words} words'),
        ('Language',  lang or 'auto-detect'),
        ('Models',    'auto-swap' if not no_swap else 'manual'),
        ('Output',    os.path.basename(out_path)),
    ]
    if extra_prompt:
        details.insert(3, ('Instructions', extra_prompt[:60] + ('…' if len(extra_prompt) > 60 else '')))
    if not confirm(details): return

    args = [
        os.path.join(_AI_UTILS, 'transform_md.py'), path,
        '--template', template,
        '--output', out_path,
    ]
    if same_length:
        args += ['--same-length']
    else:
        args += ['--target-words', str(target_words)]
    if extra_prompt: args += ['--prompt', extra_prompt]
    if lang:         args += ['--lang', lang]
    if no_swap:      args += ['--no-model-swap']

    print()
    rc = _py(*args)
    _show_result(rc, out_path)
    pause()


def _act_cleanup_ocr(path: str, name: str, base: str):
    """Post-OCR cleanup branch of act_transform_md."""
    lang_idx = select_menu([
        ('Auto-detect', 'detect language from document  (default)'),
        ('French',      ''),
        ('English',     ''),
        ('Other',       'enter manually'),
    ], title='Source language')
    if lang_idx is None: return
    if lang_idx == 0:
        lang = ''
    elif lang_idx == 3:
        lang = ask('Language')
    else:
        lang = ['', 'French', 'English'][lang_idx]

    extra_prompt = ask('Extra cleanup instructions (Enter to skip)', '')

    swap_idx = select_menu([
        ('Auto   — script manages model load/unload', 'default'),
        ('Manual — model is already loaded',          'skip lifecycle management'),
    ], title='Model lifecycle')
    if swap_idx is None: return
    no_swap = swap_idx == 1

    out_path = os.path.join(os.path.dirname(path), f'{base}_cleaned.md')

    details = [
        ('File',     name),
        ('Template', 'Post-OCR cleanup  (1:1 content, fix artefacts)'),
        ('Language', lang or 'auto-detect'),
        ('Models',   'auto-swap' if not no_swap else 'manual'),
        ('Output',   os.path.basename(out_path)),
    ]
    if extra_prompt:
        details.insert(3, ('Focus', extra_prompt[:60] + ('…' if len(extra_prompt) > 60 else '')))
    if not confirm(details): return

    args = [os.path.join(_AI_UTILS, 'cleanup_ocr.py'), path,
            '--output', out_path]
    if lang:         args += ['--lang', lang]
    if extra_prompt: args += ['--prompt', extra_prompt]
    if no_swap:      args += ['--no-model-swap']

    print()
    rc = _py(*args)
    _show_result(rc, out_path)
    pause()


def act_md_to_audio(path: str):
    name = os.path.basename(path)
    print(f'\n  {bold("Text to Speech")}  {dim(name)}\n')

    lang_idx = select_menu([
        ('English — American', 'default'),
        ('English — British',  ''),
        ('French',             ''),
        ('Spanish',            ''),
        ('Italian',            ''),
        ('Japanese',           ''),
        ('Other',              'enter code manually'),
    ], title='Language')
    if lang_idx is None: return
    lang_codes = ['a', 'b', 'f', 'e', 'i', 'j', None]
    lang = lang_codes[lang_idx]
    if lang is None:
        lang = ask('Language code (a/b/f/e/p/i/j/z/h/ko  or  en/fr/es/…)')

    # Per-language voice catalogue — only voices that exist in hexgrad/Kokoro-82M
    voice_opts = {
        'a': [('af_heart',  'F – Warm, expressive'),
              ('af_bella',  'F – Soft, clear'),
              ('af_nicole', 'F – Breathy, intimate'),
              ('af_alloy',  'F – Smooth, balanced'),
              ('af_jessica','F – Friendly, warm'),
              ('af_kore',   'F – Confident, professional'),
              ('af_nova',   'F – Crisp, modern'),
              ('af_river',  'F – Flowing, natural'),
              ('af_sarah',  'F – Natural, conversational'),
              ('af_sky',    'F – Young, energetic'),
              ('am_adam',   'M – Deep, steady'),
              ('am_echo',   'M – Clear, neutral'),
              ('am_eric',   'M – Warm, friendly'),
              ('am_fenrir', 'M – Strong, authoritative'),
              ('am_liam',   'M – Calm, measured'),
              ('am_michael','M – Rich, full voice'),
              ('am_onyx',   'M – Deep, resonant'),
              ('am_puck',   'M – Playful, light'),
              ('am_santa',  'M – Jolly, warm')],
        'b': [('bf_alice',   'F – Elegant, crisp'),
              ('bf_emma',    'F – Warm, articulate'),
              ('bf_isabella','F – Refined, expressive'),
              ('bf_lily',    'F – Soft, gentle'),
              ('bm_daniel',  'M – Clear, professional'),
              ('bm_fable',   'M – Storytelling tone'),
              ('bm_george',  'M – Authoritative, rich'),
              ('bm_lewis',   'M – Friendly, modern')],
        'f': [('ff_siwis',  'F – Natural French (only available voice)')],
        'e': [('ef_dora',   'F – Warm Spanish'),
              ('em_alex',   'M – Clear Spanish'),
              ('em_santa',  'M – Expressive Spanish')],
        'p': [('pf_dora',   'F – Natural Portuguese'),
              ('pm_alex',   'M – Clear Portuguese'),
              ('pm_santa',  'M – Expressive Portuguese')],
        'i': [('if_sara',   'F – Warm Italian'),
              ('im_nicola', 'M – Smooth Italian')],
        'j': [('jf_alpha',      'F – Clear female'),
              ('jf_gongitsune', 'F – Soft, gentle'),
              ('jf_nezumi',     'F – Light, nimble'),
              ('jf_tebukuro',   'F – Warm, cozy'),
              ('jm_kumo',       'M – Calm male')],
        'z': [('zf_xiaobei',  'F – Warm female'),
              ('zf_xiaoni',   'F – Gentle female'),
              ('zf_xiaoxiao', 'F – Lively female'),
              ('zf_xiaoyi',   'F – Bright female'),
              ('zm_yunjian',  'M – Clear male'),
              ('zm_yunxi',    'M – Natural male'),
              ('zm_yunxia',   'M – Deep male'),
              ('zm_yunyang',  'M – Warm male')],
        'h': [('hf_alpha',  'F – Natural female'),
              ('hf_beta',   'F – Soft female'),
              ('hm_omega',  'M – Deep male'),
              ('hm_psi',    'M – Clear male')],
    }
    voices = voice_opts.get(lang, [])
    voice  = ''
    if voices:
        v_items = [(vid, desc) for vid, desc in voices] + [('custom', 'enter name manually')]
        v_idx   = select_menu(v_items, title='Voice')
        if v_idx is None: return
        voice = ask('Voice name') if v_idx == len(voices) else voices[v_idx][0]

    speed_idx = select_menu([
        ('0.95', 'natural, slightly measured  (default)'),
        ('0.85', 'slow, very clear'),
        ('1.00', 'standard speed'),
        ('1.10', 'fast'),
        ('custom', 'enter manually'),
    ], title='Speaking speed')
    if speed_idx is None: return
    speeds = ['0.95', '0.85', '1.00', '1.10', None]
    speed  = speeds[speed_idx] or ask('Speed multiplier', '0.95')

    base    = os.path.splitext(name)[0]
    out_mp3 = os.path.join(os.path.dirname(path), f'{base}.mp3')

    if not confirm([
        ('File',   name),
        ('Lang',   lang),
        ('Voice',  voice or '(default for lang)'),
        ('Speed',  speed),
        ('Output', os.path.basename(out_mp3)),
    ]): return

    args = [path, '--lang', lang, '--speed', speed]
    if voice: args += ['--voice', voice]
    print()
    rc = _py(os.path.join(_DOC_UTILS, 'md_to_audio.py'), *args)
    _show_result(rc, out_mp3)
    pause()


def act_md_to_pdf(path: str):
    name = os.path.basename(path)
    print(f'\n  {bold("Markdown → PDF")}  {dim(name)}\n')

    tmpl_idx = select_menu([
        ('simple', 'clean layout, no table of contents  (default)'),
        ('report', 'adds TOC + colored links'),
    ], title='Template')
    if tmpl_idx is None: return
    tmpl = ['simple', 'report'][tmpl_idx]

    base = os.path.splitext(name)[0]
    out  = os.path.join(os.path.dirname(path), f'{base}.pdf')

    if not confirm([('File', name), ('Template', tmpl), ('Output', os.path.basename(out))]):
        return
    print()
    rc = _py(os.path.join(_DOC_UTILS, 'md_to_pdf', 'md_to_pdf.py'), path, '--template', tmpl)
    _show_result(rc, out)
    pause()


def _simple_convert(path: str, title: str, script: str, out_ext: str):
    name = os.path.basename(path)
    base = os.path.splitext(name)[0]
    out  = os.path.join(os.path.dirname(path), f'{base}{out_ext}')
    print(f'\n  {bold(title)}  {dim(name)}\n')
    if not confirm([('File', name), ('Output', os.path.basename(out))]):
        return
    print()
    rc = _py(script, path)
    _show_result(rc, out)
    pause()


def act_md_to_docx(path: str):
    _simple_convert(path, 'Markdown → DOCX', os.path.join(_DOC_UTILS,'md_to_docx.py'), '.docx')

def act_doc_to_md(path: str):
    _simple_convert(path, 'DOC → Markdown', os.path.join(_DOC_UTILS,'doc_to_md.py'), '.md')

def act_doc_to_pdf(path: str):
    _simple_convert(path, 'DOC → PDF', os.path.join(_DOC_UTILS,'doc_to_pdf.py'), '.pdf')

def act_odt_to_docx(path: str):
    _simple_convert(path, 'ODT → DOCX', os.path.join(_DOC_UTILS,'odt_to_docx.py'), '.docx')

def act_odt_to_pdf(path: str):
    _simple_convert(path, 'ODT → PDF', os.path.join(_DOC_UTILS,'doc_to_pdf.py'), '.pdf')

def act_ppt_to_pdf(path: str):
    _simple_convert(path, 'PPT → PDF', os.path.join(_DOC_UTILS,'ppt_to_pdf.py'), '.pdf')

def act_xls_to_pdf(path: str):
    _simple_convert(path, 'XLS → PDF', os.path.join(_DOC_UTILS,'xls_to_pdf.py'), '.pdf')


def _pick_ocr_lang(title='OCR language') -> str | None:
    idx = select_menu([
        ('FR — French',  'default'),
        ('EN — English', ''),
    ], title=title)
    if idx is None: return None
    return ['fra', 'eng'][idx]


def act_pdf_vision_ocr(path: str):
    name = os.path.basename(path)
    base = os.path.splitext(name)[0]
    print(f'\n  {bold("Vision OCR")}  {dim(name)}\n')

    dpi_idx = select_menu([
        ('150 DPI', 'fast — clean printed text'),
        ('200 DPI', 'balanced quality / speed  (default)'),
        ('300 DPI', 'best — dense text, small fonts, complex layouts'),
    ], title='Rendering resolution')
    if dpi_idx is None: return
    dpi = [150, 200, 300][dpi_idx]

    lang_idx = select_menu([
        ('English', 'default'),
        ('French',  ''),
        ('Dutch',   ''),
        ('Other…',  'enter manually'),
    ], title='Document language')
    if lang_idx is None: return
    if lang_idx == 3:
        lang = ask('Language (e.g. German, Spanish, Italian)', '').strip()
        if not lang: return
    else:
        lang = ['English', 'French', 'Dutch'][lang_idx]

    pages = ask('Page range (e.g. 1-3,5  — Enter for all pages)', '')
    extra_prompt = ask('Extra instructions (Enter to skip)', '')

    out_path = os.path.join(os.path.dirname(path), f'{base}_ocr.md')

    details = [('File', name), ('DPI', str(dpi)), ('Language', lang)]
    if pages:        details.append(('Pages', pages))
    if extra_prompt: details.append(('Instructions', extra_prompt[:60] + ('…' if len(extra_prompt) > 60 else '')))
    details.append(('Output', os.path.basename(out_path)))
    if not confirm(details): return

    args = [os.path.join(_AI_UTILS, 'pdf_vision_ocr.py'), path,
            '--dpi', str(dpi), '--lang', lang, '--output', out_path]
    if pages:        args += ['--pages', pages]
    if extra_prompt: args += ['--prompt', extra_prompt]

    print()
    rc = _py(*args)
    _show_result(rc, out_path)
    pause()


def act_pdf_mineru(path: str):
    name = os.path.basename(path)
    base = os.path.splitext(name)[0]
    print(f'\n  {bold("MinerU OCR")}  {dim(name)}\n')

    method_idx = select_menu([
        ('auto', 'detect text layer vs scanned  (default)'),
        ('txt',  'text-layer only — fastest, no OCR'),
        ('ocr',  'force OCR on every page'),
    ], title='Parsing method')
    if method_idx is None: return
    method = ['auto', 'txt', 'ocr'][method_idx]

    lang_idx = select_menu([
        ('en',     'English  (default)'),
        ('latin',  'Latin scripts — French / Spanish / Italian / German / Dutch'),
        ('ch',     'Simplified Chinese'),
        ('Other…', 'japan, korean, arabic, cyrillic, devanagari, …'),
    ], title='OCR language')
    if lang_idx is None: return
    if lang_idx == 3:
        lang = ask('Language code (japan, korean, ch_server, arabic, …)', '').strip()
        if not lang: return
    else:
        lang = ['en', 'latin', 'ch'][lang_idx]

    dev_idx = select_menu([
        ('CUDA', 'GPU — fits in ~3 GB VRAM with pipeline backend  (default)'),
        ('CPU',  'no GPU — slower but always works'),
    ], title='Inference device')
    if dev_idx is None: return
    device = ['cuda', 'cpu'][dev_idx]

    out_dir = os.path.join(os.path.dirname(path), f'{base}_mineru')
    out_md  = os.path.join(os.path.dirname(path), f'{base}_mineru.md')

    if not confirm([
        ('File',   name),
        ('Method', method),
        ('Lang',   lang),
        ('Device', device),
        ('Output', f'{base}_mineru.md  +  {base}_mineru/'),
    ]): return

    if os.path.exists(out_dir):
        print(err(f'\n  ✗ Output folder already exists: {out_dir}'))
        print(dim('    Delete it manually and re-run.'))
        pause()
        return

    print()
    rc = _py(os.path.join(_DOC_UTILS, 'pdf_mineru_to_md.py'), path,
             '--method', method, '--lang', lang, '--device', device)
    _show_result(rc, out_md)
    pause()


def act_pdf_extract(path: str):
    name = os.path.basename(path)
    print(f'\n  {bold("PDF → Markdown")}  {dim(name)}\n')
    lang = _pick_ocr_lang()
    if lang is None: return
    base = os.path.splitext(name)[0]
    out  = os.path.join(os.path.dirname(path), f'{base}_extracted.md')
    if not confirm([('File', name), ('OCR lang', lang), ('Output', os.path.basename(out))]):
        return
    print()
    rc = _py(os.path.join(_DOC_UTILS, 'pdf_extract_to_md.py'), path, '--language', lang)
    _show_result(rc, out)
    pause()


def act_pdf_ocr(path: str):
    name = os.path.basename(path)
    print(f'\n  {bold("PDF OCR")}  {dim(name)}\n')
    lang = _pick_ocr_lang()
    if lang is None: return

    mode_idx = select_menu([
        ('skip',  'OCR only pages without text  (default, safe)'),
        ('redo',  'replace existing text layer with fresh OCR'),
        ('force', 'discard all existing text and re-OCR everything'),
    ], title='Existing text handling')
    if mode_idx is None: return
    mode = ['skip', 'redo', 'force'][mode_idx]

    out_idx = select_menu([
        ('PDF', 'embed OCR text layer — keeps original layout  (default)'),
        ('MD',  'extract text to Markdown — plain text, no layout'),
    ], title='Output format')
    if out_idx is None: return
    output = ['pdf', 'md'][out_idx]

    base    = os.path.splitext(name)[0]
    out_ext = f'_ocr.{output}'
    out     = os.path.join(os.path.dirname(path), f'{base}{out_ext}')

    if not confirm([
        ('File',   name),
        ('Lang',   lang),
        ('Mode',   mode),
        ('Output', os.path.basename(out)),
    ]): return

    print()
    rc = _py(os.path.join(_DOC_UTILS, 'pdf_ocr.py'), path,
             '--language', lang, '--mode', mode, '--output', output)
    _show_result(rc, out)
    pause()


def act_split_pdf(path: str):
    name = os.path.basename(path)
    print(f'\n  {bold("Split PDF pages")}  {dim(name)}\n')
    if not confirm([('File', name), ('Output', 'one PDF per page in same folder')]):
        return
    print()
    rc = _py(os.path.join(_DOC_UTILS, 'split_pdf_pages.py'), path)
    _show_result(rc)
    pause()

# ---------------------------------------------------------------------------
# Actions — Image / folder
# ---------------------------------------------------------------------------

def act_thumbnails(path: str):
    """path is a directory."""
    label = os.path.basename(path) or path
    print(f'\n  {bold("Create thumbnails")}  {dim(label)}\n')

    # Output folder — blank = thumbnails/ subfolder beside each source file
    out_folder = ask('Output folder  (Enter = thumbnails/ subfolder)', '')

    size_idx = select_menu([
        ('1920 × 1080', 'Full HD                      (default)'),
        ('1280 × 720',  'HD'),
        ('800 × 600',   'medium'),
        ('Custom',      'enter manually'),
    ], title='Image thumbnail size')
    if size_idx is None: return
    if size_idx == 3:
        w, h = ask('Width (px)', '1920'), ask('Height (px)', '1080')
    else:
        w, h = [('1920','1080'),('1280','720'),('800','600')][size_idx]

    vid_idx = select_menu([
        ('640 × 480',   'SD                           (default)'),
        ('1280 × 720',  'HD'),
        ('1920 × 1080', 'Full HD'),
        ('Custom',      'enter manually'),
    ], title='Video thumbnail size')
    if vid_idx is None: return
    if vid_idx == 3:
        vw, vh = ask('Video width (px)', '640'), ask('Video height (px)', '480')
    else:
        vw, vh = [('640','480'),('1280','720'),('1920','1080')][vid_idx]

    quality = ask('JPEG quality (1–100)', '85')

    rec_idx = select_menu([('Yes — include subfolders','default'),('No  — top only','')], title='Recursive')
    if rec_idx is None: return
    ow_idx  = select_menu([('No  — skip existing','default'),('Yes — overwrite','')], title='Overwrite existing')
    if ow_idx  is None: return

    out_label = out_folder.strip() if out_folder.strip() else 'thumbnails/ (beside source)'
    if not confirm([
        ('Source folder', label),
        ('Output',        out_label),
        ('Image size',    f'{w} × {h} px'),
        ('Video size',    f'{vw} × {vh} px'),
        ('Quality',       quality),
        ('Recursive',     'yes' if rec_idx==0 else 'no'),
        ('Overwrite',     'yes' if ow_idx==1  else 'no'),
    ]): return

    args = [path, '--size', w, h, '--video-size', vw, vh, '--quality', quality]
    if out_folder.strip(): args += ['-o', out_folder.strip()]
    if rec_idx == 1: args.append('--no-recursive')
    if ow_idx  == 1: args.append('--overwrite')
    print()
    rc = _py(_FOLDER_REGISTRY['thumbnails']['script'], *args)
    _show_result(rc)
    pause()


def act_raw_to_jpg(path: str):
    """path is a directory."""
    label = os.path.basename(path) or path
    print(f'\n  {bold("RAW → JPEG")}  {dim(label)}\n')

    quality = ask('JPEG quality (1–100)', '90')
    rec_idx = select_menu([('Yes — include subfolders','default'),('No  — top only','')], title='Recursive')
    if rec_idx is None: return
    ow_idx  = select_menu([('No  — skip existing','default'),('Yes — overwrite','')], title='Overwrite existing')
    if ow_idx  is None: return
    move    = ask('Move RAWs to subfolder after conversion (blank to skip)', '')

    details = [('Folder',label),('Quality',quality),
               ('Recursive','yes' if rec_idx==0 else 'no'),
               ('Overwrite','yes' if ow_idx==1  else 'no')]
    if move: details.append(('Move RAWs to', move))
    if not confirm(details): return

    args = [path, '--quality', quality]
    if rec_idx == 1: args.append('--no-recursive')
    if ow_idx  == 1: args.append('--overwrite')
    if move:         args += ['--move-raws-to', move]
    print()
    rc = _py(_FOLDER_REGISTRY['raw-to-jpg']['script'], *args)
    _show_result(rc)
    pause()


def act_image_dedup(path: str):
    """path is a directory — exact (SHA-256) duplicate detection → duplicates.json."""
    label = os.path.basename(path) or path
    print(f'\n  {bold("Find duplicate images (exact)")}  {dim(label)}\n')

    workers = ask('Hashing threads (I/O)', '8')
    if not confirm([
        ('Folder',   label),
        ('Method',   'exact — SHA-256 (zero false positives)'),
        ('Output',   'duplicates.json  (at folder root)'),
        ('Threads',  workers),
    ]): return

    print()
    out = os.path.join(path, 'duplicates.json')
    rc = _py(_FOLDER_REGISTRY['image-dedup']['script'], path, '--workers', workers)
    _show_result(rc, out)
    pause()


def act_merge_pdf(directory: str):
    files = sorted(f for f in os.listdir(directory)
                   if os.path.splitext(f)[1].lower() == '.pdf'
                   and os.path.isfile(os.path.join(directory, f)))
    print(f'\n  {bold("Merge PDFs")}\n')
    if len(files) < 2:
        print(warn(f'  Need at least 2 PDF files, found {len(files)}.'))
        pause()
        return
    for f in files:
        print(f'  {dim("·")} {hi(f)}')
    out_name = os.path.basename(directory) + '_merged.pdf'
    out_path = os.path.join(directory, out_name)
    if not confirm([('Files', str(len(files))), ('Output', out_name)]): return
    print()
    rc = _py(os.path.join(_DOC_UTILS, 'merge_pdf.py'), out_path,
             *[os.path.join(directory, f) for f in files])
    _show_result(rc, out_path)
    pause()


def act_merge_md(directory: str):
    files = sorted(f for f in os.listdir(directory)
                   if f.endswith('.md') and os.path.isfile(os.path.join(directory, f)))
    print(f'\n  {bold("Merge Markdown")}\n')
    if len(files) < 2:
        print(warn(f'  Need at least 2 .md files, found {len(files)}.'))
        pause()
        return
    for f in files:
        print(f'  {dim("·")} {hi(f)}')
    if not confirm([('Files', str(len(files))),
                    ('Output', os.path.basename(directory) + '_merged.md')]): return
    print()
    rc = _py(os.path.join(_DOC_UTILS, 'merge_md.py'), '--folder', directory)
    _show_result(rc)
    pause()


def act_git_pull_all(directory: str):
    print(f'\n  {bold("Git Pull All")}  {dim(os.path.basename(directory))}\n')

    mode_idx = select_menu([
        ('Pull',         'git pull — keep local changes when possible (default)'),
        ('Force reset',  'fetch + reset --hard — DISCARD all local changes'),
    ], title='Mode')
    if mode_idx is None:
        return

    force = mode_idx == 1

    if force:
        print(f'\n  {err("WARNING: all local modifications in every repository will be permanently lost.")}')
        if not confirm([
            ('Root',   directory),
            ('Action', 'git fetch --all + git reset --hard origin/<branch> on EVERY repo'),
            ('Risk',   'local uncommitted changes and unpushed commits will be LOST'),
        ]):
            return
    else:
        if not confirm([('Root', directory), ('Action', 'git pull on every repository found recursively')]):
            return

    print()
    extra = ['--force'] if force else []
    rc = _py(os.path.join(_DEV_UTILS, 'git_pull_all.py'), directory, *extra)
    _show_result(rc)
    pause()


def act_chat_model(directory: str):
    print(f'\n  {bold("Chat with AI")}  {dim(os.path.basename(directory))}\n')

    quality_idx = select_menu([
        ('Model B — balanced',    'quality / speed  (default)'),
        ('Model A — best quality','larger model, slower, more capable'),
    ], title='Model quality')
    if quality_idx is None: return
    quality = ['b', 'a'][quality_idx]

    default_name = datetime.now().strftime('%Y-%m-%d') + ' - Chat.md'
    out_name = ask('Output file name', default_name).strip()
    if not out_name: return
    if not out_name.endswith('.md'):
        out_name += '.md'
    out_path = os.path.join(directory, out_name)

    details = [
        ('Folder', os.path.basename(directory)),
        ('Model',  'A — best quality' if quality == 'a' else 'B — balanced'),
        ('Output', out_name),
    ]
    if not confirm(details): return

    print()
    _py(os.path.join(_AI_UTILS, 'chat_model.py'), directory,
        '--quality', quality, '--output', out_path)
    # No pause() — the chat session is its own interactive experience


# ---------------------------------------------------------------------------
# Action registry
# ---------------------------------------------------------------------------

_CAT_ORDER  = ['AI', 'Audio', 'Video', 'Document', 'Image', 'Developer']
_CAT_COLORS = {
    'AI':        _CYN,
    'Audio':     _YEL,
    'Video':     _BLU,
    'Document':  _GRN,
    'Image':     _MGT,
    'Developer': _RED,
}
_ACTION_CAT = {
    'Chat with AI':               'AI',
    'AI Transform':               'AI',
    'Vision OCR':                 'AI',
    'MinerU OCR':                 'AI',
    'Transcribe':                 'Audio',
    'Record audio':               'Audio',
    'Extract audio':              'Audio',
    'Convert to MP3':             'Audio',
    'Improve quality':            'Audio',
    'Compress audio':             'Audio',
    'Compress':                   'Audio',
    'Text to speech':             'Audio',
    'Split video':                'Video',
    'Merge videos':               'Video',
    'Merge PDFs':                 'Document',
    'Merge Markdown':             'Document',
    'Export to PDF':              'Document',
    'Export to DOCX':             'Document',
    'Extract to Markdown':        'Document',
    'Add OCR layer':              'Document',
    'Split pages':                'Document',
    'Convert to Markdown':        'Document',
    'Convert to PDF':             'Document',
    'Create thumbnails':          'Image',
    'Convert folder RAWs to JPEG':'Image',
    'RAW → JPEG':                 'Image',
    'Find duplicate images':      'Image',
    'Git Pull All':               'Developer',
}


def _group_actions(acts: list) -> list:
    """
    Sort actions by category then alphabetically.
    Prefixes each label with a colored ▎ bar — the color change between
    consecutive items visually separates groups without non-selectable rows.
    Returns [(prefixed_label, hint, fn), ...]
    """
    buckets: dict = {}
    for label, hint, fn in acts:
        cat = _ACTION_CAT.get(label, 'Other')
        buckets.setdefault(cat, []).append((label, hint, fn))

    for cat in buckets:
        buckets[cat].sort(key=lambda x: x[0].lower())

    result = []
    order = _CAT_ORDER + sorted(set(buckets) - set(_CAT_ORDER))
    for cat in order:
        if cat not in buckets:
            continue
        color = _CAT_COLORS.get(cat, _GRY)
        for label, hint, fn in buckets[cat]:
            result.append((f'{color}▎{_R} {label}', hint, fn))

    return result


def _actions_for(path: str) -> list[tuple[str, str, callable]]:
    """Return [(label, hint, fn), …] for path (file or directory)."""
    acts = []

    if os.path.isdir(path):
        try:
            exts = {os.path.splitext(f)[1].lower()
                    for f in os.listdir(path)
                    if os.path.isfile(os.path.join(path, f))}
        except:
            exts = set()
        acts.append(('Chat with AI',      'interactive chat — saved to Markdown', act_chat_model))
        acts.append(('Record audio',      'capture micro + sortie → FLAC multicanal', act_record_audio))
        acts.append(('Git Pull All',      'pull every git repo found recursively', act_git_pull_all))
        if '.pdf' in exts:
            acts.append(('Merge PDFs',        'combine all .pdf files in folder',      act_merge_pdf))
        if '.md' in exts:
            acts.append(('Merge Markdown',    'combine all .md files in folder',       act_merge_md))
        # Show thumbnails for any folder that contains images, RAW or video files
        # (also shown when folder only has sub-dirs — recursive mode will find them)
        if (IMAGE_EXTS | RAW_EXTS | VIDEO_EXTS) & exts or not exts:
            acts.append(('Create thumbnails', 'batch JPEG/video thumbnails',           act_thumbnails))
        if RAW_EXTS & exts:
            acts.append(('RAW → JPEG',        'batch convert all RAW files',           act_raw_to_jpg))
        if (IMAGE_EXTS | RAW_EXTS) & exts:
            acts.append(('Find duplicate images', 'exact dedup (SHA-256) → duplicates.json', act_image_dedup))
        return acts

    ext = os.path.splitext(path)[1].lower()

    if ext in VIDEO_EXTS:
        acts += [
            ('Transcribe',     'speech → Markdown / SRT  (Whisper)',  act_transcribe),
            ('Extract audio',  'stream-copy + EBU R128 FLAC',          act_extract_audio),
            ('Split video',    'cut between two timestamps',            act_split),
            ('Merge videos',   'append a second video at the end',      act_merge_videos),
            ('Compress audio', 'MP3 / AAC / Opus / OGG',               act_compress_audio),
        ]
    if ext in AUDIO_EXTS:
        acts += [
            ('Transcribe',      'speech → Markdown / SRT  (Whisper)', act_transcribe),
            ('Convert to MP3',  're-encode at chosen quality',          act_convert_mp3),
            ('Improve quality', 'EBU R128 loudness normalization',      act_improve_audio),
            ('Compress',        'choose format + bitrate',              act_compress_audio),
        ]
    if ext in MD_EXT:
        acts += [
            ('AI Transform',   'summary / podcast / …  (LMStudio · Ollama)',  act_transform_md),
            ('Text to speech', 'Kokoro TTS → MP3',      act_md_to_audio),
            ('Export to PDF',  'pandoc + XeLaTeX',      act_md_to_pdf),
            ('Export to DOCX', 'pandoc',                act_md_to_docx),
        ]
    if ext in PDF_EXT:
        acts += [
            ('Vision OCR',          'local AI vision model → Markdown',  act_pdf_vision_ocr),
            ('MinerU OCR',          'layout + formulas + tables → Markdown', act_pdf_mineru),
            ('Extract to Markdown', 'PyMuPDF + OCR fallback',         act_pdf_extract),
            ('Add OCR layer',       'OCRmyPDF',                       act_pdf_ocr),
            ('Split pages',         'one PDF per page',               act_split_pdf),
        ]
    if ext in DOC_EXTS:
        acts += [
            ('Convert to Markdown', 'pandoc',       act_doc_to_md),
            ('Convert to PDF',      'LibreOffice',  act_doc_to_pdf),
        ]
    if ext in ODT_EXTS:
        acts += [
            ('Convert to DOCX', 'LibreOffice',  act_odt_to_docx),
            ('Convert to PDF',  'LibreOffice',  act_odt_to_pdf),
        ]
    if ext in PPT_EXTS:
        acts += [('Convert to PDF', 'LibreOffice',  act_ppt_to_pdf)]
    if ext in SHEET_EXTS:
        acts += [('Convert to PDF', 'LibreOffice',  act_xls_to_pdf)]
    if ext in RAW_EXTS:
        acts += [('Convert folder RAWs to JPEG', 'batch convert all RAWs in this folder',
                  lambda p: act_raw_to_jpg(os.path.dirname(p)))]

    return acts

# ---------------------------------------------------------------------------
# Action menu — shown after file selection
# ---------------------------------------------------------------------------

def action_menu(path: str):
    acts = _actions_for(path)
    if not acts:
        print(warn(f'\n  No actions available for: {os.path.basename(path)}\n'))
        pause()
        return

    is_dir = os.path.isdir(path)
    name   = os.path.basename(path) + ('/' if is_dir else '')
    sub    = _count(path) if is_dir else _size(path)

    grouped    = _group_actions(acts)
    menu_items = [(label, hint) for label, hint, _ in grouped]

    idx = select_menu(menu_items, title=name, subtitle=sub)
    if idx is None:
        return

    _, _, fn = grouped[idx]
    fn(path)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_AUTO_KEYS = {'OMP_NUM_THREADS', 'THUMBNAIL_MAX_GPU_SESSIONS',
              'THUMBNAIL_NUM_CORES', 'RAW_TO_JPG_NUM_CORES',
              'LIVE_TRANSCRIBE_MODEL', 'TRANSCRIBE_MODEL'}

def _maybe_setup_env():
    """Run setup_env.py if any auto-detected keys are missing from .env."""
    env_path = os.path.join(_SCRIPT_DIR, '.env')
    missing  = set(_AUTO_KEYS)
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith('#') and '=' in s:
                    missing.discard(s.split('=', 1)[0])
    if not missing:
        return  # all keys present, nothing to do

    setup = os.path.join(_SCRIPT_DIR, 'setup_env.py')
    if not os.path.exists(setup):
        return

    print(dim('  Configuring .env for this machine…'))
    subprocess.run([_PYTHON, setup], check=False)
    print()


def main():
    parser = argparse.ArgumentParser(description='Utils Tools — interactive TUI.')
    parser.add_argument('--workdir', default=os.getcwd(),
                        help='Starting directory (passed by Windows launcher)')
    args = parser.parse_args()

    workdir = os.path.abspath(args.workdir)
    if not os.path.isdir(workdir):
        print(f'Error: not a directory: {workdir}', file=sys.stderr)
        sys.exit(1)

    _maybe_setup_env()

    sys.stdout.write('\033[2J\033[H')
    sys.stdout.flush()

    last_dir = workdir
    try:
        while True:
            result = browse(last_dir)
            if result is None:
                break
            path, last_dir = result
            action_menu(path)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(_SHOW)
        sys.stdout.flush()
    print()


if __name__ == '__main__':
    main()
