#!/usr/bin/env python3
"""
recorder.py — moteur de capture audio multi-sources pour utils.

Première étape vers une app de transcription par canal : enregistre plusieurs
sources audio (micros + sorties système) dans un **seul fichier FLAC
multicanal**, accompagné d'un sidecar `<name>.channels.json` qui décrit quel
intervalle de canaux appartient à quelle source — la base du futur démux +
transcription par canal (`transcribe_audio.py`).

Backend-aware (la TUI ne voit qu'une interface uniforme) :

  * 'linux' : PulseAudio / PipeWire. Les sources sont les `pactl` sources ;
              les SORTIES sont les sources `<sink>.monitor`. Capture via
              `parec`, encodage via `ffmpeg`. Tout est local.

  * 'wsl'   : la capture tourne côté Windows via le binaire WASAPI autonome
              `bin/capture.exe` (loopback WASAPI = API Windows intégrée, donc
              **rien à installer** côté Windows), lancé en interop WSL. Le
              binaire émet du PCM s16le interleavé sur stdout (ordre = ordre des
              `--source`) et des lignes `LEVEL <idx> <rms>` sur stderr.
              L'encodage FLAC reste dans WSL avec `ffmpeg`.

Aucune dépendance Python externe au niveau module (stdlib seulement) afin que
l'import et l'énumération fonctionnent même sur une box headless sans audio.
numpy n'est importé que si présent, pour le calcul de niveau du backend Linux.
"""

import json
import math
import os
import re
import shutil
import struct
import subprocess
import threading
import time

_HERE         = os.path.dirname(os.path.abspath(__file__))
CAPTURE_EXE   = os.path.join(_HERE, 'bin', 'capture.exe')
CAPTURE_DIR   = os.path.join(_HERE, 'capture')
CAPTURE_BUILD = os.path.join(CAPTURE_DIR, 'build.sh')

DEFAULT_RATE = 48000


# ---------------------------------------------------------------------------
# Binaire de capture Windows (non versionné — construit à la demande)
# ---------------------------------------------------------------------------

def capture_exe_present() -> bool:
    return os.path.exists(CAPTURE_EXE)


def build_capture() -> tuple[bool, str]:
    """Tente de construire capture.exe via capture/build.sh. Renvoie (ok, log)."""
    if not os.path.exists(CAPTURE_BUILD):
        return False, f'script de build absent : {CAPTURE_BUILD}'
    try:
        r = subprocess.run(['bash', CAPTURE_BUILD],
                           capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)
    log = ((r.stdout or '') + (r.stderr or '')).strip()
    return (r.returncode == 0 and capture_exe_present()), log


# ---------------------------------------------------------------------------
# Détection d'environnement
# ---------------------------------------------------------------------------

def detect_backend() -> str:
    """'wsl' si on tourne sous WSL (capture déléguée à Windows), sinon 'linux'."""
    try:
        with open('/proc/version', 'r', encoding='utf-8', errors='ignore') as f:
            v = f.read().lower()
        if 'microsoft' in v or 'wsl' in v:
            return 'wsl'
    except OSError:
        pass
    return 'linux'


# ---------------------------------------------------------------------------
# Énumération des sources
# ---------------------------------------------------------------------------
#
# Forme d'une source : dict
#   { 'id': str,            # identifiant passé au backend (nom pulse / id WASAPI)
#     'name': str,          # libellé lisible
#     'kind': 'input'|'output',
#     'channels': int }     # nb de canaux capturés pour cette source

def list_sources(backend: str | None = None) -> list[dict]:
    """Liste les sources capturables, groupables par `kind`. [] si indisponible."""
    backend = backend or detect_backend()
    if backend == 'wsl':
        return _list_sources_wsl()
    return _list_sources_linux()


def _list_sources_linux() -> list[dict]:
    if not shutil.which('pactl'):
        return []
    try:
        short = subprocess.run(['pactl', 'list', 'short', 'sources'],
                               capture_output=True, text=True, timeout=5)
        full = subprocess.run(['pactl', 'list', 'sources'],
                              capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    if short.returncode != 0:
        return []

    # nom -> nb canaux, depuis les blocs détaillés ("Sample Specification: ... 2ch ...")
    channels_by_name: dict[str, int] = {}
    desc_by_name: dict[str, int] = {}
    cur = None
    for line in full.stdout.splitlines():
        s = line.strip()
        if s.startswith('Name:'):
            cur = s.split(':', 1)[1].strip()
        elif cur and s.startswith('Description:'):
            desc_by_name[cur] = s.split(':', 1)[1].strip()
        elif cur and s.startswith('Sample Specification:'):
            m = re.search(r'(\d+)ch', s)
            if m:
                channels_by_name[cur] = int(m.group(1))

    sources = []
    for line in short.stdout.splitlines():
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        name = parts[1].strip()
        if not name:
            continue
        kind = 'output' if name.endswith('.monitor') else 'input'
        sources.append({
            'id': name,
            'name': desc_by_name.get(name, name),
            'kind': kind,
            'channels': channels_by_name.get(name, 2 if kind == 'output' else 1),
        })
    return sources


def _list_sources_wsl() -> list[dict]:
    if not os.path.exists(CAPTURE_EXE):
        return []
    try:
        r = subprocess.run([CAPTURE_EXE, '--list'],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    if r.returncode != 0 or not r.stdout.strip():
        return []
    try:
        raw = json.loads(r.stdout)
    except json.JSONDecodeError:
        return []
    sources = []
    for e in raw:
        kind = e.get('kind', 'input')
        sources.append({
            'id': str(e.get('id', '')),
            'name': e.get('name', e.get('id', '')),
            'kind': 'output' if kind == 'output' else 'input',
            'channels': int(e.get('channels', 2 if kind == 'output' else 1)),
        })
    return [s for s in sources if s['id']]


# ---------------------------------------------------------------------------
# Nom de fichier par défaut
# ---------------------------------------------------------------------------

def default_basename(now: time.struct_time | None = None) -> str:
    """`YYYY-MM-DD-HH-MM - capture` (format par défaut demandé)."""
    t = now or time.localtime()
    return time.strftime('%Y-%m-%d-%H-%M', t) + ' - capture'


def unique_path(dirpath: str, basename: str, ext: str = 'flac') -> str:
    """Chemin <dir>/<basename>.<ext>, suffixé d'un compteur si déjà présent."""
    cand = os.path.join(dirpath, f'{basename}.{ext}')
    if not os.path.exists(cand):
        return cand
    i = 2
    while True:
        cand = os.path.join(dirpath, f'{basename} ({i}).{ext}')
        if not os.path.exists(cand):
            return cand
        i += 1


# ---------------------------------------------------------------------------
# Enregistreur
# ---------------------------------------------------------------------------

class Recorder:
    """
    Orchestre la capture multi-sources → FLAC multicanal + sidecar channels.json.

    Usage :
        rec = Recorder(sources, out_path)   # sources = sous-ensemble de list_sources()
        rec.start()
        ... rec.elapsed(), rec.levels(), rec.pause()/resume() ...
        rec.stop()                          # finalise le FLAC
        rec.cancel()                        # arrête et supprime le fichier
    """

    def __init__(self, sources: list[dict], out_path: str,
                 backend: str | None = None, rate: int = DEFAULT_RATE,
                 on_pcm=None):
        if not sources:
            raise ValueError('au moins une source est requise')
        self.sources = sources
        self.out_path = out_path
        self.backend = backend or detect_backend()
        self.rate = rate
        self.total_channels = sum(int(s['channels']) for s in sources)
        # Callback optionnel recevant le PCM s16le interleavé (tee → transcription
        # live). Quand fourni, Python est dans le chemin des données.
        self.on_pcm = on_pcm

        self._procs: list[subprocess.Popen] = []
        self._ff: subprocess.Popen | None = None
        self._cap: subprocess.Popen | None = None      # capture.exe (wsl)
        self._pump: threading.Thread | None = None     # tee PCM → ffmpeg / on_pcm
        self._taps: list[subprocess.Popen] = []        # parec (linux)
        self._threads: list[threading.Thread] = []
        self._levels = [0.0] * len(sources)
        self._levels_lock = threading.Lock()
        self._stop_flag = threading.Event()

        self._start_ts = 0.0
        self._end_ts = 0.0
        self._paused = False
        self._pause_started = 0.0
        self._paused_total = 0.0

    # -- carte des canaux (sidecar) -----------------------------------------

    def channel_map(self) -> dict:
        chans, start = [], 0
        for i, s in enumerate(self.sources):
            n = int(s['channels'])
            chans.append({
                'index': i,
                'id': s['id'],
                'name': s['name'],
                'kind': s['kind'],
                'channels': n,
                'channel_start': start,
                'channel_end': start + n,   # exclusif
            })
            start += n
        return {
            'rate': self.rate,
            'codec': 'flac',
            'total_channels': self.total_channels,
            'backend': self.backend,
            'sources': chans,
        }

    def _write_sidecar(self):
        side = os.path.splitext(self.out_path)[0] + '.channels.json'
        try:
            with open(side, 'w', encoding='utf-8') as f:
                json.dump(self.channel_map(), f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    # -- démarrage -----------------------------------------------------------

    def start(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        if self.backend == 'wsl':
            self._start_wsl()
        else:
            self._start_linux()
        self._start_ts = time.monotonic()
        self._write_sidecar()

    def _ffmpeg_pcm_cmd(self) -> list[str]:
        """ffmpeg lisant du PCM s16le interleavé sur stdin → FLAC multicanal."""
        return [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 's16le', '-ar', str(self.rate), '-ac', str(self.total_channels),
            '-i', 'pipe:0',
            '-c:a', 'flac', self.out_path,
        ]

    def _start_wsl(self):
        if not capture_exe_present():
            raise RuntimeError(
                f'binaire de capture introuvable : {CAPTURE_EXE}\n'
                '  → le construire : actions/audio_utils/capture/build.sh '
                '(ou build_capture()).')
        cmd = [CAPTURE_EXE, '--rate', str(self.rate)]
        for s in self.sources:
            cmd += ['--source', s['id']]
        # capture.exe : PCM s16le interleavé sur stdout, "LEVEL i rms" sur stderr
        self._cap = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        if self.on_pcm:
            # Python dans le chemin : tee stdout → ffmpeg.stdin + on_pcm
            self._ff = subprocess.Popen(
                self._ffmpeg_pcm_cmd(), stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._pump = threading.Thread(
                target=self._pump_tee, args=(self._cap.stdout, self._ff.stdin),
                daemon=True)
            self._pump.start()
        else:
            # Pipe direct capture.exe → ffmpeg (pas de Python dans le chemin)
            self._ff = subprocess.Popen(
                self._ffmpeg_pcm_cmd(), stdin=self._cap.stdout,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if self._cap.stdout:
                self._cap.stdout.close()
        self._procs = [self._cap, self._ff]
        t = threading.Thread(target=self._read_levels_wsl, daemon=True)
        t.start()
        self._threads.append(t)

    def _pump_tee(self, src, dst):
        """Lit le PCM de capture.exe, l'écrit vers ffmpeg ET le passe à on_pcm."""
        chunk = self.total_channels * 2 * max(1, self.rate // 50)   # ~20 ms
        # Drainer jusqu'à EOF (arrêt = terminate de capture.exe) pour laisser
        # ffmpeg recevoir tout le flux puis l'EOF et finaliser le FLAC.
        try:
            while True:
                data = src.read(chunk)
                if not data:
                    break
                try:
                    dst.write(data)
                except (OSError, ValueError):
                    break
                if self.on_pcm:
                    try:
                        self.on_pcm(data)
                    except Exception:
                        pass
        finally:
            for f in (dst, src):
                try:
                    f.close()
                except OSError:
                    pass

    def _pump_read(self, src):
        """Linux : lit le PCM que ffmpeg émet sur stdout et le passe à on_pcm."""
        chunk = self.total_channels * 2 * max(1, self.rate // 50)
        # Drainer jusqu'à EOF (arrêt = 'q' envoyé à ffmpeg) pour ne pas bloquer
        # ffmpeg sur l'écriture stdout pendant la finalisation du FLAC.
        try:
            while True:
                data = src.read(chunk)
                if not data:
                    break
                if self.on_pcm:
                    try:
                        self.on_pcm(data)
                    except Exception:
                        pass
        finally:
            try:
                src.close()
            except OSError:
                pass

    def _read_levels_wsl(self):
        if not self._cap or not self._cap.stderr:
            return
        for raw in self._cap.stderr:
            if self._stop_flag.is_set():
                break
            try:
                line = raw.decode('utf-8', 'ignore').strip()
            except Exception:
                continue
            # format attendu : "LEVEL <idx> <rms 0..1>"
            parts = line.split()
            if len(parts) == 3 and parts[0] == 'LEVEL':
                try:
                    idx, val = int(parts[1]), float(parts[2])
                except ValueError:
                    continue
                with self._levels_lock:
                    if 0 <= idx < len(self._levels):
                        self._levels[idx] = max(0.0, min(1.0, val))

    def _start_linux(self):
        # Un seul ffmpeg, N entrées pulse, aresample (les monitors/micros n'ont
        # pas le même taux) → amerge → FLAC. Sortie PCM additionnelle si on_pcm.
        n = len(self.sources)
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y']
        for s in self.sources:
            # -ac avant -i : force le nb de canaux de l'entrée pulse (sinon stéréo
            # par défaut) pour rester cohérent avec channels.json.
            cmd += ['-f', 'pulse', '-ac', str(int(s['channels'])), '-i', s['id']]
        chains = [f'[{i}:a]aresample={self.rate}[r{i}]' for i in range(n)]
        if n == 1:
            merge = '[r0]anull[a]'
        else:
            merge = ''.join(f'[r{i}]' for i in range(n)) + f'amerge=inputs={n}[a]'
        cmd += ['-filter_complex', ';'.join(chains + [merge])]
        cmd += ['-map', '[a]', '-c:a', 'flac', self.out_path]
        if self.on_pcm:
            cmd += ['-map', '[a]', '-f', 's16le', '-ar', str(self.rate),
                    '-ac', str(self.total_channels), 'pipe:1']
        self._ff = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=(subprocess.PIPE if self.on_pcm else subprocess.DEVNULL),
            stderr=subprocess.DEVNULL)
        self._procs = [self._ff]
        if self.on_pcm:
            self._pump = threading.Thread(
                target=self._pump_read, args=(self._ff.stdout,), daemon=True)
            self._pump.start()
        # Vumètres : un tap parec léger par source (capture parallèle autorisée).
        for i, s in enumerate(self.sources):
            tap = subprocess.Popen(
                ['parec', '--device=' + s['id'], '--rate=8000',
                 '--channels=1', '--format=s16le'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
            self._taps.append(tap)
            t = threading.Thread(target=self._read_levels_linux,
                                 args=(i, tap), daemon=True)
            t.start()
            self._threads.append(t)

    def _read_levels_linux(self, idx: int, tap: subprocess.Popen):
        try:
            import numpy as _np
        except Exception:
            _np = None
        chunk = 1600  # 0.2 s @ 8 kHz mono s16
        nbytes = chunk * 2
        while not self._stop_flag.is_set() and tap.stdout:
            buf = tap.stdout.read(nbytes)
            if not buf:
                break
            if _np is not None:
                samples = _np.frombuffer(buf, dtype='<i2').astype('float32')
                rms = float(_np.sqrt(_np.mean(samples * samples))) / 32768.0 if samples.size else 0.0
            else:
                cnt = len(buf) // 2
                if not cnt:
                    continue
                acc = 0
                for v in struct.unpack(f'<{cnt}h', buf[:cnt * 2]):
                    acc += v * v
                rms = math.sqrt(acc / cnt) / 32768.0
            with self._levels_lock:
                self._levels[idx] = max(0.0, min(1.0, rms))

    # -- contrôle -----------------------------------------------------------

    def levels(self) -> list[float]:
        with self._levels_lock:
            return list(self._levels)

    def elapsed(self) -> float:
        if not self._start_ts:
            return 0.0
        if self._end_ts:
            base = self._end_ts
        elif self._paused:
            base = self._pause_started
        else:
            base = time.monotonic()
        return max(0.0, base - self._start_ts - self._paused_total)

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self):
        if self._paused:
            return
        self._paused = True
        self._pause_started = time.monotonic()
        for p in self._procs + self._taps:
            self._signal(p, 'STOP')

    def resume(self):
        if not self._paused:
            return
        self._paused_total += time.monotonic() - self._pause_started
        self._paused = False
        for p in self._procs + self._taps:
            self._signal(p, 'CONT')

    @staticmethod
    def _signal(proc: subprocess.Popen | None, sig: str):
        if not proc or proc.poll() is not None:
            return
        try:
            import signal as _sig
            proc.send_signal(getattr(_sig, 'SIG' + sig))
        except Exception:
            pass

    def is_alive(self) -> bool:
        return self._ff is not None and self._ff.poll() is None

    def stop(self) -> bool:
        """Arrêt propre : ferme la source, laisse ffmpeg finaliser le FLAC."""
        if self._paused:
            self.resume()
        if not self._end_ts:
            self._end_ts = time.monotonic()
        self._stop_flag.set()

        if self.backend == 'wsl':
            # Terminer capture.exe → EOF sur stdin de ffmpeg → flush du FLAC.
            self._terminate(self._cap)
        else:
            for tap in self._taps:
                self._terminate(tap)
            # 'q' sur stdin de ffmpeg = arrêt propre.
            if self._ff and self._ff.stdin:
                try:
                    self._ff.stdin.write(b'q')
                    self._ff.stdin.flush()
                    self._ff.stdin.close()
                except OSError:
                    pass

        rc = 0
        if self._ff:
            try:
                rc = self._ff.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._terminate(self._ff)
                rc = self._ff.wait()
        # Le tee PCM (on_pcm) tourne sur self._pump : on l'attend pour garantir
        # que toutes les données ont été poussées vers la transcription avant
        # que l'appelant ne finalise (finalize()).
        if self._pump:
            self._pump.join(timeout=5)
        self._write_sidecar()
        return rc == 0 and os.path.exists(self.out_path)

    def cancel(self):
        """Arrêt + suppression du fichier (et du sidecar)."""
        if not self._end_ts:
            self._end_ts = time.monotonic()
        self._stop_flag.set()
        for p in [self._cap, self._ff] + self._taps:
            self._terminate(p)
        for p in [self._cap, self._ff] + self._taps:
            if p:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if self._pump:
            self._pump.join(timeout=5)
        for path in (self.out_path,
                     os.path.splitext(self.out_path)[0] + '.channels.json'):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    @staticmethod
    def _terminate(proc: subprocess.Popen | None):
        if not proc or proc.poll() is not None:
            return
        try:
            proc.terminate()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CLI de diagnostic (utile sur le laptop : `python3 recorder.py --list`)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    if '--list' in sys.argv:
        be = detect_backend()
        print(f'backend: {be}')
        for s in list_sources(be):
            print(f"  [{s['kind']:6}] {s['channels']}ch  {s['name']}  ({s['id']})")
    else:
        print('usage: recorder.py --list')
