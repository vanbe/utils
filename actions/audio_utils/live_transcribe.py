#!/usr/bin/env python3
"""
live_transcribe.py — transcription live par canal, par-dessus l'enregistreur.

Reçoit le flux PCM s16le interleavé de l'enregistreur (`Recorder(on_pcm=…)`),
désentrelace chaque source (via `channels.json`), downmix mono + resample 16 kHz,
détecte les énoncés par énergie (VAD-gate), et les transcrit avec **un seul
modèle faster-whisper partagé** (un worker, canaux traités séquentiellement —
voir le choix d'archi : 6 Go VRAM ne tient pas 2 modèles, et un GPU décode de
toute façon en série). Chaque segment finalisé est étiqueté par sa source
(micro → « Moi », sortie → « Système ») et poussé via `on_segment`. À l'arrêt,
un `.srt` aligné sur le FLAC est écrit.

Réutilise la logique de `transcribe_audio.py` (MODEL_NAME_MAP, device/
compute_type, fallback OOM, `model.transcribe(vad_filter=True)`). torch et
faster_whisper sont importés paresseusement (lourds, GPU laptop uniquement).
"""

import os
import queue
import threading
import time

import numpy as np

# Code Whisper partagé avec transcribe_audio.py (table modèles, load, SRT/MD)
# et labels par canal (mutualisés avec transcribe_channels.py → mêmes libellés).
from whisper_common import load_whisper_model, write_srt, write_md, channel_labels

# Charge les réglages .env (LIVE_TRANSCRIBE_*) — même source que transcribe_audio.py
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_MODELS_DIR = os.path.join(_PROJECT_ROOT, 'vendor', 'models', 'whisper')
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))
except Exception:
    pass

# Paramètres de gating (overridables via .env LIVE_TRANSCRIBE_*)
_SILENCE_HANG_S = 0.6       # silence avant de clore un énoncé
_MAX_UTTER_S    = 15.0      # longueur max d'un énoncé (flux continu type musique)
_MIN_UTTER_S    = 0.4       # ignore les micro-bruits plus courts
_RMS_THRESH     = 0.012     # seuil d'énergie voix/silence (0..1)


def _env(name: str, default: str = '') -> str:
    return os.environ.get(name, default).strip()


class _SourceGate:
    """État VAD/énoncé d'une source."""
    def __init__(self):
        self.parts = []          # list[np.ndarray] (mono, native rate)
        self.start_sec = 0.0
        self.in_speech = False
        self.silence_s = 0.0
        self.length_s = 0.0
        self.last_interim_len = 0.0   # longueur au dernier aperçu interim


class LiveTranscriber:
    def __init__(self, channel_map: dict, on_segment, srt_path: str,
                 language: str | None = 'fr', model_name: str | None = None):
        self.rate = int(channel_map['rate'])
        self.total_channels = int(channel_map['total_channels'])
        self.sources = channel_map['sources']
        self.labels = channel_labels(self.sources)
        self.on_segment = on_segment
        self.srt_path = srt_path
        self.md_path = os.path.splitext(srt_path)[0] + '.md'   # version sans timings
        self.language = (None if (language or '').lower() in ('auto', '')
                         else language)

        self.model_name = (model_name or _env('LIVE_TRANSCRIBE_MODEL', 'turbo'))
        self._sil_hang = float(_env('LIVE_TRANSCRIBE_SILENCE_S', str(_SILENCE_HANG_S)) or _SILENCE_HANG_S)
        self._max_utter = float(_env('LIVE_TRANSCRIBE_WINDOW_SEC', str(_MAX_UTTER_S)) or _MAX_UTTER_S)
        # Real-time : aperçus "interim" pendant qu'on parle (0 = désactivé).
        self._interim_sec = float(_env('LIVE_TRANSCRIBE_INTERIM_SEC', '2') or 2)

        self._gates = {s['index']: _SourceGate() for s in self.sources}
        self._interim_pending = set()  # idx avec un interim déjà en file/cours (≤1)
        self._residual = b''
        self._pos_frames = 0           # position globale (frames consommées)
        self._q: queue.Queue = queue.Queue()
        self._segments = []            # [{source_idx,label,start,end,text}]
        self._seg_lock = threading.Lock()
        self._model = None
        self._worker = None
        self._stop = threading.Event()
        self.status = 'init'           # init|loading|ready|error
        self.error = ''

    # -- chargement modèle (reprend transcribe_audio.py) ---------------------

    def _load_model(self):
        # Chargement + fallback OOM mutualisés avec transcribe_audio.py.
        # download_root partagé → mêmes poids que la transcription différée.
        threads_env = _env('LIVE_TRANSCRIBE_CPU_THREADS')
        cpu_threads = int(threads_env) if threads_env.isdigit() else max(1, (os.cpu_count() or 2) - 1)
        try:
            self._model, self.model_name, _, _ = load_whisper_model(
                self.model_name,
                device=_env('LIVE_TRANSCRIBE_DEVICE'),
                compute_type=_env('LIVE_TRANSCRIBE_COMPUTE_TYPE'),
                download_root=_MODELS_DIR, cpu_threads=cpu_threads)
        except Exception as e:
            self.status = 'error'
            self.error = str(e)
            raise

    def start(self):
        """Charge le modèle (bloquant), le préchauffe, puis lance le worker."""
        self.status = 'loading'
        self._load_model()
        # Warmup : 1ère inférence (JIT kernels CUDA) hors temps réel
        try:
            list(self._model.transcribe(np.zeros(16000, dtype=np.float32),
                 language=self.language, beam_size=1, vad_filter=False)[0])
        except Exception:
            pass
        self.status = 'ready'
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    # -- entrée audio (appelée par Recorder.on_pcm) -------------------------

    def feed_bytes(self, data: bytes):
        if not data:
            return
        self._residual += data
        frame_bytes = self.total_channels * 2
        nframes = len(self._residual) // frame_bytes
        if nframes == 0:
            return
        usable = nframes * frame_bytes
        block = np.frombuffer(self._residual[:usable], dtype='<i2')
        self._residual = self._residual[usable:]
        block = block.reshape(-1, self.total_channels).astype(np.float32) / 32768.0

        block_start_sec = self._pos_frames / self.rate
        block_dur = nframes / self.rate

        for s in self.sources:
            mono = block[:, s['channel_start']:s['channel_end']].mean(axis=1)
            self._gate(s['index'], mono, block_start_sec, block_dur)

        self._pos_frames += nframes

    def _gate(self, idx: int, mono: np.ndarray, block_start: float, block_dur: float):
        g = self._gates[idx]
        rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
        voiced = rms > _RMS_THRESH

        if voiced:
            if not g.in_speech:
                g.in_speech = True
                g.start_sec = block_start
                g.parts = []
                g.length_s = 0.0
            g.parts.append(mono)
            g.length_s += block_dur
            g.silence_s = 0.0
        elif g.in_speech:
            g.parts.append(mono)               # garde un peu de traîne
            g.length_s += block_dur
            g.silence_s += block_dur
            if g.silence_s >= self._sil_hang:
                self._flush(idx)
                return

        if g.in_speech and g.length_s >= self._max_utter:
            self._flush(idx)
            return

        # Aperçu temps réel : transcrit l'énoncé en cours (≤1 interim en vol/source)
        if (g.in_speech and self._interim_sec > 0
                and g.length_s >= 1.0
                and (g.length_s - g.last_interim_len) >= self._interim_sec
                and idx not in self._interim_pending):
            self._interim_pending.add(idx)
            g.last_interim_len = g.length_s
            self._q.put((idx, self._to_16k(np.concatenate(g.parts)), g.start_sec, True))

    def _flush(self, idx: int):
        g = self._gates[idx]
        if not g.parts:
            g.in_speech = False
            return
        audio = np.concatenate(g.parts)
        start = g.start_sec
        dur = len(audio) / self.rate
        g.parts = []
        g.in_speech = False
        g.silence_s = 0.0
        g.length_s = 0.0
        g.last_interim_len = 0.0
        if dur >= _MIN_UTTER_S:
            self._q.put((idx, self._to_16k(audio), start, False))

    def _to_16k(self, x: np.ndarray) -> np.ndarray:
        if self.rate == 16000 or x.size == 0:
            return x.astype(np.float32)
        n = int(round(len(x) * 16000 / self.rate))
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        idx = np.linspace(0, len(x) - 1, n)
        return np.interp(idx, np.arange(len(x)), x).astype(np.float32)

    # -- worker de transcription -------------------------------------------

    def _kind(self, idx: int) -> str:
        return next((s['kind'] for s in self.sources if s['index'] == idx), 'input')

    def _emit(self, idx, start, end, text, interim):
        rec = {
            'source_idx': idx, 'label': self.labels.get(idx, '?'),
            'kind': self._kind(idx), 'start': start, 'end': end,
            'text': text, 'interim': interim,
        }
        if not interim:
            with self._seg_lock:
                self._segments.append(rec)
        if self.on_segment:
            try:
                self.on_segment(rec)
            except Exception:
                pass

    def _decode(self, audio):
        # On fait DÉJÀ le VAD (gate par énergie) en amont → on neutralise le
        # second-guessing de faster-whisper qui, sur un énoncé court déjà
        # découpé, rejette parfois TOUT le chunk (0 segment → canal « perdu ») :
        #  - vad_filter=False        : pas de re-segmentation Silero,
        #  - no_speech_threshold=1.0 : ne jamais classer le chunk « non-parole »
        #                              (c'était LA cause du canal perdu).
        segs, _ = self._model.transcribe(
            audio, language=self.language, beam_size=1,
            vad_filter=False, condition_on_previous_text=False,
            no_speech_threshold=1.0)
        return [s for s in segs if (s.text or '').strip()]

    def _run_worker(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            idx, audio, start_sec, is_interim = item
            try:
                segs = self._decode(audio)
                if not segs and not is_interim:
                    segs = self._decode(audio)        # retry défensif (jamais vide à tort)
                if is_interim:
                    txt = ' '.join(s.text.strip() for s in segs)
                    if txt:
                        self._emit(idx, start_sec, start_sec + len(audio) / 16000.0,
                                   txt, True)
                else:
                    for seg in segs:
                        self._emit(idx, start_sec + seg.start,
                                   start_sec + seg.end, seg.text.strip(), False)
            except Exception:
                pass
            finally:
                if is_interim:
                    self._interim_pending.discard(idx)

    # -- arrêt + écriture .srt ---------------------------------------------

    def finalize(self):
        self._stop.set()
        for idx in list(self._gates):           # vide les énoncés en cours
            if self._gates[idx].in_speech:
                self._flush(idx)
        self._q.put(None)
        if self._worker:
            self._worker.join(timeout=120)
        self._write_outputs()

    def _write_outputs(self):
        """Écrit le .srt (avec timings) ET le .md (sans timings) — les deux par défaut."""
        with self._seg_lock:
            segs = list(self._segments)
        if not segs:
            return
        try:
            write_srt(self.srt_path, segs)
            write_md(self.md_path, segs)
        except OSError:
            pass

    def segments_snapshot(self) -> list:
        with self._seg_lock:
            return list(self._segments)
