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
from whisper_common import (load_whisper_model, write_srt, write_md,
                            channel_labels, is_hallucination, is_echo_duplicate,
                            _norm_phrase)
import aec as _aec
import diarize_online as _diar   # import léger (pyannote chargé paresseusement)

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


# Cache du modèle Whisper PARTAGÉ entre instances de LiveTranscriber : permet de
# lancer plusieurs sessions d'affilée (stop & nouvelle session) SANS recharger le
# modèle ni reprendre la VRAM. Clé = (nom, device, compute_type). Les sessions
# sont séquentielles (une à la fois) → un même WhisperModel est réutilisable.
_MODEL_CACHE: dict = {}


def unload_models() -> bool:
    """Décharge le(s) modèle(s) Whisper en cache et libère la VRAM. À appeler en
    QUITTANT la fonctionnalité d'enregistrement (« Terminer ») : sinon le modèle
    reste en mémoire GPU pour rien (laptop sur batterie). Renvoie True si quelque
    chose a effectivement été déchargé. ⚠ L'appelant doit aussi lâcher ses propres
    références au modèle (ex. `transcriber = None`) pour que le GC libère la VRAM."""
    had = bool(_MODEL_CACHE)
    _MODEL_CACHE.clear()
    try:
        had = _diar.unload_models() or had     # modèle d'embedding de diarisation
    except Exception:
        pass
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return had


class _SourceGate:
    """État VAD/énoncé d'une source."""
    def __init__(self):
        self.parts = []          # list[np.ndarray] (mono, native rate)
        self.start_sec = 0.0
        self.in_speech = False
        self.silence_s = 0.0
        self.length_s = 0.0
        self.last_interim_len = 0.0   # longueur au dernier aperçu interim


class _Ring:
    """File de chunks mono (float32) bornée à ~`keep` échantillons, indexée en
    POSITION ABSOLUE (échantillons depuis le début). `push()` ajoute un bloc et
    purge les plus vieux ; `slice(a, b)` reconstruit la fenêtre absolue [a, b).

    Sert au garde anti-écho : on garde un historique court de la référence (sortie
    HP) et du micro BRUT pour pouvoir, à la clôture d'un énoncé, ré-extraire les
    deux signaux sur sa fenêtre et mesurer la corrélation d'écho. Bornée à quelques
    secondes (un énoncé fait ≤ `_max_utter`), donc empreinte mémoire minime."""

    def __init__(self, keep: int):
        self.keep = int(keep)
        self.chunks: list = []
        self.start = 0                 # index absolu du 1er échantillon retenu
        self.end = 0                   # index absolu juste après le dernier

    def push(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return
        self.chunks.append(x)
        self.end += x.size
        # Purge les chunks les plus anciens tant qu'on dépasse `keep` sans eux.
        while self.chunks and (self.end - self.start) - self.chunks[0].size >= self.keep:
            self.start += self.chunks.pop(0).size

    def slice(self, a: int, b: int) -> np.ndarray:
        a = max(int(a), self.start)
        b = min(int(b), self.end)
        if b <= a or not self.chunks:
            return np.zeros(0, dtype=np.float32)
        buf = self.chunks[0] if len(self.chunks) == 1 else np.concatenate(self.chunks)
        return buf[a - self.start:b - self.start]


class LiveTranscriber:
    def __init__(self, channel_map: dict, on_segment, srt_path: str,
                 language: str | None = 'fr', model_name: str | None = None,
                 aec: bool = False, diarize: bool = False):
        self.rate = int(channel_map['rate'])
        self.total_channels = int(channel_map['total_channels'])
        self.sources = channel_map['sources']
        self.labels = channel_labels(self.sources)
        self.on_segment = on_segment
        self.srt_path = srt_path
        self.md_path = os.path.splitext(srt_path)[0] + '.md'   # version sans timings
        self.language = (None if (language or '').lower() in ('auto', '')
                         else language)

        # --- AEC live : retire du micro la sortie système recaptée (HP) -------
        # Référence = mix de TOUTES les colonnes des sources de sortie (ce qui
        # part aux HP). Un annuleur d'écho stateful par canal MICRO, au débit
        # natif. Sans sortie, l'option est sans objet → désactivée.
        self._out_cols = [c for s in self.sources if s['kind'] == 'output'
                          for c in range(s['channel_start'], s['channel_end'])]
        self.aec = bool(aec and self._out_cols
                        and any(s['kind'] == 'input' for s in self.sources))
        self._cancellers = ({s['index']: _aec.EchoCanceller(self.rate)
                             for s in self.sources if s['kind'] == 'input'}
                            if self.aec else {})
        # Position (échantillons déjà transmis au gate) PAR source : l'AEC peut
        # restituer un peu moins d'échantillons qu'il n'en reçoit (bufferisation
        # d'un hop), donc chaque source avance à son propre rythme.
        self._src_pos = {s['index']: 0 for s in self.sources}

        # --- Diarisation LIVE des intervenants du canal SYSTÈME (sortie) -------
        # Identifie « Système · P1/P2/P3 » à la volée (embedding par énoncé sur
        # CPU + clustering en ligne). Sans sortie, sans objet. Le micro (« Moi »)
        # n'est pas diarisé (un seul locuteur par micro). UN diariseur PARTAGÉ →
        # numéros de personnes cohérents sur tous les canaux de sortie.
        self.diarize = bool(diarize and any(s['kind'] == 'output' for s in self.sources))
        self._diarizer = None
        if self.diarize:
            thr = float(_env('LIVE_DIARIZE_THRESHOLD', '0.5') or 0.5)
            maxsp = int(_env('LIVE_DIARIZE_MAX_SPEAKERS', '8') or 8)
            upd = _env('LIVE_DIARIZE_UPDATE_MIN')
            mrg = _env('LIVE_DIARIZE_MARGIN')
            self._diarizer = _diar.OnlineDiarizer(
                threshold=thr, max_speakers=maxsp, device='cpu',
                update_min=float(upd) if upd else None,
                margin=float(mrg) if mrg else 0.1)

        self.model_name = (model_name or _env('LIVE_TRANSCRIBE_MODEL', 'turbo'))
        self._sil_hang = float(_env('LIVE_TRANSCRIBE_SILENCE_S', str(_SILENCE_HANG_S)) or _SILENCE_HANG_S)
        self._max_utter = float(_env('LIVE_TRANSCRIBE_WINDOW_SEC', str(_MAX_UTTER_S)) or _MAX_UTTER_S)
        # Real-time : aperçus "interim" pendant qu'on parle (0 = désactivé).
        self._interim_sec = float(_env('LIVE_TRANSCRIBE_INTERIM_SEC', '2') or 2)

        # --- Garde anti-écho au niveau ÉNONCÉ (backstop de l'AEC) -------------
        # Même AEC active, un RÉSIDU d'écho HP (distorsion du haut-parleur, réverb,
        # double-talk) peut rester transcrit sur « Moi » alors qu'on n'a rien dit
        # (« je n'ai rien dit » → faux énoncé Moi). Pour CHAQUE énoncé MICRO finalisé,
        # on mesure la part de son énergie qui s'explique par la référence (ce qui
        # est sorti aux HP) via la COHÉRENCE spectrale `aec.echo_coherence` (≈ écho/
        # (écho+voix)). coh ≥ seuil ⇒ l'énoncé est de l'écho recapté (la conférence
        # figure déjà, PROPRE, sur « Système ») → on le JETTE. Indépendant du flag
        # `aec`, car c'est une DÉTECTION (pas une annulation) :
        #   • Sûr au casque : la voix réelle est DÉCORRÉLÉE de la sortie → coh≈0,
        #     jamais jetée ; le gate ne se déclenche QUE s'il y a vrai couplage
        #     acoustique (HP). Donc activable par défaut dès qu'on capte micro+sortie.
        #   • Sûr en double-talk : la voix proche ajoute de l'énergie décorrélée →
        #     coh sous le seuil dès que la voix ≳ l'écho → énoncé gardé.
        # On utilise la cohérence (et NON `cancel_array`+ERLE) car le filtre adaptatif
        # ne converge pas en 1–3 s d'audio coloré ; la cohérence est convergée par
        # construction → fiable énoncé par énoncé. La décision est faite dans le
        # WORKER (pas le thread audio) → on évite même l'inférence Whisper sur l'écho
        # (gain GPU). On garde un court historique du micro BRUT (avant AEC) + de la
        # référence pour ré-extraire la fenêtre exacte de l'énoncé.
        self.echo_gate = bool(
            self._out_cols
            and any(s['kind'] == 'input' for s in self.sources)
            and _env('LIVE_ECHO_GATE', '1').lower() not in ('0', 'false', 'no', 'off'))
        self._echo_coh = float(_env('LIVE_ECHO_GATE_COH', '0.65') or 0.65)
        _keep = int((self._max_utter + 5.0) * self.rate)
        self._ref_ring = _Ring(_keep) if self.echo_gate else None
        self._mic_rings = ({s['index']: _Ring(_keep)
                            for s in self.sources if s['kind'] == 'input'}
                           if self.echo_gate else {})

        self._gates = {s['index']: _SourceGate() for s in self.sources}
        self._interim_pending = set()  # idx avec un interim déjà en file/cours (≤1)
        self._paused = False           # en pause : feed_bytes ignore l'audio (pas de GPU)
        self._residual = b''
        self._q: queue.Queue = queue.Queue()
        self._segments = []            # [{source_idx,label,start,end,text}]
        self._recent = []              # (start, label, norm) pour le dé-doublonnage d'écho
        self._seg_lock = threading.Lock()
        self._model = None
        self.model_loaded_from_cache = False
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
        device = _env('LIVE_TRANSCRIBE_DEVICE')
        compute = _env('LIVE_TRANSCRIBE_COMPUTE_TYPE')
        key = (self.model_name, device, compute)
        cached = _MODEL_CACHE.get(key)
        if cached is not None:            # déjà chargé (session précédente) → réutilise
            self._model = cached
            self.model_loaded_from_cache = True
            return
        try:
            self._model, self.model_name, dev, ct = load_whisper_model(
                self.model_name, device=device, compute_type=compute,
                download_root=_MODELS_DIR, cpu_threads=cpu_threads)
            # Mémorise sous la clé demandée ET la clé résolue (fallback éventuel).
            _MODEL_CACHE[key] = self._model
            _MODEL_CACHE[(self.model_name, dev, ct)] = self._model
        except Exception as e:
            self.status = 'error'
            self.error = str(e)
            raise

    def start(self):
        """Charge le modèle (bloquant), le préchauffe, puis lance le worker."""
        self.status = 'loading'
        self._load_model()
        # Warmup : 1ère inférence (JIT kernels CUDA) hors temps réel — inutile si
        # le modèle vient du cache (déjà préchauffé par une session précédente).
        if not self.model_loaded_from_cache:
            try:
                list(self._model.transcribe(np.zeros(16000, dtype=np.float32),
                     language=self.language, beam_size=1, vad_filter=False)[0])
            except Exception:
                pass
        # Précharge le modèle d'embedding de diarisation (CPU) hors temps réel.
        if self._diarizer is not None:
            self._diarizer.warmup()
        self.status = 'ready'
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    # -- entrée audio (appelée par Recorder.on_pcm) -------------------------

    def set_paused(self, flag: bool):
        """Pause/reprise de l'ALIMENTATION de Whisper. En pause, `feed_bytes`
        jette l'audio entrant → plus rien ne part au GPU (économie batterie),
        mais le modèle reste chargé. On repart proprement à la reprise."""
        flag = bool(flag)
        if flag == self._paused:
            return
        self._paused = flag
        if flag:
            # L'énoncé EN COURS (parole captée AVANT la pause) doit être transcrit,
            # pas perdu : on le FLUSH vers Whisper avant de geler. `_flush` enfile
            # l'audio puis réinitialise le gate ; les gates sans parole sont juste
            # remis à zéro. Le worker traite la file même en pause (feed_bytes,
            # lui, ignore l'audio ENTRANT). On vide aussi le tampon d'octets.
            self._residual = b''
            for idx in list(self._gates):
                self._flush(idx)
            self._interim_pending.clear()

    def feed_bytes(self, data: bytes):
        if self._paused or not data:        # en pause → on n'alimente PAS Whisper
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

        # Référence d'écho de ce bloc (mix des colonnes de sortie). Nécessaire à
        # l'AEC ET au garde anti-écho par énoncé → calculée si l'un OU l'autre.
        ref_mono = (block[:, self._out_cols].mean(axis=1)
                    if (self.aec or self.echo_gate) and self._out_cols else None)

        # Historique BRUT (réf + micros AVANT AEC) pour le garde anti-écho : la
        # décision compare le micro brut à la référence, indépendamment de l'AEC
        # (qui, elle, ne nettoie que le signal envoyé à Whisper). Avancée en débit
        # natif (par bloc), un léger décalage AEC est absorbé par la marge + GCC-PHAT.
        if self.echo_gate and ref_mono is not None:
            self._ref_ring.push(ref_mono)
            for s in self.sources:
                if s['kind'] == 'input':
                    self._mic_rings[s['index']].push(
                        block[:, s['channel_start']:s['channel_end']].mean(axis=1))

        for s in self.sources:
            idx = s['index']
            mono = block[:, s['channel_start']:s['channel_end']].mean(axis=1)
            ec = self._cancellers.get(idx)
            if ec is not None and ref_mono is not None:
                mono = ec.process(mono, ref_mono)          # micro nettoyé (longueur ≤ bloc)
            if mono.size == 0:
                continue
            start = self._src_pos[idx] / self.rate
            self._gate(idx, mono, start, mono.size / self.rate)
            self._src_pos[idx] += mono.size

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
            self._q.put((idx, self._to_16k(np.concatenate(g.parts)), g.start_sec, True, None))

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
            probe = self._echo_probe(idx, start, len(audio))
            self._q.put((idx, self._to_16k(audio), start, False, probe))

    def _echo_probe(self, idx: int, start: float, n_native: int):
        """Snapshot (micro BRUT, référence) sur la fenêtre de l'énoncé, pour le
        garde anti-écho évalué dans le worker. None si gate inactif / hors micro /
        fenêtre trop courte. Lu ici (thread audio) tant que les rings sont à jour ;
        le worker ne touche QUE les arrays renvoyés (pas les rings) → pas de course."""
        if not self.echo_gate or self._kind(idx) != 'input':
            return None
        ring = self._mic_rings.get(idx)
        if ring is None:
            return None
        base = int(round(start * self.rate))
        m = int(0.3 * self.rate)                        # marge (latence d'écho HP)
        a, b = base - m, base + n_native + m
        mic = ring.slice(a, b)
        ref = self._ref_ring.slice(a, b)
        if mic.size < self.rate // 5 or ref.size < self.rate // 5:
            return None
        return (mic, ref)

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

    def _emit(self, idx, start, end, text, interim, label=None):
        if label is None:
            label = self.labels.get(idx, '?')
        # Dé-doublonnage d'écho (segments FINAUX seulement) : une phrase courte
        # répétée sur le même canal, ou la MÊME phrase quasi simultanée sur les
        # deux canaux (écho/résidu), est presque toujours une hallucination.
        if not interim:
            if is_echo_duplicate(text, label, start, self._recent):
                return
            now = start
            self._recent.append((now, label, _norm_phrase(text)))
            self._recent = [r for r in self._recent if abs(now - r[0]) <= 6.0]

        rec = {
            'source_idx': idx, 'label': label,
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
        # GARDE-FOU ACOUSTIQUE (Silero VAD) : le gate d'énergie en amont laisse
        # passer le bruit NON-VOCAL fort (frappes clavier, clics) que Whisper
        # transcrirait en mots inventés (« and », « Okay », « Bye »). vad_filter
        # rejette ces chunks sans parole réelle. Paramètres permissifs pour ne
        # PAS perdre un énoncé court réel (seuil 0.5, min_speech 200 ms). On garde
        # no_speech_threshold=1.0 : sur une région que Silero a VALIDÉE comme
        # parole, on ne re-jette pas le chunk (évite le « canal perdu »).
        segs, _ = self._model.transcribe(
            audio, language=self.language, beam_size=1,
            vad_filter=True,
            vad_parameters=dict(threshold=0.5, min_speech_duration_ms=200,
                                min_silence_duration_ms=300),
            condition_on_previous_text=False,
            no_speech_threshold=1.0)
        # 2ᵉ ligne : filtre texte (tics de sous-titres, mots-outils isolés, résidu
        # d'écho via no_speech_prob/avg_logprob — cf. whisper_common).
        return [s for s in segs
                if (s.text or '').strip()
                and not is_hallucination(s.text, getattr(s, 'avg_logprob', 0.0),
                                         getattr(s, 'no_speech_prob', 0.0))]

    def _is_echo(self, mic_native: np.ndarray, ref_native: np.ndarray) -> bool:
        """True si `mic_native` est DOMINÉ par l'écho de `ref_native` (la sortie HP
        recaptée par le micro). On mesure la COHÉRENCE spectrale micro↔réf (part de
        l'énergie micro explicable linéairement par la sortie, après alignement du
        retard) : coh ≥ seuil ⇒ l'énergie vient de la sortie ⇒ écho recapté, pas la
        voix. La vraie voix proche est décorrélée de la sortie → coh≈0. Convergé par
        construction → fiable sur un énoncé court (cf. `aec.echo_coherence`)."""
        try:
            if float(np.dot(ref_native, ref_native)) <= 1e-7:   # sortie silencieuse → pas d'écho
                return False
            if float(np.dot(mic_native, mic_native)) <= 1e-9:
                return False
            return _aec.echo_coherence(mic_native, ref_native, self.rate) >= self._echo_coh
        except Exception:
            return False

    def _run_worker(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            idx, audio, start_sec, is_interim, probe = item
            # Garde anti-écho : l'énoncé MICRO est-il de l'écho HP recapté ? Si oui,
            # on l'écarte AVANT Whisper (ni inférence GPU, ni faux « Moi »).
            if probe is not None and self._is_echo(*probe):
                continue
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
                    # Diarisation : UN embedding par énoncé (pas par segment) sur
                    # les canaux SORTIE → « Système · P1/P2/P3 ». audio = 16 kHz.
                    spk_label = None
                    if (self._diarizer is not None and segs
                            and self._kind(idx) == 'output'):
                        spk = self._diarizer.assign(audio)
                        if spk >= 0:
                            base = self.labels.get(idx, 'Système')
                            spk_label = f'{base} · P{spk + 1}'
                    for seg in segs:
                        self._emit(idx, start_sec + seg.start,
                                   start_sec + seg.end, seg.text.strip(), False,
                                   label=spk_label)
            except Exception:
                pass
            finally:
                if is_interim:
                    self._interim_pending.discard(idx)

    # -- arrêt + écriture .srt ---------------------------------------------

    def finalize(self):
        self._stop.set()
        # Vide la traîne bufferisée par les annuleurs d'écho (dernier hop partiel)
        # vers le gate, pour ne perdre aucun échantillon nettoyé.
        for idx, ec in self._cancellers.items():
            tail = ec.flush()
            if tail.size:
                self._gate(idx, tail, self._src_pos[idx] / self.rate,
                           tail.size / self.rate)
                self._src_pos[idx] += tail.size
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
