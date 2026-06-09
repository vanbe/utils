#!/usr/bin/env python3
"""
aec.py — annulation d'écho acoustique (AEC) par filtre adaptatif, mutualisée
entre la transcription DIFFÉRÉE par canal (`transcribe_channels.py`) et le LIVE
(`live_transcribe.py`).

Problème résolu : quand la sortie système est jouée sur des haut-parleurs, le
micro la recapte. La conférence (canal « Système ») se retrouve, atténuée et
retardée, dans le canal micro (« Moi ») → doublons / mauvaise attribution dans
le transcript. Comme on possède le **signal de référence propre** (le canal
loopback = exactement ce qui est sorti), on peut le soustraire du micro : c'est
de l'annulation d'écho classique avec référence connue.

Méthode : filtre adaptatif FIR **block-NLMS** (Normalized Least Mean Squares),
calculé dans le domaine fréquentiel (overlap-save) pour la vitesse — 100 % numpy,
aucune dépendance. Le filtre `w` (longueur L) modélise le trajet
haut-parleur→air→micro (retard + réverbération + coloration) ; à chaque bloc on
estime l'écho `y = w * ref`, on le retire (`e = mic - y`), et on adapte `w` dans
le sens qui réduit `e`. Un **détecteur de double-talk** gèle l'adaptation quand
l'utilisateur parle en même temps (sinon le filtre se mettrait à annuler la voix
proche). Pour le différé, un **alignement de délai par GCC-PHAT** recadre la
référence avant le filtrage (convergence plus rapide, filtre plus court).

API :
  EchoCanceller(sr, ...).process(mic, ref) -> cleaned   # streaming (live + différé)
  estimate_delay(mic, ref, sr)             -> delay_samples (ref vs mic)
  cancel_array(mic, ref, sr, ...)          -> cleaned    # différé, avec alignement

`mic`/`ref` : np.ndarray 1-D float (mono, MÊME fréquence d'échantillonnage et
MÊME longueur, déjà alignés à l'échantillon — ce qui est le cas dans un FLAC
multicanal). `ref` = mix (somme/moyenne) de TOUTES les sorties si plusieurs.
"""

from __future__ import annotations

import numpy as np


def _next_pow2(n: int) -> int:
    return 1 << (max(1, int(n) - 1)).bit_length()


class EchoCanceller:
    """Filtre adaptatif **FDAF contraint** (Frequency-domain Adaptive Filter,
    overlap-save) — calcul de l'écho et du gradient par FFT, mise à jour NLMS à
    **normalisation scalaire** (denom ≈ L·E_bloc). C'est cette normalisation par
    la longueur du filtre qui rend le pas stable : une normalisation par bin se
    met mal à l'échelle ici et fait diverger le filtre.

    Le filtre `W` (longueur effective L taps) vit dans le domaine fréquentiel et
    persiste entre les appels à `process()`. On peut donc nourrir l'objet en
    streaming (live, bloc par bloc) ou sur un canal entier (différé). À chaque
    hop : on estime l'écho `y = W·X` (conv linéaire valide par overlap-save), on
    le retire (`e = mic − y`), et on adapte `W` dans le sens qui réduit `e`,
    proportionnellement à la corrélation réf/erreur.

    `process()` conserve le nombre d'échantillons (sortie alignée sur l'entrée) ;
    les échantillons en attente d'un hop complet sont bufferisés et restitués par
    `flush()` (à appeler à la fin pour vider la traîne).
    """

    def __init__(self, sr: int, filter_ms: float = 150.0, hop: int = 1024,
                 mu: float = 1.0, eps: float = 1e-6, lam: float = 0.9,
                 dtd: bool = True, dtd_ratio: float = 3.0, dtd_hang_ms: float = 150.0):
        self.sr = int(sr)
        self.L = max(64, int(self.sr * filter_ms / 1000.0))   # longueur du filtre (taps)
        self.H = max(256, int(hop))                            # hop (échantillons par pas)
        self.N = _next_pow2(self.L + self.H)                   # taille FFT (overlap-save)
        self.mu = float(mu)                                    # pas NLMS (0<mu<~1.5 stable)
        self.eps = float(eps)
        self.lam = float(lam)                                  # lissage de l'énergie de réf
        self.dtd = bool(dtd)
        self.dtd_ratio = float(dtd_ratio)                      # e²/y² (après convergence) → gel
        self._dtd_hang = int(self.sr * dtd_hang_ms / 1000.0)

        nb = self.N // 2 + 1
        self.W = np.zeros(nb, dtype=np.complex128)             # filtre (domaine fréquentiel)
        self._rb = np.zeros(self.N, dtype=np.float64)          # buffer glissant de référence (N)
        self._pw = 0.0                                         # énergie de réf du bloc (lissée, scalaire)
        self._started = False
        self._freeze = 0
        self._mic_buf = np.zeros(0, dtype=np.float64)          # entrées micro en attente d'un hop
        self._ref_buf = np.zeros(0, dtype=np.float64)

    # ------------------------------------------------------------------ public
    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Retourne `mic` débarrassé de l'écho corrélé à `ref`. La sortie suit
        l'entrée échantillon par échantillon ; un reliquat < hop est gardé pour
        l'appel suivant (vidé par `flush()`)."""
        mic = np.asarray(mic, dtype=np.float64).ravel()
        ref = np.asarray(ref, dtype=np.float64).ravel()
        n = min(mic.size, ref.size)
        self._mic_buf = np.concatenate([self._mic_buf, mic[:n]])
        self._ref_buf = np.concatenate([self._ref_buf, ref[:n]])
        return self._drain(flush=False)

    def flush(self) -> np.ndarray:
        """Traite la traîne (< hop) en complétant par des zéros. À appeler une
        fois à la fin pour récupérer les derniers échantillons nettoyés."""
        return self._drain(flush=True)

    # --------------------------------------------------------------- interne
    def _drain(self, flush: bool) -> np.ndarray:
        H = self.H
        outs = []
        while self._mic_buf.size >= H or (flush and self._mic_buf.size > 0):
            take = min(H, self._mic_buf.size)
            d = self._mic_buf[:take]
            x = self._ref_buf[:take]
            if take < H:                                  # dernier bloc : padding zéro
                d = np.concatenate([d, np.zeros(H - take)])
                x = np.concatenate([x, np.zeros(H - take)])
            e = self._step_hop(d, x)
            outs.append(e[:take])                         # ne restitue que les vrais échantillons
            self._mic_buf = self._mic_buf[take:]
            self._ref_buf = self._ref_buf[take:]
            if take < H:
                break
        return (np.concatenate(outs) if outs else np.zeros(0)).astype(np.float32)

    def _step_hop(self, d: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Un hop de H échantillons (overlap-save). d, x : longueur H."""
        N, H, L = self.N, self.H, self.L
        # Buffer glissant de référence : on pousse H nouveaux échantillons
        self._rb = np.concatenate([self._rb[H:], x])      # longueur N
        X = np.fft.rfft(self._rb)
        # Écho estimé : conv linéaire valide = H derniers échantillons
        y = np.fft.irfft(self.W * X, N)[N - H:]
        e = d - y                                         # micro nettoyé

        ey = float(np.dot(y, y))
        ee = float(np.dot(e, e))
        ed = float(np.dot(d, d)) + self.eps
        # DTD robuste au démarrage : on ne gèle QUE si le filtre a BIEN convergé
        # (l'écho estimé explique ≥40 % du micro) ET que l'erreur explose (voix
        # proche). Le seuil 0.4 est crucial : trop bas (0.05) il gèle pendant la
        # convergence et casse l'annulation ; à 0.4 il reste inactif en usage
        # normal (le NLMS est déjà robuste à la voix proche, décorrélée) et ne
        # protège que les double-talks francs sur un filtre établi.
        converged = ey > 0.4 * ed
        if self.dtd and converged and ee > self.dtd_ratio * ey:
            self._freeze = self._dtd_hang
        adapt = self._freeze <= 0
        if self._freeze > 0:
            self._freeze -= H

        # Énergie de réf du bloc (lissée) — normalisation NLMS SCALAIRE :
        # denom ≈ Σ‖x_vec‖² ≈ L·E_bloc. Le facteur L est essentiel (sans lui le
        # pas est ~L× trop grand → divergence ; une normalisation par bin se
        # met mal à l'échelle ici). Cold start : on amorce sans lissage.
        ex = float(np.dot(x, x))
        self._pw = ex if not self._started else self.lam * self._pw + (1.0 - self.lam) * ex
        self._started = True

        # Pas de réf utile → pas d'écho à estimer, pas d'adaptation (évite
        # d'amplifier le bruit du micro pendant les silences système).
        if adapt and self._pw > self.eps:
            epad = np.zeros(N)
            epad[N - H:] = e
            E = np.fft.rfft(epad)
            phi = np.fft.irfft(np.conj(X) * E, N)
            phi[L:] = 0.0                                 # contrainte : filtre de L taps
            self.W += (self.mu / (L * self._pw + self.eps)) * np.fft.rfft(phi)
        return e


# --------------------------------------------------------------------- helpers
def estimate_delay(mic: np.ndarray, ref: np.ndarray, sr: int,
                   max_ms: float = 500.0) -> int:
    """Retard (en échantillons) de l'écho dans `mic` par rapport à `ref`, estimé
    par GCC-PHAT. Positif => l'écho arrive APRÈS la référence (cas normal).
    Robuste au niveau et à la coloration (blanchiment par la phase)."""
    mic = np.asarray(mic, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    n = min(mic.size, ref.size)
    if n < sr // 10:                                    # trop court pour estimer
        return 0
    mic, ref = mic[:n], ref[:n]
    nfft = _next_pow2(2 * n)
    M = np.fft.rfft(mic, nfft)
    R = np.fft.rfft(ref, nfft)
    cross = M * np.conj(R)
    cc = np.fft.irfft(cross / (np.abs(cross) + 1e-9), nfft)   # PHAT : phase pure
    max_lag = min(int(sr * max_ms / 1000.0), nfft // 2 - 1)
    # lags positifs (mic en retard sur ref) dans cc[0:max_lag], négatifs en queue
    pos = cc[:max_lag + 1]
    neg = cc[-max_lag:]
    if pos.max() >= (neg.max() if neg.size else -np.inf):
        return int(np.argmax(pos))                      # 0..max_lag
    return int(np.argmax(neg) - max_lag)                # -max_lag..-1


def echo_coherence(mic: np.ndarray, ref: np.ndarray, sr: int,
                   win_ms: float = 64.0, align: bool = True,
                   max_ms: float = 500.0) -> float:
    """Fraction de la puissance de `mic` linéairement explicable par `ref`, estimée
    par **cohérence spectrale** (magnitude-squared coherence, moyennée façon Welch
    puis pondérée par la PSD du micro). Valeur dans [0, 1].

    Sert à DÉCIDER, sur un COURT extrait, si un énoncé micro est en réalité l'écho
    de la sortie (haut-parleurs recaptés) : ~1 ⇒ le micro est une version filtrée/
    retardée de `ref` (écho) ; ~0 ⇒ signal indépendant (vraie voix proche) ; valeur
    intermédiaire en double-talk (≈ part d'énergie d'écho dans le micro).

    Pourquoi pas `cancel_array` + ERLE ? Le filtre adaptatif n'a PAS le temps de
    converger sur un extrait de 1–3 s d'audio coloré (parole) → ERLE proche de 0
    même pour de l'écho franc. La cohérence, elle, est **convergée par construction**
    (estimateur statistique, pas de filtre à établir) → exploitable énoncé par énoncé.

    La cohérence est invariante à la coloration et à la réverbération (filtrage
    linéaire de `ref`), MAIS un délai comparable à la fenêtre d'analyse la détruit ;
    on aligne donc d'abord le retard (GCC-PHAT, `estimate_delay`). ⚠ L'alignement
    RETARDE la réf pour un délai positif (sens inverse de `cancel_array`, dont le
    filtre causal, lui, veut la réf avancée). `mic`/`ref` : mono, même fréquence."""
    mic = np.asarray(mic, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    n = min(mic.size, ref.size)
    if n < sr // 8:                                     # < 125 ms : trop court pour estimer
        return 0.0
    mic, ref = mic[:n].copy(), ref[:n].copy()
    if align:
        d = estimate_delay(mic, ref, sr, max_ms=max_ms)
        if d > 0:                                       # écho en retard : RETARDE la réf pour l'aligner
            ref = np.concatenate([np.zeros(d), ref[:n - d]])
        elif d < 0:                                     # écho en avance (rare) : avance la réf
            ref = np.concatenate([ref[-d:], np.zeros(-d)])
    nper = min(max(256, int(sr * win_ms / 1000.0)), n)
    step = max(1, nper // 2)
    win = np.hanning(nper)
    nb = nper // 2 + 1
    Sxx = np.zeros(nb)
    Syy = np.zeros(nb)
    Sxy = np.zeros(nb, dtype=np.complex128)
    K = 0
    for s in range(0, n - nper + 1, step):              # Welch : fenêtres à 50 % de recouvrement
        Xm = np.fft.rfft(mic[s:s + nper] * win)
        Xr = np.fft.rfft(ref[s:s + nper] * win)
        Syy += np.abs(Xm) ** 2                          # PSD micro
        Sxx += np.abs(Xr) ** 2                          # PSD réf
        Sxy += Xm * np.conj(Xr)                         # interspectre
        K += 1
    if K < 2:                                           # pas assez de fenêtres → estimateur biaisé
        return 0.0
    msc = (np.abs(Sxy) ** 2) / (Sxx * Syy + 1e-12)      # cohérence par bin ∈ [0, 1]
    # Fraction de puissance MIC expliquée = cohérence moyenne pondérée par la PSD micro.
    return float(np.clip(np.sum(msc * Syy) / (np.sum(Syy) + 1e-12), 0.0, 1.0))


def cancel_array(mic: np.ndarray, ref: np.ndarray, sr: int,
                 filter_ms: float = 150.0, mu: float = 1.0,
                 align: bool = True, dtd: bool = True) -> np.ndarray:
    """Annulation d'écho « différé » sur des signaux entiers déjà alignés.

    Si `align`, estime d'abord le délai (GCC-PHAT) et recadre la référence pour
    que l'écho tombe dans la portée du filtre (convergence plus sûre). Retourne
    le micro nettoyé (même longueur, float32)."""
    mic = np.asarray(mic, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    n = min(mic.size, ref.size)
    if n == 0 or float(np.dot(ref[:n], ref[:n])) <= 0.0:
        return mic[:n].astype(np.float32)               # pas de réf utile → no-op
    mic, ref = mic[:n].copy(), ref[:n].copy()

    if align:
        d = estimate_delay(mic, ref, sr)
        if d > 0:                                        # écho en retard : avance la réf
            ref = np.concatenate([ref[d:], np.zeros(d)])
        elif d < 0:                                       # écho en avance (rare) : retarde la réf
            ref = np.concatenate([np.zeros(-d), ref[:d]])

    ec = EchoCanceller(sr, filter_ms=filter_ms, mu=mu, dtd=dtd)
    return np.concatenate([ec.process(mic, ref), ec.flush()])
