#!/usr/bin/env python3
"""
diarize_online.py — diarisation LIVE « light » : identifie les intervenants d'un
canal à la volée (Personne 1 / 2 / 3…) par clustering EN LIGNE d'embeddings de
voix, sans le pipeline streaming complet (diart).

Pourquoi c'est léger ici : le multicanal a déjà séparé « Moi » du reste, et notre
VAD a déjà découpé les énoncés. Il ne reste qu'à coller une étiquette de locuteur
sur chaque énoncé du canal « Système ». On extrait donc UN embedding par énoncé
(modèle `wespeaker` de pyannote, **sur CPU** → aucune pression sur la VRAM
partagée avec Whisper) puis on l'agrège par **clustering en ligne** : on compare
au centroïde de chaque locuteur connu (cosinus) ; au-dessus d'un seuil → même
personne (et on met à jour le centroïde), sinon → nouvelle personne.

Limites assumées (c'est du LIVE, erreurs tolérées) :
- pas de vue globale → un locuteur peut être scindé/fusionné, surtout au début ;
- un énoncé trop court (< `min_dur`) donne un embedding peu fiable → on renvoie
  -1 (« inconnu », l'appelant garde l'étiquette générique sans numéro).
Le différé fera la passe de référence (pyannote complet) si besoin.

Modèle mutualisé via cache module-level (déchargé par `unload_models`).
"""

from __future__ import annotations

import os
import numpy as np

_MODEL_REPO = 'pyannote/wespeaker-voxceleb-resnet34-LM'   # embarqué par speaker-diarization-3.1
_INFER_CACHE = {}                                          # device -> Inference (partagé)


def _hf_token() -> str | None:
    tok = os.environ.get('AUDIO_UTILS_HF_TOKEN') or os.environ.get('HF_TOKEN')
    if tok:
        return tok.strip()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        with open(os.path.join(root, '.env'), encoding='utf-8') as f:
            for line in f:
                if line.startswith('AUDIO_UTILS_HF_TOKEN='):
                    return line.split('=', 1)[1].split('#')[0].strip()
    except OSError:
        pass
    return None


def _get_inference(device: str = 'cpu'):
    """Charge (ou réutilise) le modèle d'embedding sur `device`. Import paresseux
    (pyannote/torch sont lourds)."""
    inf = _INFER_CACHE.get(device)
    if inf is not None:
        return inf
    import torch
    from pyannote.audio import Model, Inference
    os.environ.setdefault('HF_HOME', os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'vendor', 'models', 'huggingface'))
    model = Model.from_pretrained(_MODEL_REPO, use_auth_token=_hf_token())
    model.to(torch.device(device))
    inf = Inference(model, window='whole')
    _INFER_CACHE[device] = inf
    return inf


def unload_models():
    """Décharge le modèle d'embedding (libère la RAM/CPU). Renvoie True si qqch
    était chargé."""
    had = bool(_INFER_CACHE)
    _INFER_CACHE.clear()
    import gc
    gc.collect()
    return had


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


class OnlineDiarizer:
    """Clustering en ligne d'embeddings de voix → identifiants de locuteurs.

    `assign(audio16k)` renvoie un index de locuteur (0,1,2,…) stable dans la
    session, ou -1 si l'énoncé est trop court pour être fiable.
    """

    def __init__(self, threshold: float = 0.5, max_speakers: int = 8,
                 min_dur: float = 0.6, device: str = 'cpu',
                 update_min: float | None = None, margin: float = 0.1):
        self.threshold = float(threshold)     # cosinus ≥ seuil → ASSIGNE à ce locuteur
        # Seuil SÉPARÉ (plus haut) pour APPRENDRE : on ne met à jour un centroïde
        # que sur une attribution TRÈS confiante ET nette (marge sur le 2ᵉ). Sinon
        # les énoncés ambigus / chevauchés (mélange de voix) feraient DÉRIVER le
        # centroïde vers le centre → effondrement (tout matche un seul cluster).
        self.update_min = float(update_min if update_min is not None
                                else max(self.threshold + 0.15, 0.6))
        self.margin = float(margin)
        self.max_speakers = int(max_speakers)
        self.min_samples = int(16000 * float(min_dur))
        self.device = device
        self._centroids: list[np.ndarray] = []   # centroïde unitaire par locuteur
        self._counts: list[int] = []             # nb d'énoncés agrégés (pondération)
        self._inf = None

    def _embed(self, audio16k: np.ndarray) -> np.ndarray:
        if self._inf is None:
            self._inf = _get_inference(self.device)
        import torch
        wav = np.asarray(audio16k, dtype=np.float32).ravel()
        out = self._inf({'waveform': torch.from_numpy(wav).unsqueeze(0),
                         'sample_rate': 16000})
        return _unit(np.asarray(out, dtype=np.float64).ravel())

    def assign(self, audio16k: np.ndarray) -> int:
        """Index du locuteur pour cet énoncé (−1 si trop court / non fiable)."""
        audio16k = np.asarray(audio16k, dtype=np.float32).ravel()
        if audio16k.size < self.min_samples:
            return -1
        try:
            emb = self._embed(audio16k)
        except Exception:
            return -1
        # similarités à tous les centroïdes (cosinus = produit scalaire d'unitaires)
        if not self._centroids:
            self._centroids.append(emb)
            self._counts.append(1)
            return 0
        sims = [float(np.dot(emb, c)) for c in self._centroids]
        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        best = order[0]
        best_sim = sims[best]
        second_sim = sims[order[1]] if len(order) > 1 else -1.0

        if best_sim >= self.threshold:
            # Apprentissage CONSERVATEUR : on ne fait évoluer le centroïde que si
            # l'attribution est franche (très au-dessus du seuil ET nette vs le 2ᵉ).
            # Un énoncé chevauché/ambigu reçoit une étiquette mais NE pollue PAS.
            if best_sim >= self.update_min and (best_sim - second_sim) >= self.margin:
                self._merge(best, emb)
            return best
        if len(self._centroids) < self.max_speakers:
            self._centroids.append(emb)
            self._counts.append(1)
            return len(self._centroids) - 1
        # plafond atteint → rattache au plus proche, SANS mise à jour
        return best

    def _merge(self, i: int, emb: np.ndarray):
        n = self._counts[i]
        self._centroids[i] = _unit((self._centroids[i] * n + emb) / (n + 1))
        self._counts[i] = n + 1

    def warmup(self):
        """Charge le modèle d'embedding hors temps réel (sans créer de locuteur)."""
        try:
            self._embed(np.zeros(16000, dtype=np.float32))
        except Exception:
            pass

    @property
    def n_speakers(self) -> int:
        return len(self._centroids)
