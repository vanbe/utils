# TODO — utils

## Audio / transcription

- [ ] **DeepFilterNet en option « max qualité »** pour audio dégradé (conf captée
  de loin / bruyante). Débruitage + déréverb neuronal, en complément du
  pré-passage ffmpeg actuel (`SPEECH_ENHANCE_FILTERS` dans
  `actions/audio_utils/audio_utils_common.py`). À câbler **uniquement sur le
  différé** (`transcribe_audio.py` / `transcribe_channels.py`), pas en live
  (budget temps réel, 6 Go VRAM). pip `deepfilternet`, CPU ou GPU, +2–5 min/conf.
  Prévoir un fallback propre si le paquet est absent.
- [ ] **Régler `afftdn`/`dynaudnorm` sur audio réel** : valider/ajuster
  `nr`/`nf`/`g` de `SPEECH_ENHANCE_FILTERS` sur de vraies confs (un denoise trop
  agressif crée des artefacts qui dégradent la reconnaissance). Un seul endroit
  à modifier (constante partagée).
- [ ] (Optionnel) Normalisation pic/RMS par énoncé en live : volontairement non
  implémentée — n'améliore pas le SNR et Whisper normalise déjà. À ne faire que
  si un besoin concret apparaît.
