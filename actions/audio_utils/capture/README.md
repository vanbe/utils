# capture.exe — capteur audio WASAPI autonome

Binaire Windows minimal utilisé par l'action **Record audio** de la TUI utils
(`recorder.py`). Il capture, côté Windows, plusieurs endpoints audio simultanés
— micro(s) **et** sortie(s) système via **loopback WASAPI** — et écrit du PCM
sur stdout. Tout l'encodage FLAC et la transcription restent côté WSL.

## Pourquoi un binaire natif (et pas un câble virtuel / Python / DShow)

- WSL ne peut pas capter la sortie système Windows nativement.
- Le **loopback WASAPI est une API Windows intégrée** : aucun driver, aucun
  filtre DirectShow (`regsvr32`), aucun VB-CABLE/VoiceMeeter, aucun VC++ redist,
  aucun privilège admin. Un simple `.exe` statique suffit.

## Le binaire n'est PAS versionné

`bin/capture.exe` est gitignoré. On committe **la source + les scripts de
build**, pas l'artefact. La TUI propose de le construire automatiquement s'il
manque (si un compilateur est disponible).

## Build

Depuis Linux/WSL (cross-compile, recommandé) :

```bash
sudo apt-get install g++-mingw-w64-x86-64    # une fois
./build.sh                                   # → ../bin/capture.exe
```

Depuis Windows (MinGW g++ ou MSVC) :

```bat
build.bat
```

## Contrat CLI (consommé par recorder.py)

```
capture.exe --list
    → stdout : JSON [{"id","name","kind":"input|output","channels":N}, ...]
      render endpoints → "output" (loopback) ; capture endpoints → "input"

capture.exe --rate 48000 --source <id> [--source <id> ...]
    → stdout : PCM s16le interleavé (canaux de la source 0, puis 1, ...),
               cadencé à `rate` Hz ; total canaux = somme des canaux par source
    → stderr : "LEVEL <idx> <rms 0..1>" ~10×/s par source (vumètres)
    → s'arrête à la fermeture de stdout / Ctrl-C / terminate
```

## Test (à lancer sur le laptop WSL)

`test_recorder.py` exerce la chaîne complète (recorder → capture.exe|ffmpeg →
FLAC) sans la TUI, puis vérifie le résultat (ffprobe + channels.json).

```bash
cd actions/audio_utils/capture
PY=../../../.venv/bin/python3          # python du venv utils

# 1. lister les sources (et récupérer les ids)
$PY test_recorder.py --probe

# 2. enregistrer 8 s (auto : 1er micro + 1ère sortie), parler + jouer un son
$PY test_recorder.py --seconds 8 --out /mnt/c/Users/<toi>/Desktop

# 3. sources explicites
$PY test_recorder.py --source "<id1>" --source "<id2>"

# (re)construire le binaire seul
$PY test_recorder.py --build

# 4. transcription LIVE par canal (Moi / Système) — écrit .srt + .md
$PY test_recorder.py --transcribe --seconds 12 --out /mnt/c/Users/<toi>/Desktop
$PY test_recorder.py --transcribe --lang auto --model medium --seconds 12
```

Le script affiche les vumètres en direct, puis le `ffprobe` du FLAC (codec /
canaux / rate), le contenu de `channels.json`, et les commandes ffmpeg de démux
par canal. Avec `--transcribe`, il charge le modèle Whisper, affiche les phrases
au fil de l'eau et écrit `<nom>.srt` (avec timings) **et** `<nom>.md` (sans
timings) — exactement comme l'action TUI *Record audio → transcription Oui*.

## Limites connues

- Conversion de fréquence/format déléguée au mixeur partagé WASAPI
  (`AUTOCONVERTPCM`) ; pas de resampling haute qualité.
- Synchronisation inter-sources approximative (pacing 10 ms, zero-fill sur
  sous-alimentation) — suffisant pour une transcription par canal.
