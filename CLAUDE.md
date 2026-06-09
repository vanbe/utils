# Utils Tools — Claude Project Notes

## Project overview

Terminal TUI (`utils_tools.py`) and a collection of actions scripts under `actions/`.
Primary working environment: WSL2 on Windows, with paths under `/mnt/c/` pointing to the Windows filesystem (Nextcloud sync folder).

**Non-interactive CLI for agents**: `utils_run.py` exposes all document conversion actions as a JSON API for use by automated agents (including Claude in other projects).

```bash
# Discover what actions are available for a file
python3 utils_run.py <file>

# Run an action — progress goes to stderr, JSON result to stdout
python3 utils_run.py <file> <action>

# List every registered action with supported extensions
python3 utils_run.py --list-all
```

Key actions for document extraction:
- `pdf-extract` — PDF → Markdown (PyMuPDF, fast, no GPU needed)
- `pdf-mineru` — PDF → Markdown (MinerU, best for layout/tables/formulas)
- `doc-to-md` — DOCX/DOC → Markdown (pandoc)
- `odt-to-docx` — ODT → DOCX (LibreOffice)

Exit code 0 on success, 1 on error. JSON shape:
- Success: `{"status": "ok", "output_file": "/abs/path/to/result.md"}`
- Error: `{"status": "error", "message": "..."}`

---

## Keeping `utils_run.py` and the skill in sync

The skill file at `.claude/skills/utils-convert/SKILL.md` documents every action exposed by `utils_run.py`. **Whenever you make any of the following changes, update both files together:**

| Change | What to update |
|--------|---------------|
| New action added to `_REGISTRY` in `utils_run.py` | Add a row to the action table in the skill |
| New **folder** action added to `_FOLDER_REGISTRY` in `utils_run.py` | Add a row to the "Folder operations" table in the skill **and** wire it into `_actions_for` + `_ACTION_CAT` in `utils_tools.py` (which reads the script path from `_FOLDER_REGISTRY` — single source of truth, no path drift) |
| Action removed or renamed in `utils_run.py` | Remove/rename the corresponding row in the skill |
| Supported extensions change for an action | Update the skill table and the "Choosing the right action" notes |
| Output filename convention changes (e.g. `_extracted.md` suffix) | Update the skill table |
| New action wired up in `_actions_for` in `utils_tools.py` but not yet in `utils_run.py` | Add it to `utils_run.py` first, then update the skill |
| New document type added to `utils_tools.py` (new `*_EXTS` constant + actions) | Mirror it in `utils_run.py` and the skill |

The single source of truth for the registries is `utils_run.py` — `_REGISTRY` (per-file actions) and `_FOLDER_REGISTRY` (folder/batch actions). The skill and `utils_tools.py` (TUI) are mirrors of it — they must stay in sync. Folder-action **script paths live only in `_FOLDER_REGISTRY`**; the TUI imports it, so a path can't drift between TUI and CLI.

---

## UX conventions for `utils_tools.py`

- **Default option always first** in `select_menu` lists — the cursor starts there, Enter confirms immediately without navigating.
- After an action completes, return directly to the file browser (no re-showing the action list).
- No mode-selection prompt on startup — go straight into the file browser.
- Folder actions are accessible via the `[./]` entry inside the browser; no separate mode needed.
- TTS speaking speed: 0.95 is default and must appear first in the speed menu.
- **Actions destructives en raw-mode** : l'écran live de *Record audio* exige une
  **confirmation O/oui** avant d'annuler-supprimer (Esc/Q n'efface pas directement
  le FLAC). En pause, les vumètres retombent à 0 et les aperçus interim « … » sont
  purgés (la capture est gelée par SIGSTOP, aucune donnée fraîche). Le nombre de
  lignes rendues doit rester constant entre les états (REC/PAUSE/confirm) pour que
  le redraw par `\033[NF` ne dérive pas — **à taille de terminal donnée**.
- **Transcript live : zone à hauteur DYNAMIQUE** (`_record_lines`) — elle occupe
  toute la fenêtre (`_term_height() − en-tête − pied − 2 − 1 marge`, plancher
  `_TRANSCRIPT_H_MIN`) au lieu d'un nombre fixe. Les phrases longues **wrappent**
  (`_fmt_seg` → liste de lignes via `textwrap`, continuation indentée sous le
  texte) au lieu d'être tronquées par « … ». La contrainte « lignes constantes »
  tient toujours **pour une taille de terminal donnée** ; le **redimensionnement**
  est géré dans `_record_live` (détection `(_term_height(), _term_width())` → si
  ça change : `\033[2J\033[H` efface tout + `first=True`, donc pas de lignes
  fantômes). La marge d'1 ligne évite que le dernier `\n` ne fasse scroller (ce
  qui casserait le `\033[NF`).
- **Anti va-et-vient (`_transcript_lines`)** : l'aperçu **interim** est réinterprété
  au fil de la parole → son nombre de lignes wrappées change. Si on le mélangeait
  aux finaux dans une seule liste « N dernières lignes », les lignes finales
  **bondiraient** de haut en bas à chaque révision. Donc **deux régions à tailles
  stables** : les segments **finaux** au-dessus (remplissent vers le bas puis
  défilent), une région interim de **taille fixe** (`_INTERIM_RESERVE`, 3) en bas.
  L'interim ayant sa propre zone, sa réinterprétation ne décale jamais les finaux
  déjà figés (seul un *commit* d'énoncé fait défiler les finaux d'un cran — normal).

---

## Model parameter benchmarking — MANDATORY for every model

**Every Ollama model used by this project must be empirically benched on the
target machine before being committed to `.env`.** Pure heuristics ("model
size + VRAM → num_gpu") do **not** predict the shared-memory paging cliff:
the cliff is a driver-level behaviour that triggers below the threshold raw
weights-and-cache math suggests. Two configs that look equivalent on paper
can differ by 8× in real decode rate.

**Worked example, this hardware** (6 GB dedicated VRAM, gemma4:e4b, 11 GB
total): `num_gpu=42` (33 % on GPU, rest on CPU) → 15 t/s. `num_gpu=46` (the
"100 % GPU" config Ollama recommends and what raw math would pick) → 2 t/s,
TTFT 18 s. The driver was paging the missing ~5 GB through shared GPU memory.
Partial offload that keeps CPU layers in fast system RAM beats paged-GPU.

**Tool**: [actions/ai_utils/bench_ollama_params.py](actions/ai_utils/bench_ollama_params.py)
**Detailed docs**: [actions/ai_utils/bench_ollama_params.md](actions/ai_utils/bench_ollama_params.md) — full CLI reference, modes, examples, troubleshooting.

**Auto-tune** (recommended for new / unknown models):
```
python3 actions/ai_utils/bench_ollama_params.py <doc> --model X --auto-tune
```
Probes the model via `/api/show` (size, layer count, max ctx), reads VRAM via
`nvidia-smi`, runs a coarse `num_gpu` sweep at `auto_ctx=8192`, then a fine
±2-layer sweep around the best non-paging result. Runtime scales with model
size: ~5 min for 4B models, 20–40 min for 26B+.

**Manual sweep** (when you already have a guess to validate):
```
python3 actions/ai_utils/bench_ollama_params.py <doc> --model X \
    --num-gpu A,B,C --num-ctx M,N
```

**Two-step workflow** for picking final `.env` values:
1. `--auto-tune` to find the best `num_gpu` at small ctx (8192).
2. With that `num_gpu` locked in, re-run with explicit `--num-ctx` covering
   the contexts you actually need (e.g. `--num-ctx 40000,80000,128000`) to
   confirm VRAM headroom and decode rate hold at larger contexts.

**Selection rules**:
- Maximise **decode tok/s** (steady-state generation speed).
- **Reject any config with TTFT > 5 s at "100 % GPU"** — that's the paging
  signature; pick a lower `num_gpu` even if its decode looks slightly worse.
- Leave **~1 GB nvidia-smi VRAM headroom** for other GPU consumers (browser,
  Windows DWM, model swap between A and B).
- For a model larger than dedicated VRAM, the all-on-GPU config
  (`num_gpu = num_layers`) is **always wrong**, even when `ollama ps` reports
  "100 % GPU".

**Reference results on this machine** (Lenovo, 6 GB dedicated VRAM):
- **Model B — `gemma4:e4b`** (11 GB total): `OLLAMA_NUM_GPU_B=42`,
  `OLLAMA_MODEL_B_CTX=128000` → ~15 t/s decode, 5.7 GB VRAM total, ctx fully
  pre-allocated. Use `100000` instead of `128000` if you want ~950 MB of
  headroom for other GPU consumers.
- **Model A — `gemma4:26b`** (~17 GB total): not yet empirically benched.
  Current `.env` guess `OLLAMA_NUM_GPU_A=7` / `OLLAMA_MODEL_A_CTX=4096` —
  must be validated with `--auto-tune` before being trusted.

When you change a model tag in `.env` (new quantisation, new minor version),
re-bench. Quantisation changes shift the per-layer footprint and can move
the paging cliff by several layers.

---

## AI actions (`actions/ai_utils/`)

- **Dedicated OCR model**: `pdf_vision_ocr.py` uses `OLLAMA_OCR_MODEL` / `OLLAMA_OCR_MODEL_CTX` / `OLLAMA_OCR_NUM_GPU` — independent of the A/B transform pipeline. The TUI must not ask for model/quality selection for Vision OCR.
- **Tuned-OCR mode**: `OCR_TUNED_PROMPT` env var. When set, `pdf_vision_ocr.py` skips the system prompt and the lang/text-hint augmentations, sending only the literal env value as the user message (e.g. `<image>\nFree OCR.` for DeepSeek-OCR). `\n` is converted to a real newline. Use this for specialised OCR models that ship with a fixed instruction phrase.
- **Inference extra_body**: every Ollama inference call must pass `extra_body=backend.get_inference_extra_body()` to prevent Ollama from silently reloading the model with default context (262144).
- **Context priority** in OCR: explicit env var → `_OCR_DEFAULT_CTX` floor (8192) → `--context-size` CLI override.
- **Model load timeout**: `_LOAD_TIMEOUT = 600` seconds (26B models take several minutes).
- **Unload events** must be logged as a dedicated `step/ok` line, not silently swallowed.

---

## Audio recording (action "Record audio")

Capture multi-sources (micros + sorties système) → **un seul FLAC multicanal** +
sidecar `<name>.channels.json` (map canal→source). Transcription par canal en
direct (`live_transcribe.py`) ou différée (`transcribe_channels.py`). **Action de
dossier, TUI-only.**

- **Moteur** : `actions/audio_utils/recorder.py` — `detect_backend()`,
  `list_sources()`, `class Recorder`. Backend-aware :
  - `linux` : capture locale PulseAudio/PipeWire (`parec` pour les vumètres,
    `ffmpeg -f pulse … amerge` pour le FLAC). Les SORTIES = sources `.monitor`.
  - `wsl` : WSL ne peut pas capter la sortie système Windows nativement. La
    capture est déléguée à Windows via le binaire **`bin/capture.exe`** (loopback
    WASAPI = **API Windows intégrée, rien à installer** : pas de driver, pas de
    filtre DShow/`regsvr32`, pas de VB-CABLE/VoiceMeeter, pas de VC++ redist).
    Lancé en interop ; PCM s16le sur stdout → `ffmpeg` (WSL) encode le FLAC ;
    lignes `LEVEL i rms` sur stderr → vumètres.
  - ⚠ **`capture.exe` DOIT être lancé avec `stdin=DEVNULL`** : sinon le binaire
    Windows hérite du stdin du terminal et **l'interop WSL lance un relais qui lit
    le pty** pour le lui transmettre → il **VOLE les frappes au TUI raw-mode**
    (symptôme : Espace/pause « ne marche qu'1 fois sur 4-5 »). Tout `Popen`/`run`
    d'un binaire Windows depuis le TUI doit fixer `stdin=DEVNULL`.
- **`capture.exe` n'est PAS versionné** (gitignoré dans `bin/`). On committe la
  source + les scripts : `actions/audio_utils/capture/` (`capture.cpp`,
  `build.sh` cross-compile mingw, `build.bat` Windows, `README.md`). La TUI
  propose de le **construire** s'il manque (`recorder.build_capture()`). Cross-build
  vérifié : `g++-mingw-w64-x86-64`, binaire statique, imports = `kernel32/msvcrt/
  ole32` uniquement.
- **Confidentialité / intégrité du flux audio** :
  - capture.exe ne communique QUE par **pipes anonymes hérités** (stdout PCM,
    stderr LEVEL) → privés au couple parent↔enfant, **aucun tiers ne peut
    s'y brancher**. Il ne lit jamais stdin ; entrée = `argv` seulement.
  - Le binaire buildé **ne linke aucune lib réseau** (`build.sh` : `ole32/
    oleaut32/uuid/winmm`) → il ne PEUT pas exfiltrer l'audio. **INVARIANT :
    rester stdio-only, ne JAMAIS ajouter de socket/named pipe** (sinon il
    faudrait authentifier l'endpoint). La transcription tourne 100 % en local
    (faster-whisper/CTranslate2) — l'audio ne part jamais sur le réseau.
  - Une **clé compile-time serait inutile** : pas d'endpoint à garder, et un
    secret embarqué dans un binaire distribué n'est pas secret. La vraie menace
    réaliste = un **capture.exe substitué** (trojan ouvrant une socket). Parade =
    **épinglage SHA-256** : `build_capture()`/`pin_capture()` écrivent
    `bin/capture.exe.sha256` (gitignoré, propre à la machine) ; `verify_capture()`
    renvoie `ok|mismatch|unpinned|absent` ; `recorder._start_wsl()` **REFUSE**
    un `mismatch`. La TUI (`_ensure_capture_exe`) propose rebuild/ré-épinglage.
  - ⚠ **Exposition résiduelle réelle** : les sorties (FLAC/srt/md) sont écrites
    dans le dossier de travail, souvent **synchronisé Nextcloud** → elles partent
    vers le cloud + tous les appareils synchronisés. C'est là, pas dans le pipe,
    que se joue la confidentialité côté usage.
- **TUI-only assumé** : l'action est câblée dans `_actions_for` (branche dossier)
  + `_ACTION_CAT` de `utils_tools.py`, mais **PAS** dans `_REGISTRY`/
  `_FOLDER_REGISTRY` de `utils_run.py` ni dans `SKILL.md` (enregistrement live =
  interactif). Ne pas la croire « manquante » lors d'une passe de sync registries.
- **Démux + transcription différée par canal** : `transcribe_channels.py` lit le
  sidecar `channels.json`, démux chaque source (`pan=mono|c0=…` = **moyenne** des
  canaux de la source, pas seulement `c0`), transcrit par canal et fusionne →
  `<base>.srt` + `<base>.md`. **Attribution Moi/Système par construction** (pas de
  pyannote). Câblé dans la TUI : `act_transcribe` détecte le sidecar et propose
  « Par canal » (recommandé) vs « Standard — diarisation » (`transcribe_audio.py`,
  qui downmix tout en mono + diarise). ⚠ Ne PAS passer un FLAC multicanal au mode
  standard sans raison : son préprocess `pan=mono|c0=c0` ne garde que le 1er canal.
- **Labels par canal mutualisés** : `whisper_common.channel_labels(sources)` est
  l'unique source des libellés « Moi / Système » — utilisé par `live_transcribe.py`
  ET `transcribe_channels.py` (donc live ≡ différé par canal).

### Transcription live par canal

Option « Transcription live ? Oui » dans *Record audio* → transcrit chaque canal
en direct, affichage défilant (`Moi` / `Système`) + `.srt` aligné sur le FLAC.

- **Moteur** : `actions/audio_utils/live_transcribe.py` — `class LiveTranscriber`.
  Reçoit le PCM via `Recorder(on_pcm=…)` (tee : `capture.exe`/pulse → Python →
  ffmpeg FLAC **et** transcription), désentrelace par canal (`channels.json`),
  downmix mono + resample 16 kHz, **VAD-gate** par énergie, file → **un seul
  worker WhisperModel partagé**.
- **Attribution locuteur = par canal**, pas de pyannote : `input`→« Moi »,
  `output`→« Système ». C'est l'intérêt majeur du multicanal.
- **1 seul modèle partagé** (canaux en série) : sur un GPU unique, dédoubler ne
  gagne rien et **2 modèles turbo ne tiennent pas en 6 Go** + DWM Windows. VAD-gate
  → micro muet = coût nul. faster-whisper n'est pas streaming → **chunk-on-silence**.
- **Sorties** : `.srt` (avec timings) **et** `.md` (sans timings) par défaut.
- **Réhaussement audio dégradé (capté de loin / bruyant)** : option `--enhance`
  (différé seulement) = pré-passage ffmpeg `SPEECH_ENHANCE_FILTERS`
  (`highpass=80 + afftdn + dynaudnorm`, dans `audio_utils_common.py`, sans
  dépendance) appliqué avant Whisper. ⚠ La **normalisation** (loudnorm/RMS) ne
  change PAS le SNR et Whisper normalise déjà → inutile pour de l'audio lointain ;
  ce qui aide = débruitage (`afftdn`) + remontée dynamique (`dynaudnorm`) + un
  **modèle qualité en 2ᵉ passe** (le live reste un moniteur `beam=1` temps réel,
  le différé fait le transcript fiable). Câblé : `transcribe_audio.py --enhance`
  (menu « Enhance — far-field / noisy »), `transcribe_channels.py --enhance`
  (menu « Réhaussement audio »). **Pas appliqué en live** (budget temps réel).
- **Code mutualisé** : `whisper_common.py` (table modèles, chargement+fallback OOM,
  **`write_srt`/`write_md`**) est utilisé par `live_transcribe.py` ET
  `transcribe_audio.py` → le MD live est **identique** au MD différé (même writer).
  Poids de modèle partagés (`download_root` = `vendor/models/whisper`).
- **Modèles recommandés = défaut « (recommandé) »**, **distincts live vs différé**
  via `whisper_common.recommended_model(role)` : `role='live'` → `LIVE_TRANSCRIBE_MODEL`
  (contrainte temps réel), `role='transcribe'` → `TRANSCRIBE_MODEL` (qualité, sans
  contrainte). Calcul matériel (`setup_env.py`) : GPU 5–10 Go → turbo pour les deux ;
  ≥10 Go → live turbo / différé large ; **CPU → live base (temps réel) / différé small**.
  Clés dans `_AUTO_KEYS`. (Bench espeak CPU 2 cœurs : la qualité dépend surtout de
  l'audio réel ; small > base > tiny ; small n'est pas temps réel sur CPU faible.)
- **Hôte réactif** : `LIVE_TRANSCRIBE_CPU_THREADS` (= cœurs − 1) limite les threads
  CPU du live (sans effet sur GPU).
- **Anti-hallucination** : sur un canal quasi-silencieux (micro après AEC, blancs
  entre énoncés), Whisper **invente** des phrases passe-partout (« Thank you »,
  « You », « We'll be right back »…). Double parade mutualisée (`whisper_common`) :
  (1) `is_hallucination()` — jette une phrase du blocklist si `no_speech_prob`
  élevé, ou tout segment à `avg_logprob < −1.2` ; (2) `is_echo_duplicate()` —
  jette une phrase COURTE (≤4 mots) **répétée sur le même canal** ou **quasi
  simultanée sur les deux** (résidu d'écho). Appliqué en live (`_emit`, état
  `_recent`) ET différé (passe finale sur `segs`). La 1ʳᵉ occurrence isolée d'une
  phrase courte est gardée (seuls répétitions/échos tombent) → ne supprime pas un
  vrai « Okay. » isolé.
- **Affichage TUI** : les segments finaux sont **triés par timestamp** au rendu
  (`_transcript_lines`) — les canaux étant transcrits en série, ils arrivent dans
  le désordre temporel. État **PAUSE** affiché « ⏸ PAUSE » en **orange** (`_ORG`).
- **Pause = plus AUCUNE inférence** : `LiveTranscriber.set_paused(True)` →
  `feed_bytes` jette l'audio entrant (vide tampon + gates) → rien ne part au GPU
  (économie batterie laptop), **le modèle reste chargé**. Indispensable car sur
  WSL le `SIGSTOP` du recorder ne fige pas toujours `capture.exe` (binaire Windows
  via interop) → le PCM continuait d'arriver et Whisper tournait sur le bruit. La
  TUI appelle `set_paused` dans le toggle Espace (`_record_live` reçoit le
  `transcriber`). `_src_pos` n'avance pas en pause (timeline alignée sur le FLAC).
- **Boucle clavier de `_record_live` (réactivité)** : 3 règles pour que l'appui
  Espace réponde du 1ᵉʳ coup (le « il faut faire 2× » venait de là) :
  (1) on **draine TOUTES les touches en attente AVANT de redessiner** (input lu
  en tête de boucle, plus une seule touche/cycle de 0,1 s) ; (2) on **ne
  redessine que si le contenu a changé** (`lines != prev_lines`) → en pause/idle
  le terminal est muet donc la frappe est instantanée ; (3) la frame est écrite
  en **UN seul `write`** (≈40 petites écritures via le pty WSL = lent → latence).
- **Sessions enchaînées sans recharger le modèle** : `live_transcribe._MODEL_CACHE`
  (clé `(nom, device, compute)`) garde le `WhisperModel` chargé entre sessions.
  La TUI (`act_record_audio`) boucle « Nouvelle session / Terminer » → on enchaîne
  les conversations sans recharger Whisper (warmup sauté si `model_loaded_from_cache`).
  **À la sortie** (Terminer / Esc / erreur), un `try/finally` appelle
  `live_transcribe.unload_models()` → **VRAM libérée** (sinon le modèle restait
  chargé pour rien, mauvais sur batterie). ⚠ Le `finally` doit casser **toutes**
  les références au modèle : `rec.on_pcm = None` (le recorder référence
  `transcriber.feed_bytes`) + `transcriber = None` + `rec = None`, sinon le GC ne
  peut pas libérer. Le cache n'accélère que l'enchaînement INTERNE à la boucle.
- **Arrêt = `S` uniquement** (plus `Enter`) : `Enter` est trop facile à presser
  par erreur et coupait l'enregistrement → ignoré pendant la capture. Aide mise à
  jour : « Space pause/resume · **S** stop & save · Esc/Q annuler ».
- **Décodage par canal — VAD à DEUX étages** : (1) gate d'ÉNERGIE en amont
  (découpe les énoncés, micro muet = coût nul) ; (2) **Silero VAD** dans le
  décodeur (`vad_filter=True`, params permissifs `threshold=0.5 /
  min_speech_duration_ms=200 / min_silence_duration_ms=300`). Le gate d'énergie
  laisse passer le **bruit NON-VOCAL fort** (frappes clavier, clics) que Whisper
  transcrirait en mots inventés (« and », « Okay », « Bye ») ; Silero rejette ces
  chunks sans parole réelle (vérifié : bruit blanc → 0 segment). On garde
  **`no_speech_threshold=1.0`** : sur une région que Silero a VALIDÉE comme
  parole, on ne re-jette pas le chunk → évite le « canal perdu » (énoncé court
  d'un canal qui disparaissait quand on s'appuyait sur le no_speech interne).
  ⚠ Historique : on avait d'abord `vad_filter=False` (énergie seule) pour ne pas
  perdre d'énoncé court, mais ça transcrivait le bruit clavier → Silero permissif
  est le bon compromis. Filtre TEXTE en 3ᵉ ligne (`whisper_common.is_hallucination` :
  tics de sous-titres, **mots-outils isolés** « and/the/uh/euh… », résidu d'écho
  via no_speech_prob/avg_logprob ; + `is_echo_duplicate`).
- **Diarisation LIVE des intervenants (canal Système → P1/P2/P3)** :
  `actions/audio_utils/diarize_online.py` — `class OnlineDiarizer`. Le multicanal
  ayant déjà isolé « Moi » et le VAD ayant déjà découpé les énoncés, il ne reste
  qu'à coller un locuteur sur chaque énoncé de SORTIE : **1 embedding par énoncé**
  (`pyannote/wespeaker-voxceleb-resnet34-LM`, **sur CPU** → 0 VRAM, ~85 ms/énoncé)
  + **clustering en ligne** par seuil cosinus (≥ seuil → même personne + MAJ
  centroïde ; sinon nouvelle personne). Câblé : `LiveTranscriber(diarize=True)` →
  le worker diarise les énoncés des canaux `output` (un diariseur PARTAGÉ →
  numéros cohérents) et émet `label = "Système · Pn"` ; le micro reste « Moi ».
  Énoncé < `min_dur` (0.6 s) → embedding non fiable → -1 → libellé générique sans
  numéro. ⚠ **Seuil à calibrer sur audio réel** (`LIVE_DIARIZE_THRESHOLD`, défaut
  0.5 ; `LIVE_DIARIZE_MAX_SPEAKERS`=8). Le modèle d'embedding (gated `pyannote/
  embedding` INaccessible → on prend `wespeaker`, embarqué par
  speaker-diarization-3.1, déjà autorisé par le token) est préchauffé dans
  `start()` et déchargé par `live_transcribe.unload_models()`.
  - **Apprentissage CONSERVATEUR des centroïdes (anti-effondrement)** : l'ASSIGNATION
    (cos ≥ `threshold`) est séparée de l'APPRENTISSAGE. On ne met à jour un centroïde
    que si l'attribution est *franche* : `cos ≥ update_min` (≈ threshold+0.15, défaut
    0.6) **ET** marge sur le 2ᵉ centroïde ≥ `margin` (0.1). Sinon le centroïde n'évolue
    pas. Sans ça, les énoncés **chevauchés** (embedding = mélange de voix) polluaient un
    centroïde en moyenne mobile → il dérivait vers le centre → *tout* matchait ce cluster
    → effondrement (« tout l'historique réécrit en P3 »). Avec : un mélange 50/50 a une
    marge ~0 → ignoré pour l'apprentissage ; les centroïdes restent purs (vérifié :
    rafale de chevauchements → 0 dérive, voix propres toujours bien classées). Overrides
    `.env` : `LIVE_DIARIZE_UPDATE_MIN`, `LIVE_DIARIZE_MARGIN`.
  - ⚠ Limite restante (inhérente au live, sans lookahead) : la 1ʳᵉ phrase d'un locuteur
    qui répond TRÈS vite peut être étiquetée comme le précédent (son cluster n'existe pas
    encore). Monter `LIVE_DIARIZE_THRESHOLD` réduit ce cas mais risque de sur-scinder.
  ⚠ Interaction
  connue : `is_echo_duplicate` peut jeter deux énoncés COURTS identiques (même
  texte) rapprochés même de locuteurs différents (cas-limite, parole réelle =
  textes différents). C'est du LIVE (erreurs tolérées) ; le différé pyannote
  complet reste la passe de référence.
- **Temps réel** : aperçus **interim** pendant qu'on parle
  (`LIVE_TRANSCRIBE_INTERIM_SEC`, défaut 2 s) — affichés grisés « … » puis
  remplacés par le final ; ≤1 interim en vol par source (pas de pile-up).
  `condition_on_previous_text=False` + warmup modèle au démarrage.
- **Config `.env`** : `LIVE_TRANSCRIBE_MODEL` + `TRANSCRIBE_MODEL` / `_DEVICE` /
  `_COMPUTE_TYPE` / `_CPU_THREADS` / `_LANGUAGE` (`fr`|`auto`) / `_WINDOW_SEC` /
  `_INTERIM_SEC`. Aucune dépendance nouvelle (faster-whisper marche en CPU via
  CTranslate2, **sans torch**).

### Annulation d'écho acoustique (AEC) — écoute sur haut-parleurs

Quand la sortie système est jouée sur des **haut-parleurs** (pas au casque), le
micro la recapte → la conférence se retrouve, atténuée et retardée, dans le canal
« Moi » → doublons / mauvaise attribution. Comme on a le **signal de référence
propre** (le canal loopback = ce qui est sorti), on le soustrait du micro.

- **Moteur** : `actions/audio_utils/aec.py` — `class EchoCanceller` (filtre
  adaptatif **FDAF contraint**, overlap-save, **100 % numpy**, aucune dépendance)
  + `cancel_array()` (différé, avec **alignement de délai GCC-PHAT**) +
  `estimate_delay()`. Mutualisé live ⇆ différé.
- **Normalisation NLMS SCALAIRE** (`denom ≈ L·E_bloc`) : le facteur longueur-de-
  filtre `L` est essentiel. Une normalisation *par bin* se met mal à l'échelle et
  **fait diverger** le filtre — ne pas y revenir. Cold start : amorcer la
  puissance sans lissage (sinon 1ʳᵉ MAJ ÷0.1 → overshoot).
- **Double-talk** : gel de l'adaptation UNIQUEMENT si le filtre a bien convergé
  (écho estimé ≥ 40 % du micro) ET que l'erreur explose. Seuil 0.4 critique :
  trop bas (0.05) il gèle pendant la convergence et casse tout. Le NLMS est de
  toute façon robuste à la voix proche (décorrélée → bruit de gradient moyenné).
- **Référence** = **moyenne de TOUTES les sorties** (16 kHz mono différé / débit
  natif live). Annulation appliquée **par canal micro**.
- **Live** (`live_transcribe.py`, param `aec=`) : un `EchoCanceller` stateful par
  micro, au débit natif. `process()` peut restituer < bloc (bufferise un hop) →
  **position par source** (`_src_pos`) au lieu d'une horloge globale ; `finalize`
  vide la traîne via `flush()`. Perf : RTF ≈ 0,05 (négligeable devant Whisper).
- **Différé** (`transcribe_channels.py`, flag `--aec`) : `_reference_mix()` +
  `cancel_array()` (GCC-PHAT + filtre) sur chaque canal micro avant Whisper.
- **TUI** : menu « Annulation d'écho » proposé en live SEULEMENT si micro **et**
  sortie sont capturés, et dans le différé « Par canal ». Idéal réel = **casque**
  (zéro écho) ; l'AEC est le rattrapage quand on est sur HP.
- ⚠ **Bug ffmpeg corrigé au passage** : `_demux_channel` formait `pan=mono|c0=
  (c0+c1)/2` → **syntaxe `pan` invalide** (« Invalid argument »), cassait le démux
  de toute source multi-canaux. Forme correcte = gains explicites
  `0.5*c0+0.5*c1`.

### Garde anti-écho au niveau ÉNONCÉ (live) — faux « Moi » sur haut-parleurs

Backstop de l'AEC. Même AEC active, un **résidu** d'écho HP (distorsion du
haut-parleur, réverbération, double-talk) reste parfois transcrit sur « Moi »
alors qu'on n'a rien dit — symptôme typique : une phrase de la conférence
attribuée à « Moi » sans correspondance texte sur « Système » (donc
`is_echo_duplicate`, limité aux phrases ≤4 mots quasi identiques, **ne peut rien**).

- **Principe** : pour CHAQUE énoncé MICRO finalisé, on mesure la part de son
  énergie explicable linéairement par la **référence** (mix des sorties = ce qui
  est sorti aux HP) via la **cohérence spectrale** `aec.echo_coherence(mic, ref,
  sr)` ∈ [0, 1] (≈ écho/(écho+voix)). `coh ≥ LIVE_ECHO_GATE_COH` (défaut **0.65**)
  ⇒ énoncé dominé par l'écho ⇒ **jeté** (le contenu figure déjà, PROPRE, sur
  « Système »). Câblé dans `live_transcribe.py` : `_is_echo` (worker), historique
  `_Ring` du micro **brut** + de la référence (`feed_bytes`), décision dans
  `_run_worker` **avant** Whisper (ni inférence GPU, ni faux « Moi »).
- **Pourquoi la cohérence et NON `cancel_array`+ERLE** : le filtre adaptatif NLMS
  **ne converge pas** en 1–3 s d'audio coloré (vérifié : ERLE ≈ 0,03 même sur de
  l'écho franc court). La cohérence (estimateur Welch) est **convergée par
  construction** → fiable énoncé par énoncé. ⚠ Invariante à la coloration/réverb
  mais **pas au délai** : un retard ≳ fenêtre détruit la cohérence → on **aligne
  d'abord** (GCC-PHAT). L'alignement **RETARDE** la réf pour un délai positif —
  sens **inverse** de `cancel_array` (dont le filtre causal veut la réf avancée).
- **Sûr par construction** (donc activé par défaut dès micro+sortie, **indépendant
  du flag `aec`** car c'est une DÉTECTION, pas une annulation) : au **casque** la
  vraie voix est décorrélée de la sortie → `coh≈0` → jamais jetée (le gate ne se
  déclenche QUE s'il y a vrai couplage acoustique HP). En **double-talk**, la voix
  proche ajoute de l'énergie décorrélée → `coh` retombe sous le seuil dès que la
  voix ≳ l'écho → énoncé gardé. Calibration (bruit band-pass + chemin d'écho
  réverbéré) : écho pur ≈ 0,89 · voix seule ≈ 0,02–0,10 · double-talk voix≥écho ≈
  0,08 · voix=écho ≈ 0,50–0,57. Seuil 0,65 = jeter seulement si l'écho domine
  (≥ ~65 % de l'énergie) → attrape le bug, protège la vraie parole.
- **Réglages `.env`** : `LIVE_ECHO_GATE` (1/0, défaut 1) · `LIVE_ECHO_GATE_COH`
  (défaut 0,65 ; plus BAS = plus agressif, 0,55–0,75). Écrits par `setup_env.py`.
- **Différé** : pas (encore) câblé — `transcribe_channels.py --aec` annule l'écho
  sur tout le canal (le filtre **converge** sur des minutes) puis filtre le texte ;
  `echo_coherence` est mutualisable si on veut y ajouter le même garde par segment.

---

## Recurring problems

### FLAC multicanal : VLC (ou un lecteur) ne joue que la piste micro
**Symptom**: on ouvre `<name>.flac` dans VLC et on n'entend QUE le micro (« Moi »),
pas la sortie système (« Système ») → on croit que **le fichier ne contient pas
toutes les pistes**.
**Cause**: le recorder écrit un FLAC **N canaux étiqueté layout surround** (ex. 4
canaux → `4.0` = FL/FR/BL/BR). Sur une sortie **stéréo** (casque / 2 enceintes),
VLC downmixe et **atténue ou abandonne la paire arrière** — or les canaux de
SORTIE système y atterrissent → seul l'avant (micro) est audible. **Le fichier
contient bien toutes les pistes** ; c'est un artefact de downmix, pas une perte.
**Vérifier (PAS à l'oreille dans VLC)** :
- `ffprobe -v error -show_entries stream=channels,channel_layout <flac>` → nb de canaux.
- niveaux PAR canal : `ffmpeg -i <flac> -af astats=metadata=1:reset=0 -f null -`
  → un canal de sortie capturé a un `RMS level dB` franc (souvent plus fort que le
  micro). Cas vu : micro ch1-2 ≈ −36 dB, sortie ch3-4 ≈ −21 dB.
- le `.srt`/`.md` à côté contient déjà des lignes « Système » → preuve que la
  sortie a été captée ET transcrite.
**Écouter la sortie seule / un mix** (canaux 0-indexés ; sortie = `c2,c3` ici) :
`ffmpeg -i <flac> -filter_complex "pan=stereo|c0=c2|c1=c3" systeme.flac` ; mix
Moi+Système : `pan=stereo|c0=0.5*c0+0.5*c2|c1=0.5*c1+0.5*c3`. Le mapping exact
canal→source est dans `<name>.channels.json`. **La transcription par canal n'est
PAS affectée** (elle indexe les canaux via le sidecar, pas via le layout).
*Amélioration possible (non faite)* : écrire le FLAC en layout **discret/unknown**
plutôt que `4.0` pour que les lecteurs ne droppent pas la paire « arrière ».

### WSL2 `/mnt/c/` path disappears mid-session
**Symptom**: `OSError: [Errno 22] Invalid argument` on `os.listdir()` after returning from an action.
**Cause**: Nextcloud (or Windows) renames/moves the folder during sync while the TUI holds the path.
**Fix**: `_browser_entries` retries `os.listdir()` up to 4 times (0.4 s apart) before giving up. On persistent failure it shows a visible warning row instead of silently appearing empty. `browse()` checks `os.path.isdir(current)` at the top of the loop and walks up to parent if the directory is truly gone.

### Ollama reloads model on every inference call
**Symptom**: High CPU / SSD thrashing mid-inference; VRAM shows model loading again.
**Cause**: `/v1/chat/completions` sent without `options` → Ollama sees default `num_ctx=262144`, different from load-time options, and reloads.
**Fix**: `OllamaBackend` tracks `_active_options` set at `load()`; `get_inference_extra_body()` returns them; every inference call injects them via `extra_body`.

### `num_gpu` via Ollama API options not applied for vision models
**Symptom**: `qwen2.5vl:7b` overflows VRAM regardless of `OLLAMA_OCR_NUM_GPU` value.
**Cause**: (1) Ollama may ignore `num_gpu` in API options for some model families. (2) The vision encoder (mmproj/SigLIP) always loads fully to GPU regardless of `num_gpu`.
**Fix**: Use a Modelfile with `PARAMETER num_gpu N` baked in, or switch to a smaller vision model (`qwen2.5vl:3b`). `OLLAMA_NUM_GPU` env var only applies when the Ollama server starts, not per-request.

### Ollama service broken / falls back to CPU / "no models" hang on WSL2
**DO NOT touch the Ollama systemd unit or run `ollama serve` manually without reading this first.** A bad config here cost the user a full reinstall and hours of debugging — treat changes here as risky and confirm before acting.

**Invariants on this machine**:
- Service unit: `/etc/systemd/system/ollama.service` (upstream, runs as user `ollama`).
- Models live at `/usr/share/ollama/.ollama/models/` (owned by the `ollama` user). Do **not** assume `~/.ollama/models/`.
- `nvidia-smi` on WSL2 is at `/usr/lib/wsl/lib/nvidia-smi`, **not** `/usr/bin/nvidia-smi`.
- The upstream service detects the GPU on its own — no `ExecStartPre` GPU check is needed.

**Trap 1 — bad `ExecStartPre` in override.conf**:
- Symptom: `systemctl status ollama` shows `activating (auto-restart)` and `Process: … ExecStartPre=/usr/bin/nvidia-smi -L (code=exited, status=203/EXEC)`. The service never starts.
- Cause: an `/etc/systemd/system/ollama.service.d/override.conf` with `ExecStartPre=/usr/bin/nvidia-smi -L` — wrong path on WSL2.
- Fix: either remove the override (`sudo rm /etc/systemd/system/ollama.service.d/override.conf && sudo systemctl daemon-reload && sudo systemctl restart ollama`) or correct the path to `/usr/lib/wsl/lib/nvidia-smi`.

**Trap 2 — manually-launched `ollama serve` masks the broken service**:
- Symptom: API responds at `127.0.0.1:11434` but `/api/tags` returns `{"models":[]}`; AI transforms hang at "Load model" forever.
- Cause: when systemd ollama fails, running `ollama serve` as your own user starts a second instance that reads from `~/.ollama/models/` (empty) instead of `/usr/share/ollama/.ollama/models/`. It also runs without the render/video group membership and the WSL nvidia lib path, so **CUDA is not detected → everything runs on CPU**.
- Detect: `ps -ef | grep '[o]llama serve'` — if the user is `charles…` (not `ollama`), it's the rogue manual instance.
- Fix: kill the manual `ollama serve`, fix the systemd unit (Trap 1), then `sudo systemctl restart ollama`. Verify with `curl -fsS http://127.0.0.1:11434/api/tags` (should list `gemma4`, `deepseek-ocr`, `qwen2.5vl`, etc.) and check ollama logs for `library=cuda` rather than `library=cpu`.

**Diagnostic checklist before doing anything else when ollama "isn't working"**:
1. `systemctl is-active ollama` — must be `active`, not `activating`.
2. `ps -ef | grep '[o]llama serve'` — exactly one process, owned by `ollama` user.
3. `curl -fsS http://127.0.0.1:11434/api/tags` — must list the expected models.
4. `journalctl -u ollama --no-pager -n 50 | grep -iE 'cuda|cpu|library='` — should show `cuda` runner, not `cpu`.
