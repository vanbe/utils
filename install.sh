#!/bin/bash
# install.sh — installation en une passe de utils (paquets système, venv Python,
# .env, commande utils_tools, capture audio Windows, lanceurs Windows).
#
# Volontairement SANS `set -e` : une étape qui échoue (paquet apt indisponible,
# sudo refusé, build du capteur…) ne doit pas emporter tout le reste du script.
# Chaque étape rapporte son propre échec et le bilan final (--check) dit ce qui
# manque encore et pourquoi.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python3"
BIN_DIR="$HOME/.local/bin"
CMD_NAME="utils_tools"
CMD_PATH="$BIN_DIR/$CMD_NAME"
PY_FALLBACK_VERSION="3.12"          # utilisé seulement via uv, si python3 absent
SELF="$SCRIPT_DIR/$(basename "$0")"

# uv s'installe dans ~/.local/bin, qui n'est pas dans le PATH d'un shell non
# interactif (ni dans le secure_path de sudo) : on l'y remet, sinon le script
# croit uv absent et bascule sur pip.
case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) PATH="$HOME/.local/bin:$PATH"; export PATH ;;
esac

DO_SYSTEM=1
DO_WINDOWS=1
CHECK_ONLY=0
TORCH_BACKEND=auto                  # auto | cpu | cuda

ORIG_ARGS=("$@")

usage() {
    cat <<'USAGE'
Usage: bash install.sh [options]

  --skip-system     Ne pas installer les paquets système (apt) — utile si tu
                    n'as pas sudo, ou s'ils sont déjà là.
  --skip-windows    Ne pas proposer l'installation des lanceurs Windows (WSL).
  --cpu             Forcer PyTorch CPU (~2 Go) même si un GPU est détecté.
  --cuda            Forcer PyTorch CUDA (~8 Go) même sans GPU détecté.
  --check           Ne rien installer : afficher seulement le bilan.
  -h, --help        Afficher cette aide.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-system)  DO_SYSTEM=0 ;;
        --skip-windows) DO_WINDOWS=0 ;;
        --cpu)          TORCH_BACKEND=cpu ;;
        --cuda)         TORCH_BACKEND=cuda ;;
        --check)        CHECK_ONLY=1 ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "Option inconnue : $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

is_wsl() { grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; }
has()    { command -v "$1" >/dev/null 2>&1; }
has_gpu() { has nvidia-smi && nvidia-smi -L >/dev/null 2>&1; }

# uv, s'il est là : on retient le chemin absolu plutôt que de refaire `has uv`
# à chaque fois (le PATH peut être bancal sous sudo).
UV_BIN=""
if has uv; then
    UV_BIN="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
elif [ -n "${SUDO_USER:-}" ]; then
    _sudo_home="$(getent passwd "$SUDO_USER" 2>/dev/null | cut -d: -f6)"
    [ -n "$_sudo_home" ] && [ -x "$_sudo_home/.local/bin/uv" ] && UV_BIN="$_sudo_home/.local/bin/uv"
fi

# ── Bascule root → utilisateur appelant ──────────────────────────────────────
#
# `sudo bash install.sh` : root n'a ni le bon $HOME (venv, .env, ~/.local/bin,
# ~/.bashrc partiraient dans /root) ni le uv de l'utilisateur. Seuls les
# paquets apt ont besoin de root : on les pose, puis on rend la main.
reexec_as_invoking_user() {     # ne revient pas si la bascule a lieu
    [ "$(id -u)" -eq 0 ] || return 0
    if [ -z "${SUDO_USER:-}" ] || [ "$SUDO_USER" = root ] || ! has sudo; then
        echo ""
        echo "⚠ Lancé en root : venv, .env et la commande $CMD_NAME seront"
        echo "  installés pour root (HOME=$HOME), et les fichiers écrits dans"
        echo "  $VENV_DIR appartiendront à root."
        echo "  Préférer : bash install.sh   (le script appelle sudo lui-même)."
        return 0
    fi
    echo ""
    echo "Paquets système posés — reprise des étapes suivantes en tant que $SUDO_USER…"
    exec sudo -u "$SUDO_USER" -H bash "$SELF" --skip-system "${ORIG_ARGS[@]}"
}

# ── Bilan (aussi accessible seul via --check) ────────────────────────────────
#
# Chaque binaire manquant est rattaché à ce qu'il casse, pour qu'un install
# partiel reste exploitable : on sait exactement quelle action ne marchera pas.

report() {
    local missing=0
    echo ""
    echo "── Bilan ───────────────────────────────────────────────────────────"

    check_bin() {   # check_bin <binaire> <ce qu'il sert>
        if has "$1"; then
            printf '  ✓ %-14s %s\n' "$1" "$2"
        else
            printf '  ✗ %-14s %s   ← MANQUANT\n' "$1" "$2"
            missing=1
        fi
    }
    check_bin ffmpeg    'audio/vidéo : record, transcription, conversions'
    check_bin ffprobe   'inspection des médias'
    check_bin pandoc    'Markdown ⇄ DOCX/PDF'
    check_bin soffice   'DOC/ODT/PPT/XLS → PDF (LibreOffice)'
    check_bin tesseract 'OCR'
    check_bin ocrmypdf  'ajout de couche OCR aux PDF'
    check_bin exiftool  'métadonnées images'
    check_bin xelatex   'export Markdown → PDF'
    if ! is_wsl; then
        check_bin parec 'vumètres (PulseAudio)'
    fi

    if [ -x "$VENV_PY" ]; then
        printf '  ✓ %-14s %s\n' 'venv' "$($VENV_PY -V 2>&1) — $VENV_DIR"
    else
        printf '  ✗ %-14s %s   ← MANQUANT\n' 'venv' "$VENV_DIR"
        missing=1
    fi

    if [ -f "$SCRIPT_DIR/.env" ]; then
        printf '  ✓ %-14s %s\n' '.env' 'configuré'
    else
        printf '  ✗ %-14s %s   ← MANQUANT\n' '.env' 'lancer setup_env.py'
        missing=1
    fi

    if [ -x "$CMD_PATH" ]; then
        printf '  ✓ %-14s %s\n' 'utils_tools' "$CMD_PATH"
    else
        printf '  ✗ %-14s %s   ← MANQUANT\n' 'utils_tools' "$CMD_PATH"
        missing=1
    fi

    if [ -x "$BIN_DIR/record" ]; then
        printf '  ✓ %-14s %s\n' 'record' "$BIN_DIR/record"
    else
        printf '  ✗ %-14s %s   ← MANQUANT\n' 'record' "$BIN_DIR/record"
        missing=1
    fi

    if is_wsl; then
        if [ -f "$SCRIPT_DIR/actions/audio_utils/bin/capture.exe" ]; then
            printf '  ✓ %-14s %s\n' 'capture.exe' 'capture audio Windows (Record audio)'
        else
            printf '  ✗ %-14s %s   ← MANQUANT\n' 'capture.exe' \
                "construire : bash actions/audio_utils/capture/build.sh"
            missing=1
        fi
    fi

    echo ""
    if [ "$missing" -eq 0 ]; then
        echo "  Tout est en place."
    else
        echo "  Éléments manquants ci-dessus — relancer install.sh une fois"
        echo "  les paquets système installés (voir la commande apt affichée)."
    fi
    echo ""
}

if [ "$CHECK_ONLY" -eq 1 ]; then
    reexec_as_invoking_user
    report
    exit 0
fi

# ── Paquets système ──────────────────────────────────────────────────────────
#
# python3-venv (non versionné) suit le python3 par défaut de la distribution.
# Épingler python3.12-venv cassait l'install sur Debian 13 (seul 3.13 y existe).

APT_PKGS=(build-essential cmake git ffmpeg python3-venv
          pandoc
          texlive-xetex texlive-fonts-recommended texlive-latex-extra
          libreoffice
          ocrmypdf
          libimage-exiftool-perl
          tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra)

# Backend d'enregistrement audio ("Record audio") :
#   * WSL   → la sortie système Windows n'est pas captable depuis Linux ; on
#             délègue au binaire WASAPI capture.exe, cross-compilé avec mingw.
#   * Linux → capture locale PulseAudio ; parec sert aux vumètres.
if is_wsl; then
    APT_PKGS+=(g++-mingw-w64-x86-64)
else
    APT_PKGS+=(pulseaudio-utils)
fi

SUDO=""
SYSTEM_DONE=0
if [ "$DO_SYSTEM" -eq 1 ]; then
    if [ "$(id -u)" -eq 0 ]; then
        SUDO=""
    elif has sudo && { sudo -n true 2>/dev/null || [ -t 0 ]; }; then
        SUDO="sudo"
    else
        # Pas de sudo utilisable (session non interactive, pas de droits…) :
        # on saute proprement au lieu de mourir sur un prompt invisible.
        DO_SYSTEM=0
        echo ""
        echo "⚠ sudo indisponible ici — étape paquets système SAUTÉE."
        echo "  À lancer toi-même dans un terminal :"
        echo ""
        echo "      sudo apt update && sudo apt install -y ${APT_PKGS[*]}"
        echo ""
    fi
fi

if [ "$DO_SYSTEM" -eq 1 ]; then
    echo "Installation des paquets système…"
    $SUDO apt update
    if $SUDO apt install -y "${APT_PKGS[@]}"; then
        SYSTEM_DONE=1
    else
        echo ""
        echo "⚠ apt install a échoué (au moins un paquet). L'installation Python"
        echo "  continue ; le bilan final dira ce qui manque."
    fi
fi

reexec_as_invoking_user

# ── Environnement virtuel Python ─────────────────────────────────────────────
#
# Deux chemins possibles, dans cet ordre :
#   1. python3 -m venv     (cas normal : python3-venv vient d'être installé)
#   2. uv venv --python X  (repli : distro sans python3 — uv le télécharge)

if [ ! -x "$VENV_PY" ]; then
    echo ""
    echo "Création de l'environnement virtuel dans $VENV_DIR…"
    rm -rf "$VENV_DIR"
    if has python3 && python3 -m venv "$VENV_DIR" 2>/dev/null; then
        :
    elif [ -n "$UV_BIN" ] && "$UV_BIN" venv --python "$PY_FALLBACK_VERSION" "$VENV_DIR"; then
        :
    else
        echo "" >&2
        echo "✗ Impossible de créer le venv : ni 'python3 -m venv' ni 'uv' ne" >&2
        echo "  fonctionnent. Installer python3-venv (apt) ou uv, puis relancer." >&2
        exit 1
    fi
fi

# ── Dépendances Python ───────────────────────────────────────────────────────
#
# PyTorch : la roue par défaut de PyPI embarque CUDA (~8 Go de wheels nvidia-*).
# Sur une machine sans GPU c'est du poids mort → on bascule sur l'index CPU
# (~2 Go). `--cpu` / `--cuda` permettent de forcer.

if [ "$TORCH_BACKEND" = auto ]; then
    if has_gpu; then TORCH_BACKEND=cuda; else TORCH_BACKEND=cpu; fi
    echo ""
    echo "PyTorch : backend '$TORCH_BACKEND' (auto — $(has_gpu && echo 'GPU NVIDIA détecté' || echo 'aucun GPU NVIDIA détecté'))."
fi

TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

echo ""
echo "Installation des dépendances Python…"
# --torch-backend n'existe que sur les uv récents : on ne l'utilise que s'il est
# reconnu, sinon un vieux uv ferait échouer toute l'install sur une option inconnue.
if [ -n "$UV_BIN" ] && "$UV_BIN" pip install --help 2>/dev/null | grep -q -- '--torch-backend'; then
    # uv gère lui-même la sélection torch (et sait résoudre l'index CPU).
    UV_BACKEND="$TORCH_BACKEND"
    [ "$UV_BACKEND" = cuda ] && UV_BACKEND=auto
    "$UV_BIN" pip install --python "$VENV_PY" --torch-backend="$UV_BACKEND" \
        -r "$SCRIPT_DIR/requirements.txt" || {
        echo "✗ Échec de l'installation des dépendances Python." >&2; exit 1; }
elif [ -n "$UV_BIN" ]; then
    "$UV_BIN" pip install --python "$VENV_PY" -r "$SCRIPT_DIR/requirements.txt" || {
        echo "✗ Échec de l'installation des dépendances Python." >&2; exit 1; }
else
    # Un venv créé par uv n'embarque pas pip : sans amorçage, tout ce qui suit
    # meurt sur « No module named pip ».
    if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
        echo "  pip absent du venv — amorçage via ensurepip…"
        "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
    fi
    if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
        echo "" >&2
        echo "✗ Le venv n'a ni pip ni ensurepip utilisable. Deux issues :" >&2
        echo "    • installer uv : curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        echo "    • ou recréer le venv : rm -rf '$VENV_DIR' && python3 -m venv '$VENV_DIR'" >&2
        echo "  puis relancer install.sh." >&2
        exit 1
    fi
    "$VENV_PY" -m pip install --upgrade pip
    if [ "$TORCH_BACKEND" = cpu ]; then
        # Poser d'abord la pile torch depuis l'index CPU : la résolution de
        # requirements.txt la trouvera déjà satisfaite et n'ira pas chercher
        # les roues CUDA de PyPI.
        "$VENV_PY" -m pip install --index-url "$TORCH_CPU_INDEX" \
            torch torchvision torchaudio torchcodec || {
            echo "✗ Échec de l'installation de PyTorch (index CPU)." >&2; exit 1; }
    fi
    "$VENV_PY" -m pip install -r "$SCRIPT_DIR/requirements.txt" || {
        echo "✗ Échec de l'installation des dépendances Python." >&2; exit 1; }
fi

# ── Configuration .env ───────────────────────────────────────────────────────
echo ""
echo "Configuration de .env (détection matérielle)…"
"$VENV_PY" "$SCRIPT_DIR/setup_env.py"

# ── Binaire de capture audio Windows (WSL) ───────────────────────────────────
#
# capture.exe n'est pas versionné : on le construit ici et on ÉPINGLE son
# SHA-256 (bin/capture.exe.sha256) — recorder.py refuse de lancer un binaire
# dont le hash a changé depuis la compilation.
if is_wsl; then
    echo ""
    echo "Construction du capteur audio Windows (loopback WASAPI)…"
    if bash "$SCRIPT_DIR/actions/audio_utils/capture/build.sh"; then
        "$VENV_PY" -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR/actions/audio_utils')
import recorder; print('  → épinglé SHA-256', recorder.pin_capture()[:16] + '…')
"
    else
        echo "  ⚠ build de capture.exe échoué — 'Record audio' proposera de le"
        echo "    reconstruire, ou : sudo apt install -y g++-mingw-w64-x86-64"
    fi
fi

# ── Commande utils_tools ─────────────────────────────────────────────────────
echo ""
echo "Installation de la commande $CMD_NAME…"
mkdir -p "$BIN_DIR"

cat > "$CMD_PATH" << EOF
#!/bin/bash
exec "$VENV_PY" "$SCRIPT_DIR/utils_tools.py" "\$@"
EOF
chmod +x "$CMD_PATH"
echo "  → $CMD_PATH"

# Commande `record` : lance directement « Record audio » dans le dossier courant
cat > "$BIN_DIR/record" << EOF
#!/bin/bash
exec "$VENV_PY" "$SCRIPT_DIR/utils_tools.py" --record "\$@"
EOF
chmod +x "$BIN_DIR/record"
echo "  → $BIN_DIR/record"

# Ajouter ~/.local/bin au PATH dans ~/.bashrc s'il n'y est pas déjà
if ! grep -q 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null; then
    {
        echo ""
        echo '# utils_tools'
        echo 'export PATH="$HOME/.local/bin:$PATH"'
    } >> "$HOME/.bashrc"
    echo "  → ~/.local/bin ajouté au PATH dans ~/.bashrc"
    echo "  → lancer : source ~/.bashrc  (ou ouvrir un nouveau terminal)"
fi

# Idem pour zsh s'il est présent
if [ -f "$HOME/.zshrc" ] && ! grep -q 'HOME/.local/bin' "$HOME/.zshrc" 2>/dev/null; then
    {
        echo ""
        echo '# utils_tools'
        echo 'export PATH="$HOME/.local/bin:$PATH"'
    } >> "$HOME/.zshrc"
    echo "  → ~/.local/bin ajouté au PATH dans ~/.zshrc"
fi

# ── Lanceurs Windows (WSL) ───────────────────────────────────────────────────
if is_wsl && [ "$DO_WINDOWS" -eq 1 ]; then
    echo ""
    bash "$SCRIPT_DIR/install/windows/install_launchers.sh"
fi

# ── Vérification ─────────────────────────────────────────────────────────────
echo ""
echo "Vérification de l'installation…"
if "$VENV_PY" -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
import utils_run, utils_tools   # importe aussi recorder + whisper_common
print('  ✓ utils_tools et utils_run s\'importent correctement')
"; then :; else
    echo "  ✗ l'import a échoué — voir la trace ci-dessus" >&2
fi

report

echo "  utils_tools          ouvre la TUI dans le dossier courant"
echo "  utils_tools /chemin  ouvre la TUI dans un dossier précis"
echo "  record               enregistre (Record audio) directement dans le dossier courant"
echo "  bash install.sh --check   refait le bilan ci-dessus"
echo ""
echo "Renseigner AUDIO_UTILS_HF_TOKEN et NAS_* dans .env si besoin."
