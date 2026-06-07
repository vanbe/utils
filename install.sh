#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
BIN_DIR="$HOME/.local/bin"
CMD_NAME="utils_tools"
CMD_PATH="$BIN_DIR/$CMD_NAME"

# ── System dependencies ───────────────────────────────────────────────────────
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y build-essential cmake git ffmpeg python3.12-venv \
    pandoc \
    texlive-xetex texlive-fonts-recommended texlive-latex-extra \
    libreoffice \
    ocrmypdf \
    libimage-exiftool-perl \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra

# ── Audio recorder backend ("Record audio") ──────────────────────────────────
# Sous WSL : la capture de la sortie système passe par le binaire WASAPI
# autonome capture.exe (non versionné) → on installe le toolchain mingw et on le
# construit ici. Sur Linux natif : la capture est locale (PulseAudio), parec sert
# aux vumètres.
if grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
    echo ""
    echo "WSL detected — building Windows capture binary (WASAPI loopback)…"
    sudo apt install -y g++-mingw-w64-x86-64
    if bash "$SCRIPT_DIR/actions/audio_utils/capture/build.sh"; then
        echo "  → actions/audio_utils/bin/capture.exe"
    else
        echo "  ⚠ capture.exe build failed — 'Record audio' will offer to rebuild later"
    fi
else
    echo ""
    echo "Installing PulseAudio utils (parec — audio level meters)…"
    sudo apt install -y pulseaudio-utils || true
fi

# ── Python virtual environment ────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

. "$VENV_DIR/bin/activate"
pip3 install --upgrade pip
pip3 install -r "$SCRIPT_DIR/requirements.txt"

# ── Auto-configure .env ───────────────────────────────────────────────────────
echo ""
echo "Configuring .env with detected hardware settings..."
python3 "$SCRIPT_DIR/setup_env.py"

# ── Install utils_tools command ───────────────────────────────────────────────
echo ""
echo "Installing $CMD_NAME command..."
mkdir -p "$BIN_DIR"

cat > "$CMD_PATH" << EOF
#!/bin/bash
exec "$VENV_DIR/bin/python3" "$SCRIPT_DIR/utils_tools.py" "\$@"
EOF
chmod +x "$CMD_PATH"
echo "  → $CMD_PATH"

# Add ~/.local/bin to PATH in ~/.bashrc if not already present
if ! grep -q 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null; then
    echo "" >> "$HOME/.bashrc"
    echo '# utils_tools' >> "$HOME/.bashrc"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    echo "  → added ~/.local/bin to PATH in ~/.bashrc"
    echo "  → run: source ~/.bashrc  (or open a new terminal)"
fi

# Also add to ~/.zshrc if zsh is present
if [ -f "$HOME/.zshrc" ] && ! grep -q 'HOME/.local/bin' "$HOME/.zshrc" 2>/dev/null; then
    echo "" >> "$HOME/.zshrc"
    echo '# utils_tools' >> "$HOME/.zshrc"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
    echo "  → added ~/.local/bin to PATH in ~/.zshrc"
fi

echo ""
echo "Installation complete."
echo "  utils_tools      start the TUI in the current directory"
echo "  utils_tools /path  start the TUI in a specific directory"
echo ""
echo "Set AUDIO_UTILS_HF_TOKEN and NAS_* in .env if needed."
