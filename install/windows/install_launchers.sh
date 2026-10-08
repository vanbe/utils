#!/bin/bash
# install_launchers.sh — installe les lanceurs Windows de utils_tools.
#
# À lancer DEPUIS WSL (c'est là que vit le dépôt). Le script :
#   1. copie utils_tools.bat / utils_tools.ps1 dans %USERPROFILE%\bin ;
#   2. y INJECTE la distro WSL et le chemin réel du dépôt — les lanceurs
#      n'ont donc plus à supposer ~/code/utils ni la distro par défaut ;
#   3. ajoute %USERPROFILE%\bin au PATH utilisateur Windows (pas besoin d'admin,
#      contrairement au PATH machine).
#
# Idempotent : relancer écrase les lanceurs et ne duplique pas l'entrée PATH.
#
# Usage: bash install/windows/install_launchers.sh [--yes] [--dry-run]

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
UTILS_DIR="$(cd "$HERE/../.." && pwd)"

ASSUME_YES=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y)  ASSUME_YES=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "Option inconnue : $arg" >&2; exit 2 ;;
    esac
done

if ! grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
    echo "Ce script ne sert que sous WSL (lanceurs Windows)." >&2
    exit 1
fi

PS='powershell.exe'
if ! command -v "$PS" >/dev/null 2>&1; then
    echo "⚠ powershell.exe introuvable — l'interop Windows est-elle activée ?" >&2
    exit 1
fi

DISTRO="${WSL_DISTRO_NAME:-}"
if [ -z "$DISTRO" ]; then
    echo "⚠ WSL_DISTRO_NAME vide — les lanceurs utiliseront la distro par défaut."
fi

# %USERPROFILE% est lu côté Windows : c'est la seule source fiable (le nom
# d'utilisateur Windows n'a aucune raison d'égaler celui de WSL).
WIN_HOME="$(cd /mnt/c && "$PS" -NoProfile -NonInteractive -Command '$env:USERPROFILE' 2>/dev/null | tr -d '\r')"
if [ -z "$WIN_HOME" ]; then
    echo "✗ Impossible de résoudre %USERPROFILE% côté Windows." >&2
    exit 1
fi
WIN_BIN="$WIN_HOME\\bin"
WSL_BIN="$(wslpath -u "$WIN_BIN" 2>/dev/null)"
if [ -z "$WSL_BIN" ]; then
    echo "✗ Impossible de convertir $WIN_BIN en chemin WSL." >&2
    exit 1
fi

echo "Lanceurs Windows"
echo "  distro WSL   : ${DISTRO:-<défaut>}"
echo "  dépôt utils  : $UTILS_DIR"
echo "  destination  : $WIN_BIN"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "  (dry-run — rien n'est écrit)"
    exit 0
fi

if [ "$ASSUME_YES" -eq 0 ] && [ -t 0 ]; then
    read -r -p "  Installer les lanceurs et ajouter ce dossier au PATH Windows ? [O/n] " ans
    case "${ans:-o}" in
        [nN]*) echo "  Sauté."; exit 0 ;;
    esac
fi

mkdir -p "$WSL_BIN" || { echo "✗ Création de $WIN_BIN impossible." >&2; exit 1; }

# Injection de la configuration dans les copies installées. Les fichiers du
# dépôt restent génériques (valeurs vides = auto-détection), seule la copie
# déposée sous Windows est spécialisée pour cette machine.
#
# Les copies sont écrites en CRLF : c'est ce qu'attendent cmd.exe et
# PowerShell, et le dépôt (côté WSL) reste en LF.
crlf() { sed -e 's/$/\r/'; }

sed -e "s|^set \"UTILS_DISTRO=\"|set \"UTILS_DISTRO=$DISTRO\"|" \
    -e "s|^set \"UTILS_DIR=\"|set \"UTILS_DIR=$UTILS_DIR\"|" \
    "$HERE/utils_tools.bat" | crlf > "$WSL_BIN/utils_tools.bat" \
    || { echo "✗ Copie du .bat impossible." >&2; exit 1; }

sed -e "s|^\$UtilsDistro = ''|\$UtilsDistro = '$DISTRO'|" \
    -e "s|^\$UtilsDir    = ''|\$UtilsDir    = '$UTILS_DIR'|" \
    "$HERE/utils_tools.ps1" | crlf > "$WSL_BIN/utils_tools.ps1" \
    || { echo "✗ Copie du .ps1 impossible." >&2; exit 1; }

# Garde-fou : un caractère non-ASCII dans le .bat suffit à casser cmd.exe (il
# reprend la lecture du fichier à un OFFSET OCTET mémorisé — un caractère
# multi-octets décale tout et il exécute des morceaux de lignes).
if LC_ALL=C grep -q '[^[:print:][:space:]]' "$WSL_BIN/utils_tools.bat"; then
    echo "  ⚠ le .bat installé contient des caractères non-ASCII — cmd.exe va"
    echo "    mal l'interpréter. Garder install/windows/utils_tools.bat en 7-bit."
fi

echo "  → $WIN_BIN\\utils_tools.bat  (CMD)"
echo "  → $WIN_BIN\\utils_tools.ps1  (PowerShell)"

# Lanceurs `record` : mêmes fichiers, l'appel passe --record (Record audio direct dans
# le dossier Windows courant). Dérivés des copies déjà spécialisées ci-dessus.
sed -e 's|--workdir "!WSL_CWD!"|--record --workdir "!WSL_CWD!"|' \
    -e 's|^:: Utils Tools - Windows launcher (CMD)|:: Utils Tools - record launcher (CMD) - Record audio in current folder|' \
    "$WSL_BIN/utils_tools.bat" > "$WSL_BIN/record.bat" \
    || { echo "✗ Copie de record.bat impossible." >&2; exit 1; }
sed -e 's|--workdir \$wslPath|--record --workdir $wslPath|' \
    -e 's|^# Utils Tools - Windows PowerShell launcher|# Utils Tools - record launcher (PowerShell) - Record audio in current folder|' \
    "$WSL_BIN/utils_tools.ps1" > "$WSL_BIN/record.ps1" \
    || { echo "✗ Copie de record.ps1 impossible." >&2; exit 1; }
grep -q -- '--record' "$WSL_BIN/record.bat" && grep -q -- '--record' "$WSL_BIN/record.ps1" \
    || echo "  ⚠ record.bat / record.ps1 : option --record non injectée (vérifier le sed)"
echo "  → $WIN_BIN\\record.bat / record.ps1  (Record audio dans le dossier courant)"

# PATH utilisateur (HKCU) — aucun droit admin requis, et on ne touche jamais au
# PATH machine. L'entrée n'est ajoutée que si elle est absente.
PATH_STATUS="$(cd /mnt/c && "$PS" -NoProfile -NonInteractive -Command "
\$bin = '$WIN_BIN'
\$p = [Environment]::GetEnvironmentVariable('PATH','User')
if (\$null -eq \$p) { \$p = '' }
if ((\$p -split ';') -contains \$bin) {
    'already'
} else {
    \$new = (\$p.TrimEnd(';'))
    if (\$new) { \$new = \$new + ';' + \$bin } else { \$new = \$bin }
    [Environment]::SetEnvironmentVariable('PATH', \$new, 'User')
    'added'
}
" 2>&1 | tr -d '\r' | tail -1)"

case "$PATH_STATUS" in
    already) echo "  → déjà dans le PATH utilisateur Windows" ;;
    added)   echo "  → ajouté au PATH utilisateur Windows (rouvrir le terminal)" ;;
    *)       echo "  ⚠ PATH non modifié : $PATH_STATUS"
             echo "    À faire à la main (PowerShell) :"
             echo "      [Environment]::SetEnvironmentVariable('PATH', \$env:PATH + ';$WIN_BIN', 'User')" ;;
esac

echo ""
echo "  Depuis n'importe quel dossier Windows (après redémarrage du terminal) :"
echo "      utils_tools"
echo "      record          (enregistre directement dans le dossier courant)"
