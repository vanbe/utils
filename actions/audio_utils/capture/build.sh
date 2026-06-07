#!/usr/bin/env bash
# Construit capture.exe (capteur WASAPI Windows) — binaire NON versionné.
# Méthode privilégiée : cross-compile depuis Linux/WSL avec mingw-w64.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
bin="$here/../bin"
out="$bin/capture.exe"
mkdir -p "$bin"

CXX="${CXX:-x86_64-w64-mingw32-g++}"
if ! command -v "$CXX" >/dev/null 2>&1; then
  echo "Erreur : compilateur '$CXX' introuvable." >&2
  echo "  Debian/Ubuntu/WSL : sudo apt-get install g++-mingw-w64-x86-64" >&2
  echo "  Ou, sous Windows : lancer build.bat (MinGW g++ ou MSVC cl)." >&2
  exit 1
fi

"$CXX" -std=c++17 -O2 -static -static-libgcc -static-libstdc++ \
  "$here/capture.cpp" -o "$out" \
  -lole32 -loleaut32 -luuid -lwinmm

echo "OK → $out"
