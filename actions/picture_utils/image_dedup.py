#!/usr/bin/env python3
"""Détection de doublons d'images.

**Passe 1 — EXACT (implémentée ici)** : fichiers *byte-identiques* via SHA-256.
Zéro faux positif (2 hash égaux ⇒ même fichier). Aucune dépendance externe
(stdlib uniquement) → ne nécessite pas de rebuild de l'image utils.

**Passe 2 — perceptuel (à venir)** : near-duplicates (mêmes photos ré-encodées
/ redimensionnées) via `imagehash` + revue humaine. Le JSON est conçu pour être
*vivant* : la passe perceptuelle pourra ajouter des groupes `method=dhash` avec
une `distance` et un champ `decision` (pending/duplicate/distinct) rempli par
l'interface de revue.

Algorithme exact (efficace, pas de O(n²)) :
  1. parcours récursif, collecte (chemin, taille) des images ;
  2. pré-filtre par **taille** : un fichier dont la taille est unique ne peut pas
     avoir de doublon → on ne le hashe même pas (gros gain d'I/O) ;
  3. SHA-256 (en threads, I/O-bound) des seuls candidats (taille partagée) ;
  4. regroupement par hash ; un groupe de ≥2 chemins = doublons.

CLI (appelée par le futur DAG via DockerOperator) :
    python image_dedup.py <racine_photos> [-o sortie.json] [--workers N]
→ écrit un JSON des groupes de chemins **relatifs** identiques, par défaut à la
  racine des photos (`<racine>/duplicates.json`).
"""
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Extensions considérées comme « photo » (images + RAW). Le hash exact marche sur
# n'importe quel fichier, mais on cible les photos pour ne pas hasher du bruit.
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff', '.webp',
    '.heic', '.heif',
    '.rw2', '.dng', '.cr2', '.cr3', '.nef', '.arw', '.orf', '.raf', '.raw',
}


def file_sha256(path, chunk_size=1 << 20):
    """SHA-256 du contenu brut du fichier (par blocs ; aucun décodage image)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def are_identical(path_a, path_b):
    """Vrai si les 2 fichiers sont *byte-identiques* (exact).

    Primitive « est-ce que A et B sont identiques » demandée. Pré-filtre par
    taille (rapide) avant de hasher. Pour comparer une *collection* entière,
    préférer `find_exact_duplicates` (O(n), pas O(n²))."""
    try:
        if os.path.getsize(path_a) != os.path.getsize(path_b):
            return False
    except OSError:
        return False
    if os.path.abspath(path_a) == os.path.abspath(path_b):
        return True
    return file_sha256(path_a) == file_sha256(path_b)


def _iter_images(root, extensions):
    for dirpath, _dirs, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in extensions:
                yield os.path.join(dirpath, name)


def find_exact_duplicates(root, extensions=IMAGE_EXTENSIONS, workers=8):
    """Renvoie la liste des groupes de doublons exacts sous `root`.

    Chaque groupe : {'hash': <sha256>, 'size': <octets>, 'paths': [rel, ...]}
    (≥2 chemins relatifs, triés)."""
    root = os.path.abspath(root)

    # 1) collecte (path, size)
    by_size = {}
    for path in _iter_images(root, extensions):
        try:
            by_size.setdefault(os.path.getsize(path), []).append(path)
        except OSError:
            continue

    # 2) seuls les fichiers à taille partagée peuvent être des doublons
    candidates = [p for paths in by_size.values() if len(paths) > 1 for p in paths]

    # 3) hash des candidats (I/O-bound → threads)
    def _h(p):
        try:
            return p, file_sha256(p)
        except OSError:
            return p, None

    hashes = {}
    if candidates:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for p, hx in ex.map(_h, candidates):
                if hx:
                    hashes[p] = hx

    # 4) regroupement par hash
    by_hash = {}
    for p, hx in hashes.items():
        by_hash.setdefault(hx, []).append(p)

    groups = []
    for hx, paths in by_hash.items():
        if len(paths) > 1:
            groups.append({
                'hash': hx,
                'size': os.path.getsize(paths[0]),
                'paths': sorted(os.path.relpath(p, root).replace(os.sep, '/') for p in paths),
            })
    groups.sort(key=lambda g: g['paths'][0])
    return groups


def build_report(root, workers=8):
    """Rapport JSON-able : méta + groupes de doublons exacts."""
    groups = find_exact_duplicates(root, workers=workers)
    return {
        'method': 'sha256',
        'root': os.path.abspath(root),
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'group_count': len(groups),
        'redundant_files': sum(len(g['paths']) - 1 for g in groups),
        'groups': groups,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Doublons d'images EXACTS (SHA-256) → JSON des groupes de chemins relatifs.")
    ap.add_argument('root', help='Dossier racine des photos (parcours récursif)')
    ap.add_argument('-o', '--output',
                    help='Chemin du JSON de sortie (défaut: <root>/duplicates.json)')
    ap.add_argument('--workers', type=int, default=8,
                    help='Threads de hashage (I/O-bound, défaut 8)')
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"Dossier introuvable : {args.root}")

    report = build_report(args.root, workers=args.workers)
    out = args.output or os.path.join(args.root, 'duplicates.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"{report['group_count']} groupe(s) de doublons exacts, "
          f"{report['redundant_files']} fichier(s) redondant(s) → {out}")


if __name__ == '__main__':
    main()
