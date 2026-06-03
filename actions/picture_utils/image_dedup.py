#!/usr/bin/env python3
"""Détection de doublons d'images.

**Passe 1 — EXACT (implémentée ici)** : fichiers *byte-identiques* via SHA-256.
Zéro faux positif (2 hash égaux ⇒ même fichier). Aucune dépendance externe
(stdlib uniquement) → ne nécessite pas de rebuild de l'image utils.

**Passe 2 — perceptuel** (`--method perceptual`) : near-duplicates (mêmes photos
ré-encodées / redimensionnées) via `imagehash` (regroupement par distance de
Hamming ≤ `--threshold`). ⚠️ faux positifs possibles → chaque groupe porte
`decision: 'pending'` (à valider via revue humaine, jamais supprimer en aveugle).
Nécessite `imagehash` + Pillow dans l'environnement (image utils).

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


def _hash_to_int(h):
    """imagehash (matrice booléenne N×N) -> entier, pour banding + popcount."""
    v = 0
    for b in h.hash.flatten():
        v = (v << 1) | int(bool(b))
    return v


def find_perceptual_duplicates(root, hash_method='dhash', threshold=5,
                               extensions=IMAGE_EXTENSIONS, workers=8):
    """Groupes de *near-duplicates* par perceptual hash (imagehash).

    Détecte les mêmes photos ré-encodées / redimensionnées / EXIF modifiées — que
    l'exact rate. ⚠️ Peut produire des FAUX POSITIFS → groupes à **valider** (champ
    `decision: 'pending'`), jamais à supprimer en aveugle. Regroupe par distance de
    Hamming ≤ `threshold` (0 = empreinte identique). Nécessite `imagehash` + Pillow."""
    try:
        import imagehash
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            f"Le mode perceptuel nécessite 'imagehash' + 'Pillow' (absent : {e}). "
            "Les ajouter à l'image utils (rebuild).")

    hashfn = {
        'ahash': imagehash.average_hash, 'phash': imagehash.phash,
        'dhash': imagehash.dhash,        'whash': imagehash.whash,
    }.get(hash_method)
    if hashfn is None:
        raise ValueError(f"hash perceptuel inconnu : {hash_method!r}")

    root = os.path.abspath(root)
    paths = list(_iter_images(root, extensions))

    def _h(p):
        try:
            with Image.open(p) as im:
                return p, hashfn(im)
        except Exception:
            return p, None

    hashes = {}  # path -> imagehash
    if paths:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for p, h in ex.map(_h, paths):
                if h is not None:
                    hashes[p] = h

    # Union-find sur les empreintes DISTINCTES (bien moins nombreuses que les images).
    distinct = {str(h): h for h in hashes.values()}
    keys = list(distinct)
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if threshold > 0 and len(keys) > 1:
        # Multi-Index Hashing : K = threshold+1 bandes. Deux empreintes à distance
        # de Hamming ≤ threshold partagent forcément ≥1 bande identique (pigeonhole)
        # → on ne compare en plein (popcount) que les candidats partageant une bande.
        # O(n) en pratique (vs O(n²) ou BK-tree lent sur données uniformes).
        nbits = int(distinct[keys[0]].hash.size)
        ints = {k: _hash_to_int(distinct[k]) for k in keys}
        K = threshold + 1
        bounds = [round(nbits * i / K) for i in range(K + 1)]
        for bi in range(K):
            lo, hi = bounds[bi], bounds[bi + 1]
            if hi <= lo:
                continue
            shift = nbits - hi
            mask = (1 << (hi - lo)) - 1
            table = {}
            for k in keys:
                table.setdefault((ints[k] >> shift) & mask, []).append(k)
            for bucket in table.values():
                for i in range(len(bucket)):
                    ki = bucket[i]
                    for j in range(i + 1, len(bucket)):
                        kj = bucket[j]
                        ra, rb = find(ki), find(kj)
                        if ra != rb and (ints[ki] ^ ints[kj]).bit_count() <= threshold:
                            parent[ra] = rb

    clusters = {}
    for p, h in hashes.items():
        clusters.setdefault(find(str(h)), []).append(p)

    groups = []
    for members in clusters.values():
        if len(members) > 1:
            rels = sorted(os.path.relpath(p, root).replace(os.sep, '/') for p in members)
            gid = hashlib.sha1('\n'.join(rels).encode('utf-8')).hexdigest()[:12]
            groups.append({
                'id':        gid,         # id stable (hash des chemins) pour la revue
                'method':    hash_method,
                'threshold': threshold,
                'paths':     rels,
                'decision':  'pending',   # à valider (anti faux-positif)
            })
    groups.sort(key=lambda g: g['paths'][0])
    return groups


def build_report(root, method='exact', hash_method='dhash', threshold=5, workers=8):
    """Rapport JSON-able : méta + groupes de doublons (exact ou perceptuel)."""
    if method == 'perceptual':
        groups = find_perceptual_duplicates(root, hash_method=hash_method,
                                            threshold=threshold, workers=workers)
        method_label = f'perceptual:{hash_method}'
    else:
        groups = find_exact_duplicates(root, workers=workers)
        method_label = 'sha256'
    report = {
        'method': method_label,
        'root': os.path.abspath(root),
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'group_count': len(groups),
        'redundant_files': sum(len(g['paths']) - 1 for g in groups),
        'groups': groups,
    }
    if method == 'perceptual':
        report['threshold'] = threshold
    return report


def main():
    ap = argparse.ArgumentParser(
        description="Doublons d'images → JSON des groupes de chemins relatifs.")
    ap.add_argument('root', help='Dossier racine des photos (parcours récursif)')
    ap.add_argument('-o', '--output',
                    help='Chemin du JSON de sortie (défaut: <root>/duplicates.json)')
    ap.add_argument('--method', choices=['exact', 'perceptual'], default='exact',
                    help='exact = SHA-256 (0 faux positif) ; perceptual = near-duplicates '
                         '(imagehash, à valider). Défaut: exact.')
    ap.add_argument('--phash', choices=['ahash', 'phash', 'dhash', 'whash'], default='dhash',
                    help='Algo perceptuel (défaut dhash). Ignoré si --method exact.')
    ap.add_argument('--threshold', type=int, default=5,
                    help='Distance de Hamming max (perceptuel ; 0 = identique). Défaut 5.')
    ap.add_argument('--workers', type=int, default=8, help='Threads (I/O, défaut 8)')
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"Dossier introuvable : {args.root}")

    try:
        report = build_report(args.root, method=args.method, hash_method=args.phash,
                              threshold=args.threshold, workers=args.workers)
    except (RuntimeError, ValueError) as e:
        sys.exit(f"Erreur : {e}")
    out = args.output or os.path.join(args.root, 'duplicates.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[{report['method']}] {report['group_count']} groupe(s), "
          f"{report['redundant_files']} fichier(s) redondant(s) → {out}")


if __name__ == '__main__':
    main()
