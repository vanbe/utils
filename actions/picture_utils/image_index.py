#!/usr/bin/env python3
"""Index de hash des images — « base de données » de dédup sur le filesystem.

Construit / maintient un index JSON `{ chemin_relatif → {size, mtime, sha256} }`
des images sous **une ou plusieurs racines** (ex. la collection `Photos` **et** le
dossier de tri `A Ranger`), en **excluant** éventuellement des sous-dossiers.
L'index reflète la **bibliothèque de référence** (« ce que j'ai déjà »).

But : permettre à un autre process (ex. le DAG de tri `rename_unloaded_media_smb`)
de **mettre de côté** une image entrante déjà présente — via un lookup O(1) de son
SHA-256 dans l'index, sans re-scanner toute la bibliothèque.

Multi-racines : les chemins sont relatifs à une **base commune** (`--base`, défaut =
ancêtre commun des racines) → pas d'ambiguïté entre racines (`Photos/…` vs
`A Classer/A Ranger/…`).

Choix d'archi :
  - **JSON** (pas SQLite) : robuste sur FS réseau (CIFS/SMB), pas de verrou. ~100k
    images ≈ 15-20 Mo, chargé en ~1-2 s. Au-delà de ~500k, envisager un format binaire.
  - **Incrémental** : un fichier déjà indexé dont **taille + mtime** n'ont pas changé
    n'est PAS re-hashé. Le **1ᵉʳ build** (tout hasher) est lourd ; les suivants ~gratuits.

CLI (appelable par un DAG via DockerOperator) :
    python image_index.py <racine> [<racine2> ...] [--base <dir>]
                          [--exclude <sous-chemin> ...] [-o index.json]
                          [--workers N] [--rebuild]
→ par défaut écrit `<base>/.image_index.json`.

Helpers consommateurs : `load_index`, `known_hashes`, `hash_to_paths`.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Réutilise la fonction de hash et la liste d'extensions de image_dedup (source unique).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_dedup import file_sha256, IMAGE_EXTENSIONS


def _norm(p):
    return p.replace(os.sep, '/')


def _is_excluded(rel, excluded):
    """rel et excluded : chemins relatifs '/'-séparés, sans slash de tête/queue."""
    for ex in excluded:
        if ex and (rel == ex or rel.startswith(ex + '/')):
            return True
    return False


def _default_base(roots):
    roots = [os.path.abspath(r) for r in roots]
    return roots[0] if len(roots) == 1 else os.path.commonpath(roots)


def build_index(roots, base=None, excluded=(), workers=8, previous=None):
    """Construit l'index sur une ou plusieurs racines (incrémental si `previous`)."""
    roots = [os.path.abspath(r) for r in roots]
    base = os.path.abspath(base) if base else _default_base(roots)
    excluded = [_norm(e).strip('/') for e in excluded]
    prev_files = (previous or {}).get('files', {})

    files = {}
    to_hash = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = _norm(os.path.relpath(dirpath, base))
            if rel_dir == '.':
                rel_dir = ''
            dirnames[:] = [d for d in dirnames
                           if not _is_excluded((rel_dir + '/' + d).strip('/'), excluded)]
            for name in filenames:
                if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
                    continue
                full = os.path.join(dirpath, name)
                rel = _norm(os.path.relpath(full, base))
                if _is_excluded(rel, excluded) or rel in files:
                    continue
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                size, mtime = st.st_size, st.st_mtime
                prev = prev_files.get(rel)
                if (prev and prev.get('sha256')
                        and prev.get('size') == size
                        and abs(prev.get('mtime', -1.0) - mtime) < 1e-6):
                    files[rel] = {'size': size, 'mtime': mtime, 'sha256': prev['sha256']}
                else:
                    files[rel] = {'size': size, 'mtime': mtime, 'sha256': None, '_full': full}
                    to_hash.append(rel)

    def _h(rel):
        try:
            return rel, file_sha256(files[rel]['_full'])
        except OSError:
            return rel, None

    if to_hash:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for rel, hx in ex.map(_h, to_hash):
                if hx:
                    files[rel]['sha256'] = hx
                else:
                    files.pop(rel, None)

    for rec in files.values():
        rec.pop('_full', None)
    files = {k: v for k, v in files.items() if v.get('sha256')}

    return {
        'version': 1,
        'method': 'sha256',
        'base': base,
        'roots': roots,
        'excluded': excluded,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'count': len(files),
        'hashed_new': len(to_hash),
        'files': files,
    }


# --- helpers consommateurs (ex. DAG de tri) ---

def load_index(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def known_hashes(index):
    """Ensemble des SHA-256 de la bibliothèque (lookup O(1) : `h in known_hashes(idx)`)."""
    return {rec['sha256'] for rec in index.get('files', {}).values()}


def hash_to_paths(index):
    """Map SHA-256 → [chemins relatifs] (pour savoir *où* est le doublon)."""
    m = {}
    for rel, rec in index.get('files', {}).items():
        m.setdefault(rec['sha256'], []).append(rel)
    return m


def main():
    ap = argparse.ArgumentParser(
        description="Index SHA-256 d'une bibliothèque d'images (multi-racines, incrémental).")
    ap.add_argument('roots', nargs='+', help='Une ou plusieurs racines à indexer')
    ap.add_argument('--base',
                    help='Base des chemins relatifs (défaut: ancêtre commun des racines)')
    ap.add_argument('--exclude', action='append', default=[],
                    help='Sous-chemin relatif (à la base) à exclure (répétable)')
    ap.add_argument('-o', '--output',
                    help='Chemin de l\'index JSON (défaut: <base>/.image_index.json)')
    ap.add_argument('--workers', type=int, default=8, help='Threads de hashage (I/O)')
    ap.add_argument('--rebuild', action='store_true',
                    help='Ignore l\'index existant (full rebuild, pas d\'incrémental)')
    args = ap.parse_args()

    for r in args.roots:
        if not os.path.isdir(r):
            sys.exit(f"Racine introuvable : {r}")

    base = os.path.abspath(args.base) if args.base else _default_base(args.roots)
    out = args.output or os.path.join(base, '.image_index.json')

    previous = None
    if not args.rebuild and os.path.isfile(out):
        try:
            previous = load_index(out)
        except Exception:
            previous = None

    index = build_index(args.roots, base=base, excluded=args.exclude,
                        workers=args.workers, previous=previous)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print(f"{index['count']} image(s) indexée(s), {index['hashed_new']} (re)hashée(s) "
          f"→ {out}")


if __name__ == '__main__':
    main()
