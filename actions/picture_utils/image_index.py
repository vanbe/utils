#!/usr/bin/env python3
"""Index de hash des images — « base de données » de dédup sur le filesystem.

Construit / maintient un index JSON `{ chemin_relatif → {size, mtime, sha256} }`
de toutes les images sous une **racine**, en **excluant** des sous-dossiers
(typiquement le dossier « à ranger » : ses éléments sont les *candidats* à trier,
pas la référence). L'index reflète donc la **bibliothèque déjà rangée**.

But : permettre à un autre process (ex. le DAG de tri `rename_unloaded_media_smb`)
de **mettre de côté** une image entrante déjà présente dans la bibliothèque — via
un simple lookup O(1) de son SHA-256 dans l'index, sans re-scanner toute la
bibliothèque à chaque passage.

Choix d'archi :
  - **JSON** (pas SQLite) : robuste sur FS réseau (CIFS/SMB), pas de verrou. Pour
    de très gros volumes (>100k images), envisager un format binaire / SQLite local.
  - **Incrémental** : si l'index existe, un fichier déjà indexé dont la **taille
    ET la mtime** n'ont pas changé n'est PAS re-hashé (réutilise son SHA-256).

CLI (appelable par un DAG via DockerOperator) :
    python image_index.py <racine> [--exclude <sous-chemin> ...]
                                   [-o index.json] [--workers N] [--rebuild]
→ par défaut écrit `<racine>/.image_index.json`.

Helpers pour les consommateurs : `load_index`, `known_hashes`, `hash_to_paths`.
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


def build_index(root, excluded=(), workers=8, previous=None):
    """Construit l'index (incrémental si `previous` fourni)."""
    root = os.path.abspath(root)
    excluded = [_norm(e).strip('/') for e in excluded]
    prev_files = (previous or {}).get('files', {})

    files = {}
    to_hash = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = _norm(os.path.relpath(dirpath, root))
        if rel_dir == '.':
            rel_dir = ''
        # élague les dossiers exclus (ne pas y descendre)
        dirnames[:] = [d for d in dirnames
                       if not _is_excluded((rel_dir + '/' + d).strip('/'), excluded)]
        for name in filenames:
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            full = os.path.join(dirpath, name)
            rel = _norm(os.path.relpath(full, root))
            if _is_excluded(rel, excluded):
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
                files[rel] = {'size': size, 'mtime': mtime, 'sha256': prev['sha256']}  # réutilisé
            else:
                files[rel] = {'size': size, 'mtime': mtime, 'sha256': None}
                to_hash.append((rel, full))

    def _h(item):
        rel, full = item
        try:
            return rel, file_sha256(full)
        except OSError:
            return rel, None

    if to_hash:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for rel, hx in ex.map(_h, to_hash):
                if hx:
                    files[rel]['sha256'] = hx
                else:
                    files.pop(rel, None)

    files = {k: v for k, v in files.items() if v.get('sha256')}  # purge des échecs
    return {
        'version': 1,
        'method': 'sha256',
        'root': root,
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
    """Ensemble des SHA-256 présents dans la bibliothèque (lookup O(1))."""
    return {rec['sha256'] for rec in index.get('files', {}).values()}


def hash_to_paths(index):
    """Map SHA-256 → [chemins relatifs] (pour savoir *où* est le doublon)."""
    m = {}
    for rel, rec in index.get('files', {}).items():
        m.setdefault(rec['sha256'], []).append(rel)
    return m


def main():
    ap = argparse.ArgumentParser(
        description="Index SHA-256 des images d'une bibliothèque (dédup incrémentale).")
    ap.add_argument('root', help='Racine à indexer (parcours récursif)')
    ap.add_argument('--exclude', action='append', default=[],
                    help='Sous-chemin relatif à exclure (répétable) — ex. le dossier à ranger')
    ap.add_argument('-o', '--output',
                    help='Chemin de l\'index JSON (défaut: <root>/.image_index.json)')
    ap.add_argument('--workers', type=int, default=8, help='Threads de hashage (I/O)')
    ap.add_argument('--rebuild', action='store_true',
                    help='Ignore l\'index existant (full rebuild, pas d\'incrémental)')
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"Dossier introuvable : {args.root}")

    out = args.output or os.path.join(args.root, '.image_index.json')
    previous = None
    if not args.rebuild and os.path.isfile(out):
        try:
            previous = load_index(out)
        except Exception:
            previous = None

    index = build_index(args.root, excluded=args.exclude,
                        workers=args.workers, previous=previous)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print(f"{index['count']} image(s) indexée(s), {index['hashed_new']} (re)hashée(s) "
          f"→ {out}")


if __name__ == '__main__':
    main()
