#!/usr/bin/env python3
"""Interface web de revue des near-duplicates (perceptuels).

Lit le JSON de candidats produit par `image_dedup.py --method perceptual`
(groupes `decision: 'pending'`), affiche une **grille dense** de groupes — les
images d'un groupe côte à côte en vignettes — et permet de **cliquer celles à
GARDER** (la plus grosse est pré-sélectionnée). Après un lot (~50), un POST
**déplace les images écartées vers DOUBLONS_DIR** (jamais supprimées) et marque
les groupes revus dans le JSON. Conçu pour un écran large (4K) : beaucoup de
groupes visibles d'un coup.

Tourne dans le conteneur utils (Flask + Pillow), NAS bind-monté sur /nas.
    docker run --rm -p 8081:8081 -v /root/code/utils:/opt/utils:ro \
        -v /mnt/nas-homes:/nas utils:latest \
        python /opt/utils/actions/picture_utils/review_server.py

Accès : http://192.168.1.15:8081
"""
import argparse
import io
import json
import os
import shutil
import threading
import urllib.parse

from flask import Flask, request, send_file, redirect, abort
from PIL import Image, ImageOps

app = Flask(__name__)
_LOCK = threading.Lock()
_THUMB_CACHE = {}          # relpath -> JPEG bytes
CFG = {}                   # rempli dans main()


def _load():
    with open(CFG['candidates'], encoding='utf-8') as f:
        return json.load(f)


def _save(data):
    tmp = CFG['candidates'] + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, CFG['candidates'])


def _abspath(rel):
    return os.path.join(CFG['root'], rel)


def _pending(data):
    return [g for g in data.get('groups', []) if g.get('decision') == 'pending']


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Revue doublons</title>
<style>
  :root{{--th:150px}}
  body{{margin:0;background:#1e1e1e;color:#ddd;font:13px system-ui}}
  header{{position:sticky;top:0;background:#111;padding:10px 16px;display:flex;
    flex-wrap:wrap;gap:12px;align-items:center;z-index:10;box-shadow:0 2px 8px #0008}}
  header b{{font-size:15px}} .muted{{color:#888}}
  button{{background:#2d6;border:0;color:#003;font-weight:700;padding:9px 18px;
    border-radius:6px;cursor:pointer;font-size:14px}}
  .sizectl{{display:flex;gap:4px;align-items:center;color:#888}}
  .sizectl button{{padding:4px 10px;font-size:12px;background:#333;color:#ccc}}
  .sizectl button.on{{background:#2d6;color:#003}}
  .pager{{display:flex;gap:8px;align-items:center;color:#888}}
  .pg{{background:#333;color:#ccc;text-decoration:none;padding:6px 12px;
    border-radius:6px;font-weight:700}}
  .pg:hover{{background:#444}}
  .pg.disabled{{opacity:.3;pointer-events:none}}
  #grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));
    gap:10px;padding:12px}}
  .group{{background:#262626;border:1px solid #333;border-radius:6px;padding:6px;
    display:flex;flex-wrap:wrap;gap:4px;align-content:flex-start}}
  .gtools{{flex-basis:100%;display:flex;gap:6px;margin-bottom:2px}}
  .gb{{background:#333;color:#ccc;border:1px solid #555;border-radius:4px;
    padding:3px 9px;font-size:12px;font-weight:600;cursor:pointer}}
  .gb-keep:hover{{background:#2d6;color:#003}}
  .gb-ign:hover{{background:#c33;color:#fff;border-color:#c33}}
  .group.ignored{{opacity:.45;outline:2px dashed #c33;outline-offset:-2px}}
  .group.ignored .gb-ign{{background:#c33;color:#fff;border-color:#c33}}
  .im{{position:relative;cursor:pointer;border:3px solid transparent;border-radius:4px}}
  .im img{{display:block;height:var(--th);width:auto;border-radius:2px}}
  .im.keep{{border-color:#2d6}}
  .im .x{{position:absolute;inset:0;background:#c33a;display:none;
    align-items:center;justify-content:center;font-size:26px;border-radius:2px}}
  .im:not(.keep) .x{{display:flex}}
  .group.ignored .im .x{{display:none}}
  .im .sz{{position:absolute;bottom:0;left:0;background:#000a;padding:1px 4px;font-size:10px}}
</style></head><body>
<header>
  <b>Revue near-duplicates</b>
  <span class="muted">{total} groupes en attente</span>
  <span class="pager">
    <a class="pg {prevdis}" href="/?page={prev}">‹ Préc.</a>
    <span>page {page1}/{pages} <span class="muted">({n} ici)</span></span>
    <a class="pg {nextdis}" href="/?page={next}">Suiv. ›</a>
  </span>
  <span class="muted">· clic = <span style="color:#2d6">garder</span> · par groupe : <span style="color:#2d6">✓ tout</span> / <span style="color:#c33">⊘ ignorer</span> · autres → Doublons/</span>
  <span class="sizectl">Vignettes
    <button type="button" data-th="110" onclick="setTh(110,this)">S</button>
    <button type="button" data-th="150" onclick="setTh(150,this)">M</button>
    <button type="button" data-th="210" onclick="setTh(210,this)">L</button>
  </span>
  <span style="flex:1"></span>
  <button onclick="submitBatch()">Traiter ce lot ({n}) →</button>
</header>
<div id="grid">{cards}</div>
<script>
const PAGE={page};
function toggle(el){{ if(el.closest('.group').classList.contains('ignored')) return;
  el.classList.toggle('keep'); }}
function keepAll(btn){{ const g=btn.closest('.group'); g.classList.remove('ignored');
  g.querySelectorAll('.im').forEach(i=>i.classList.add('keep')); }}
function toggleIgnore(btn){{ btn.closest('.group').classList.toggle('ignored'); }}
function setTh(px,btn){{ document.documentElement.style.setProperty('--th',px+'px');
  localStorage.setItem('th',px);
  document.querySelectorAll('.sizectl button').forEach(b=>b.classList.toggle('on',b===btn)); }}
(function(){{ const px=localStorage.getItem('th')||'150';
  document.documentElement.style.setProperty('--th',px+'px');
  document.querySelectorAll('.sizectl button').forEach(b=>b.classList.toggle('on',b.dataset.th===px)); }})();
function submitBatch(){{
  const groups=[...document.querySelectorAll('.group')].map(g=>(
    g.classList.contains('ignored')
      ? {{id:g.dataset.id, ignored:true}}
      : {{id:g.dataset.id, keep:[...g.querySelectorAll('.im.keep')].map(i=>i.dataset.path)}}
  ));
  const bad=groups.filter(g=>!g.ignored && g.keep.length===0);
  if(bad.length && !confirm(bad.length+" groupe(s) sans aucune image gardée → "
     +"TOUTES leurs images iront dans Doublons/. Continuer ?")) return;
  fetch('/process',{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{groups}})}}).then(r=>r.json()).then(r=>{{
      location.href='/?page='+PAGE;
    }});
}}
</script></body></html>"""


def _card(group):
    ims = []
    sizes = {}
    for rel in group['paths']:
        try:
            sizes[rel] = os.path.getsize(_abspath(rel))
        except OSError:
            sizes[rel] = 0
    keep_default = max(group['paths'], key=lambda r: sizes.get(r, 0))  # la plus grosse
    for rel in group['paths']:
        q = urllib.parse.quote(rel)
        kb = sizes.get(rel, 0) // 1024
        cls = 'im keep' if rel == keep_default else 'im'
        ims.append(
            f'<div class="{cls}" data-path="{rel}" onclick="toggle(this)">'
            f'<img loading="lazy" src="/thumb?p={q}"><div class="x">✕</div>'
            f'<div class="sz">{kb} Ko</div></div>')
    tools = ('<div class="gtools">'
             '<button type="button" class="gb gb-keep" onclick="keepAll(this)">✓ tout garder</button>'
             '<button type="button" class="gb gb-ign" onclick="toggleIgnore(this)">⊘ ignorer</button>'
             '</div>')
    return f'<div class="group" data-id="{group["id"]}">{tools}{"".join(ims)}</div>'


@app.route('/')
def index():
    data = _load()
    pend = _pending(data)
    total = len(pend)
    if not total:
        return ("<body style='background:#1e1e1e;color:#2d6;font:20px system-ui;"
                "padding:40px'>✅ Plus aucun groupe en attente. Revue terminée.</body>")
    bs = CFG['batch']
    pages = (total + bs - 1) // bs
    try:
        page = int(request.args.get('page', 0))
    except (TypeError, ValueError):
        page = 0
    page = max(0, min(page, pages - 1))
    batch = pend[page * bs:(page + 1) * bs]
    cards = ''.join(_card(g) for g in batch)
    return PAGE.format(
        n=len(batch), total=total, cards=cards,
        page=page, page1=page + 1, pages=pages,
        prev=page - 1, next=page + 1,
        prevdis='disabled' if page == 0 else '',
        nextdis='disabled' if page >= pages - 1 else '')


@app.route('/thumb')
def thumb():
    rel = request.args.get('p', '')
    with _LOCK:
        if rel in _THUMB_CACHE:
            return send_file(io.BytesIO(_THUMB_CACHE[rel]), mimetype='image/jpeg')
    path = _abspath(rel)
    if not os.path.isfile(path):
        abort(404)
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            im.thumbnail((280, 280))
            buf = io.BytesIO()
            im.save(buf, 'JPEG', quality=80)
            data = buf.getvalue()
    except Exception:
        abort(415)
    with _LOCK:
        if len(_THUMB_CACHE) < 4000:
            _THUMB_CACHE[rel] = data
    return send_file(io.BytesIO(data), mimetype='image/jpeg')


@app.route('/process', methods=['POST'])
def process():
    payload = request.get_json(force=True)
    moved, kept_all, skipped, errors = 0, 0, 0, 0
    os.makedirs(CFG['doublons'], exist_ok=True)
    with _LOCK:
        data = _load()
        by_id = {g['id']: g for g in data.get('groups', [])}
        for pg in payload.get('groups', []):
            g = by_id.get(pg.get('id'))
            if not g or g.get('decision') != 'pending':
                continue
            if pg.get('ignored'):
                g['decision'] = 'skipped'   # parqué hors file, rien déplacé
                skipped += 1
                continue
            keep = set(pg.get('keep', []))
            discard = [r for r in g['paths'] if r not in keep]
            if not discard:
                g['decision'] = 'kept_all'   # tout gardé = pas un doublon
                kept_all += 1
                continue
            done = []
            for rel in discard:
                src = _abspath(rel)
                name = os.path.basename(rel)
                base, ext = os.path.splitext(name)
                dst = os.path.join(CFG['doublons'], name)
                i = 2
                while os.path.exists(dst):
                    dst = os.path.join(CFG['doublons'], f"{base}-{i}{ext}")
                    i += 1
                try:
                    shutil.move(src, dst)
                    done.append(rel)
                    moved += 1
                except Exception as e:
                    print(f"[review] échec move {src}: {e}")
                    errors += 1
            g['decision'] = 'reviewed'
            g['kept'] = sorted(keep)
            g['moved_to_doublons'] = done
        _save(data)
    print(f"[review] lot traité : {moved} déplacés, {kept_all} gardés-entiers, "
          f"{skipped} ignorés, {errors} erreurs")
    return {'moved': moved, 'kept_all': kept_all, 'skipped': skipped, 'errors': errors}


def main():
    ap = argparse.ArgumentParser(description="UI web de revue des near-duplicates.")
    ap.add_argument('--candidates', default='/nas/chade/Drive/.perceptual_candidates.json',
                    help='JSON des groupes (image_dedup --method perceptual)')
    ap.add_argument('--doublons', default='/nas/chade/Drive/A Classer/Doublons',
                    help='Dossier où déplacer les images écartées')
    ap.add_argument('--batch', type=int, default=50, help='Groupes par lot')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8081)
    args = ap.parse_args()

    if not os.path.isfile(args.candidates):
        raise SystemExit(f"Candidats introuvables : {args.candidates}")
    data = json.load(open(args.candidates, encoding='utf-8'))
    CFG.update(candidates=args.candidates, doublons=args.doublons,
               batch=args.batch, root=data.get('root', ''))
    print(f"Revue : {len(_pending(data))} groupes en attente | root={CFG['root']} "
          f"| http://0.0.0.0:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
