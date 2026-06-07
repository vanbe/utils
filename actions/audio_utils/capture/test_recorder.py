#!/usr/bin/env python3
"""
test_recorder.py — test bout-en-bout de l'enregistreur, À LANCER SUR LE LAPTOP.

Exerce la VRAIE chaîne (recorder.Recorder → capture.exe|ffmpeg → FLAC) sans la
TUI : énumère les sources, en sélectionne, enregistre quelques secondes en
affichant les niveaux, puis vérifie le FLAC (ffprobe) et le sidecar channels.json.

Exemples :
  # WSL : build auto si capture.exe manque, auto-sélection (1er micro + 1ère sortie)
  python3 test_recorder.py

  # durée + dossier de sortie
  python3 test_recorder.py --seconds 8 --out /mnt/c/Users/moi/Desktop

  # sources explicites (ids retournés par --probe)
  python3 test_recorder.py --probe
  python3 test_recorder.py --source "<id1>" --source "<id2>"

Prérequis : ffmpeg + ffprobe dans le PATH ; sous WSL, mingw si capture.exe doit
être construit (sinon l'option --build / la TUI s'en charge).
"""

import argparse
import json
import os
import subprocess
import sys
import time

# recorder.py est dans le dossier parent (actions/audio_utils/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import recorder  # noqa: E402


def _meter(level: float, width: int = 24) -> str:
    disp = max(0.0, min(1.0, level ** 0.5))
    filled = int(disp * width)
    return '#' * filled + '-' * (width - filled)


def probe_print(sources):
    print(f'backend : {recorder.detect_backend()}')
    print(f'capture.exe présent : {recorder.capture_exe_present()}')
    if not sources:
        print('  (aucune source — serveur audio absent / capture.exe vide ?)')
        return
    for s in sources:
        print(f"  [{s['kind']:6}] {s['channels']}ch  {s['name']}")
        print(f"            id = {s['id']}")


def pick_sources(sources, requested):
    if requested:
        by_id = {s['id']: s for s in sources}
        chosen = [by_id[r] for r in requested if r in by_id]
        missing = [r for r in requested if r not in by_id]
        if missing:
            print(f'⚠ ids introuvables : {missing}', file=sys.stderr)
        return chosen
    # auto : 1ère entrée + 1ère sortie (sinon ce qu'on a)
    ins = [s for s in sources if s['kind'] == 'input']
    outs = [s for s in sources if s['kind'] == 'output']
    chosen = []
    if ins:
        chosen.append(ins[0])
    if outs:
        chosen.append(outs[0])
    return chosen or sources[:1]


def main():
    ap = argparse.ArgumentParser(description='Test bout-en-bout du recorder (laptop).')
    ap.add_argument('--seconds', type=int, default=8)
    ap.add_argument('--out', default=os.getcwd(), help='dossier de sortie')
    ap.add_argument('--source', action='append', default=[], help='id de source (répétable)')
    ap.add_argument('--probe', action='store_true', help='lister les sources et quitter')
    ap.add_argument('--build', action='store_true', help='(re)construire capture.exe puis quitter')
    ap.add_argument('--transcribe', action='store_true', help='transcription live par canal')
    ap.add_argument('--lang', default='fr', help='langue: fr | en | auto')
    ap.add_argument('--model', default=None, help='modèle whisper (défaut: .env)')
    args = ap.parse_args()

    backend = recorder.detect_backend()

    if args.build:
        ok, log = recorder.build_capture()
        print(log)
        print('OK' if ok else 'ÉCHEC')
        sys.exit(0 if ok else 1)

    # WSL : s'assurer du binaire
    if backend == 'wsl' and not recorder.capture_exe_present():
        print('capture.exe absent → construction…')
        ok, log = recorder.build_capture()
        print(log)
        if not ok:
            print('ÉCHEC build. Installer mingw : sudo apt-get install g++-mingw-w64-x86-64',
                  file=sys.stderr)
            sys.exit(1)

    sources = recorder.list_sources(backend)
    if args.probe:
        probe_print(sources)
        return
    if not sources:
        print('Aucune source audio détectée — abandon.', file=sys.stderr)
        sys.exit(1)

    chosen = pick_sources(sources, args.source)
    if not chosen:
        print('Aucune source sélectionnée.', file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    out_path = recorder.unique_path(args.out, recorder.default_basename(), 'flac')

    print(f'backend  : {backend}')
    print('sources  :')
    for s in chosen:
        print(f"   [{s['kind']:6}] {s['channels']}ch  {s['name']}")
    total = sum(int(s['channels']) for s in chosen)
    print(f'fichier  : {out_path}  ({total} canaux)')
    print(f'durée    : {args.seconds}s  (Ctrl-C pour arrêter avant)\n')

    rec = recorder.Recorder(chosen, out_path, backend=backend)
    transcriber = None
    if args.transcribe:
        import live_transcribe

        def _seg(s):
            mark = '…' if s.get('interim') else '»'   # … = aperçu en cours
            print(f"\n  {mark} {s['label']}: {s['text']}")

        transcriber = live_transcribe.LiveTranscriber(
            rec.channel_map(), _seg, os.path.splitext(out_path)[0] + '.srt',
            language=args.lang, model_name=args.model)
        rec.on_pcm = transcriber.feed_bytes
        print(f'  chargement du modèle Whisper ({transcriber.model_name})…')
        transcriber.start()
        print('  modèle prêt — parlez / jouez un son.\n')

    rec.start()
    try:
        while rec.elapsed() < args.seconds and rec.is_alive():
            lv = rec.levels()
            cells = '   '.join(
                f"{s['kind'][:3]}:{_meter(lv[i] if i < len(lv) else 0.0)}"
                for i, s in enumerate(chosen))
            sys.stdout.write(f'\r  t={rec.elapsed():4.1f}s  {cells} ')
            sys.stdout.flush()
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    print()
    okstop = rec.stop()
    if transcriber:
        print('  finalisation de la transcription…')
        transcriber.finalize()
    print(f'\nstop : {"ok" if okstop else "échec"}')

    if not os.path.exists(out_path):
        print('✗ aucun fichier produit (capture/ffmpeg en échec ?)', file=sys.stderr)
        sys.exit(1)

    print(f'✓ {out_path}  ({os.path.getsize(out_path)} octets)')

    # ffprobe : vérifier codec / canaux / rate
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
             '-show_entries', 'stream=codec_name,channels,sample_rate',
             '-of', 'default=nw=1', out_path],
            capture_output=True, text=True, timeout=15)
        print('\nffprobe :')
        print('  ' + (r.stdout.strip().replace('\n', '\n  ') or r.stderr.strip()))
    except (OSError, subprocess.SubprocessError) as e:
        print(f'ffprobe indisponible : {e}', file=sys.stderr)

    # sidecar
    side = os.path.splitext(out_path)[0] + '.channels.json'
    if os.path.exists(side):
        with open(side, encoding='utf-8') as f:
            data = json.load(f)
        print('\nchannels.json :')
        print('  ' + json.dumps(data, ensure_ascii=False, indent=2).replace('\n', '\n  '))
        print('\nDémux d\'un canal (exemple) :')
        for c in data['sources']:
            print(f"  ffmpeg -i \"{os.path.basename(out_path)}\" -filter_complex "
                  f"\"pan=mono|c0=c{c['channel_start']}\" \"{c['kind']}_{c['index']}.wav\"")
    else:
        print('⚠ sidecar channels.json absent', file=sys.stderr)

    if args.transcribe:
        for ext in ('.srt', '.md'):
            p = os.path.splitext(out_path)[0] + ext
            print(f'\n{ext} : {p if os.path.exists(p) else "(absent)"}')
            if os.path.exists(p):
                print('  ' + open(p, encoding='utf-8').read().strip().replace('\n', '\n  '))


if __name__ == '__main__':
    main()
