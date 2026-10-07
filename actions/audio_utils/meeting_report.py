#!/usr/bin/env python3
"""
meeting_report.py — compte rendu de réunion de bout en bout, inférence DISTANTE
(passerelle LiteLLM perso → OVH AI Endpoints ; aucun GPU requis) :

  audio (+ channels.json)  →  transcription par canal / diarisation  →
  compte rendu + transcript rédigé (LLM)  →  dossier « Traité »

Configuration unique (« one fits all ») :
  • langue auto (détectée PAR TRANCHE de ~10 min de parole → réunions multilingues) ;
  • réhaussement (highpass + débruitage + dynaudnorm + loudnorm) partout ;
  • anti-écho AUTOMATIQUE (seulement si le micro recapte les haut-parleurs) ;
  • blancs retirés avant envoi (coût ≈ durée de parole), timestamps restaurés ;
  • FLAC multicanal + channels.json : « Moi » = micro (attribution exacte), sorties
    diarisées (« Système · S1/S2… ») ; autre audio (téléphone…) : diarisation globale.

Sorties dans <done-dir> (défaut : <dossier de l'audio>/Traité), base
« AAAA-MM-JJ - Sujet » :
  <base>_original.<ext> (+ <base>_original.channels.json)  ← l'audio source, déplacé
  <base>_enhanced.m4a      audio réhaussé à réécouter (stéréo G=Moi / D=autres)
  <base>.srt / <base>_brut.md   transcription brute (vérification)
  <base>.md                front-matter OKF + compte rendu + transcript rédigé

Dernière ligne de stdout = JSON résultat ({"status": "ok", "final_md": …} ou
{"status": "error", "message": …}) ; la progression va sur stderr.

Env : INFERENCE_BASE_URL / INFERENCE_API_KEY (LiteLLM), OVH_AI_API_KEY (diarisation),
      MEETING_SELF_NAME (nom de « Moi »), MEETING_SUMMARY_MODEL / MEETING_REWRITE_MODEL /
      ASR_MODEL (alias, défauts meeting-summary / meeting-rewrite / asr).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import inference_client as ic                                     # noqa: E402
import meeting_audio as ma                                        # noqa: E402
import meeting_llm as ml                                          # noqa: E402
from whisper_common import (_env_or_dotenv, channel_labels, is_hallucination,  # noqa: E402
                            is_echo_duplicate, _norm_phrase, write_srt, write_md)

ASR_CHUNK_SPEECH_S = 600          # tranche de parole par requête (langue détectée par tranche)
DIARIZE_MAX_SPEECH_S = 10_000     # limite OVH 3 h/requête (marge) ; au-delà, découpe
_T0 = time.time()


def log(msg: str):
    print(f'  [{time.time() - _T0:6.0f}s] {msg}', file=sys.stderr, flush=True)


# ── Transcription d'un flux ────────────────────────────────────────────────────

def _segments_from(resp: dict, tmap: ma.TimeMap, diarized: bool) -> list:
    """Réponse verbose_json → segments en temps ORIGINAL, filtrés anti-hallucination."""
    out = []
    if diarized and resp.get('diarization'):
        for d in resp['diarization']:
            txt = (d.get('text') or '').strip()
            if not txt or is_hallucination(txt):
                continue
            out.append({'start': tmap.to_orig(float(d['start'])),
                         'end': tmap.to_orig(float(d['end'])),
                         'speaker': str(d.get('speaker')), 'text': txt})
        return out
    for s in resp.get('segments') or []:
        txt = (s.get('text') or '').strip()
        if not txt or is_hallucination(txt, s.get('avg_logprob', 0.0),
                                       s.get('no_speech_prob', 0.0)):
            continue
        out.append({'start': tmap.to_orig(float(s['start'])),
                    'end': tmap.to_orig(float(s['end'])), 'speaker': None, 'text': txt})
    return out


def transcribe_stream(pcm, regions: list, diarize: bool, language: str | None,
                      tmp: str, tag: str, stats: dict) -> list:
    max_s = DIARIZE_MAX_SPEECH_S if diarize else ASR_CHUNK_SPEECH_S
    chunks = ma.build_chunks(pcm, regions, max_s)
    segs = []
    for i, (cpcm, tmap) in enumerate(chunks, 1):
        if cpcm.size < ma.SR // 2:
            continue
        path = os.path.join(tmp, f'{tag}_{i:03d}.flac')
        ma.write_flac(path, cpcm)
        resp, cost = ic.transcribe(path, language=language, diarize=diarize)
        os.remove(path)
        stats['asr_cost'] += cost
        stats['speech_s'] += cpcm.size / ma.SR
        if resp.get('language'):
            stats['languages'].add(resp['language'])
        got = _segments_from(resp, tmap, diarize)
        for s in got:
            s['part'] = i
        segs += got
        log(f'{tag} tranche {i}/{len(chunks)} : {cpcm.size / ma.SR:.0f}s de parole, '
            f'{len(got)} segments, langue {resp.get("language")}')
    return segs


# ── Pipeline ───────────────────────────────────────────────────────────────────

def _sources(path: str, info: dict) -> tuple[list, str]:
    """Sources à traiter → ([{index, kind, label, channels}], kind_global)."""
    side = os.path.splitext(path)[0] + '.channels.json'
    if os.path.exists(side):
        try:
            with open(side, encoding='utf-8') as f:
                cmap = json.load(f)
            srcs = cmap['sources']
            if int(cmap.get('total_channels') or 0) == info['channels'] and srcs:
                labels = channel_labels(srcs)
                return ([{'index': s['index'], 'kind': s['kind'], 'label': labels[s['index']],
                          'channels': (int(s['channel_start']), int(s['channel_end']))}
                         for s in srcs], 'multichannel')
            log('channels.json incohérent avec le fichier → traité comme audio simple')
        except (OSError, ValueError, KeyError) as e:
            log(f'channels.json illisible ({e}) → traité comme audio simple')
    return [{'index': 0, 'kind': 'mixed', 'label': '', 'channels': None}], 'single'


def _unique_base(done_dir: str, base: str, ext: str) -> str:
    cand, n = base, 2
    while (os.path.exists(os.path.join(done_dir, cand + '.md'))
           or os.path.exists(os.path.join(done_dir, f'{cand}_original{ext}'))):
        cand, n = f'{base} ({n})', n + 1
    return cand


def _yaml(v) -> str:
    """Scalaire/liste/dict → YAML (les chaînes JSON sont des scalaires YAML valides)."""
    if isinstance(v, dict):
        return '{' + ', '.join(f'{json.dumps(str(k), ensure_ascii=False)}: {_yaml(x)}'
                               for k, x in v.items()) + '}'
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(_yaml(x) for x in v) + ']'
    if v is None:
        return 'null'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return str(round(v, 4) if isinstance(v, float) else v)
    return json.dumps(str(v), ensure_ascii=False)


def _fmt_duration(s: float) -> str:
    s = int(round(s))
    h, m = divmod(s // 60, 60)
    return f'{h} h {m:02d}' if h else f'{m} min'


def run(path: str, done_dir: str | None = None, self_name: str | None = None,
        language: str | None = None, no_llm: bool = False, keep_tmp: bool = False) -> dict:
    path = os.path.abspath(path)
    src_dir, fname = os.path.split(path)
    stem, ext = os.path.splitext(fname)
    done_dir = done_dir or os.path.join(src_dir, 'Traité')
    self_name = self_name or _env_or_dotenv('MEETING_SELF_NAME') or None

    info = ma.probe(path)
    dt, title_hint = ma.parse_name(stem, os.path.getmtime(path), info.get('creation_time'))
    sources, kind = _sources(path, info)
    log(f'{fname} : {info["channels"]} canaux, {kind}, date {dt:%Y-%m-%d %H:%M}, '
        f'titre {title_hint!r}')

    stats = {'asr_cost': 0.0, 'speech_s': 0.0, 'languages': set()}
    tmp = tempfile.mkdtemp(prefix='meeting_')
    try:
        # 1. Extraction brute par source (16 kHz mono)
        raws = {}
        for s in sources:
            raws[s['index']] = os.path.join(tmp, f'raw{s["index"]}.wav')
            ma.extract_raw(path, raws[s['index']], s['channels'])
        duration = max(ma.wav_seconds(p) for p in raws.values())
        log(f'extraction : {_fmt_duration(duration)}')

        # 2. Anti-écho automatique (micro vs mix des sorties), sur le BRUT
        inputs = [s for s in sources if s['kind'] == 'input']
        outputs = [s for s in sources if s['kind'] == 'output']
        echo_score, aec_applied = None, False
        if inputs and outputs:
            ref = ma.read_wav(raws[outputs[0]['index']]).astype('int32')
            for o in outputs[1:]:
                r = ma.read_wav(raws[o['index']])
                n = min(ref.size, r.size)
                ref = ref[:n] + r[:n]
            ref = (ref // len(outputs)).astype('int16')
            for s in inputs:
                mic = ma.read_wav(raws[s['index']])
                score = ma.detect_echo(mic, ref)
                echo_score = max(echo_score or 0.0, score)
                if score > ma.ECHO_COHERENCE_THRESHOLD:
                    ma.write_wav(raws[s['index']], ma.cancel_echo(mic, ref))
                    aec_applied = True
                del mic
            del ref
            log(f'écho micro↔sortie : {echo_score:.2f} → AEC {"appliquée" if aec_applied else "non"}')

        # 3. Réhaussement + 4. retrait des blancs + 5. transcription
        enh, segs = {}, []
        for s in sources:
            enh[s['index']] = os.path.join(tmp, f'enh{s["index"]}.wav')
            ma.enhance(raws[s['index']], enh[s['index']])
            regions = ma.speech_regions(ma.read_wav(raws[s['index']]))
            speech = sum(e - b for b, e in regions)
            log(f'source {s["label"] or "audio"} : {speech / 60:.1f} min de parole '
                f'sur {duration / 60:.1f}')
            if not regions:
                continue
            diarize = s['kind'] != 'input'          # micro = « Moi » ; le reste est diarisé
            got = transcribe_stream(ma.read_wav(enh[s['index']]), regions, diarize,
                                    language, tmp, f'src{s["index"]}', stats)
            for g in got:
                g['source'] = s
            segs += got

        # Libellés : Moi / Système · Sn / Speaker n (numérotés par ordre d'apparition)
        numbering = {}
        for g in sorted(segs, key=lambda r: r['start']):
            s = g['source']
            if s['kind'] == 'input':
                g['label'] = s['label']
                continue
            # les numéros de diarisation ne sont cohérents QU'À L'INTÉRIEUR d'une requête
            # (une requête = tout le flux tant qu'il fait < ~2 h 45 de parole)
            key = (s['index'], g.get('part'), g['speaker'])
            if key not in numbering:
                numbering[key] = sum(1 for k in numbering if k[0] == s['index']) + 1
            n = numbering[key]
            g['label'] = (f'{s["label"]} · S{n}' if s['label'] else f'Speaker {n}')
        # Un seul intervenant distant → pas de suffixe
        for s in sources:
            ks = [k for k in numbering if k[0] == s['index']]
            if len(ks) == 1 and s['label']:
                for g in segs:
                    if g['source'] is s:
                        g['label'] = s['label']

        # Dé-doublonnage d'écho / répétitions (mêmes critères que le live)
        segs.sort(key=lambda r: r['start'])
        recent, kept = [], []
        for r in segs:
            if is_echo_duplicate(r['text'], r['label'], r['start'], recent):
                continue
            recent.append((r['start'], r['label'], _norm_phrase(r['text'])))
            recent = [x for x in recent if abs(r['start'] - x[0]) <= 6.0]
            kept.append({'start': r['start'], 'end': r['end'], 'label': r['label'],
                         'text': r['text']})
        segs = kept
        if not segs:
            raise RuntimeError('aucune parole transcrite')
        labels = sorted({r['label'] for r in segs}, key=lambda l: (l != 'Moi', l))
        log(f'{len(segs)} segments, locuteurs : {", ".join(labels)}, '
            f'langues : {", ".join(sorted(stats["languages"]))}')

        # 6. LLM : analyse, compte rendu, transcript rédigé
        usage = ml.Usage()
        lines = ml.transcript_lines(segs)
        if no_llm:
            analysis = {'meeting_type': None, 'subject': title_hint or 'Réunion',
                        'description': None, 'speakers': {l: None for l in labels},
                        'tags': []}
            report_md, rewritten = None, None
        else:
            # Étiquettes diarisées très marginales (< 2 % du texte) : souvent des fragments
            # d'un locuteur principal → signalées au LLM (rattachement si évident).
            chars = {l: sum(len(r['text']) for r in segs if r['label'] == l) for l in labels}
            total = sum(chars.values()) or 1
            minor = [l for l in labels if l != 'Moi' and chars[l] / total < 0.02]
            analysis = ml.analyze(lines, usage, title_hint=title_hint, file_name=fname,
                                  self_name=self_name, labels=labels, minor_labels=minor,
                                  dt=dt, self_in_labels='Moi' in labels)
            analysis['minor_labels'] = [l for l in minor if not analysis['speakers'].get(l)]
            log(f'analyse : {analysis["meeting_type"]} — {analysis.get("subject")!r}')
            report_md = ml.report(lines, analysis, usage, dt)
            log('compte rendu rédigé')
            rewritten = ml.rewrite(lines, analysis, usage)
            log(f'transcript rédigé ({len(rewritten)} caractères)')

        subject = title_hint or analysis.get('subject') or 'Réunion'
        subject = subject[:1].upper() + subject[1:]
        os.makedirs(done_dir, exist_ok=True)
        base = _unique_base(done_dir, ma.safe_filename(f'{dt:%Y-%m-%d} - {subject}'), ext)

        # 7. Fichiers de sortie (dans tmp, puis copiés)
        names = {l: (analysis['speakers'].get(l) or l) for l in labels}
        participants = list(dict.fromkeys(names[l] for l in labels
                                          if l not in (analysis.get('minor_labels') or [])))
        write_srt(os.path.join(tmp, base + '.srt'), segs)
        write_md(os.path.join(tmp, base + '_brut.md'), segs, title=f'{subject} — transcription brute')

        left = [enh[s['index']] for s in sources if s['kind'] == 'input']
        right = [enh[s['index']] for s in sources if s['kind'] != 'input']
        m4a = os.path.join(tmp, base + '_enhanced.m4a')
        if left and right:
            ma.mix_wavs(left, os.path.join(tmp, 'L.wav'))
            ma.mix_wavs(right, os.path.join(tmp, 'R.wav'))
            ma.write_listening_m4a(m4a, os.path.join(tmp, 'L.wav'), os.path.join(tmp, 'R.wav'))
        else:
            ma.mix_wavs(left or right, os.path.join(tmp, 'M.wav'))
            ma.write_listening_m4a(m4a, os.path.join(tmp, 'M.wav'))

        cost_total = stats['asr_cost'] + usage.cost
        fm = {
            'type': 'meeting-transcript',
            'title': subject,
            'description': analysis.get('description'),
            'timestamp': dt.astimezone().isoformat(timespec='seconds'),
            'tags': ['transcript'] + list(analysis.get('tags') or []),
            'status': 'draft',
            'meeting_type': analysis.get('meeting_type'),
            'languages': sorted(stats['languages']),
            'duration_sec': int(round(duration)),
            'speech_sec': int(round(stats['speech_s'])),
            'channels': info['channels'],
            'sources': [{'label': s['label'] or 'audio', 'kind': s['kind']} for s in sources],
            'speaker_count': len(participants),
            'speakers': names,
            'source_kind': 'multichannel' if kind == 'multichannel' else 'single',
            'processing': {
                'asr': 'whisper-large-v3 (OVH AI Endpoints)',
                'asr_route': f'{_env_or_dotenv("ASR_MODEL") or "asr"} via LiteLLM ; diarisation OVH direct',
                'llm_summary': None if no_llm else ml._model('summary'),
                'llm_rewrite': None if no_llm else ml._model('rewrite'),
                'enhance': True, 'silence_removed': True,
                'aec': aec_applied, 'echo_coherence': (round(echo_score, 3)
                                                       if echo_score is not None else None)},
            'cost_eur': {'asr': round(stats['asr_cost'], 4), 'llm': round(usage.cost, 4),
                         'total': round(cost_total, 4)},
            'files': {'original': f'./{base}_original{ext}', 'enhanced': f'./{base}_enhanced.m4a',
                      'raw_srt': f'./{base}.srt', 'raw_md': f'./{base}_brut.md'},
            'generated': {'actor': 'utils/meeting_report.py',
                          'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds')},
        }
        head = '---\n' + ''.join(f'{k}: {_yaml(v)}\n' for k, v in fm.items()) + '---\n\n'
        body = [f'# {subject}\n']
        if analysis.get('description'):
            body.append(f'> {analysis["description"]}\n')
        meta = [f'**Date** : {dt:%d/%m/%Y %H:%M}', f'**Durée** : {_fmt_duration(duration)}',
                f'**Participants** : {", ".join(participants)}']
        if analysis.get('meeting_type'):
            meta.append(f'**Type** : {analysis["meeting_type"]}')
        body.append(' · '.join(meta) + '\n')
        if report_md:
            body.append('## Compte rendu\n\n' + report_md.strip() + '\n')
        if rewritten:
            body.append('## Transcript rédigé\n\n' + rewritten.strip() + '\n')
        else:
            body.append('## Transcription\n\n' + '\n\n'.join(lines) + '\n')
        final_md = os.path.join(tmp, base + '.md')
        with open(final_md, 'w', encoding='utf-8') as f:
            f.write(head + '\n'.join(body))

        # 8. Livraison : sorties d'abord, l'original EN DERNIER (s'il échoue avant, rien
        #    n'est déplacé → le prochain passage retraite proprement).
        for suffix in ('.srt', '_brut.md', '_enhanced.m4a', '.md'):
            shutil.copyfile(os.path.join(tmp, base + suffix),
                            os.path.join(done_dir, base + suffix))
        side = os.path.splitext(path)[0] + '.channels.json'
        if os.path.exists(side):
            shutil.move(side, os.path.join(done_dir, f'{base}_original.channels.json'))
        shutil.move(path, os.path.join(done_dir, f'{base}_original{ext}'))

        log(f'terminé → {base}.md  (coût {cost_total:.3f} € : ASR {stats["asr_cost"]:.3f}'
            f' + LLM {usage.cost:.3f})')
        return {'status': 'ok', 'final_md': os.path.join(done_dir, base + '.md'),
                'base': base, 'done_dir': done_dir, 'subject': subject,
                'meeting_type': analysis.get('meeting_type'),
                'description': analysis.get('description'),
                'duration_sec': int(round(duration)), 'cost_eur': round(cost_total, 4),
                'languages': sorted(stats['languages']), 'participants': participants}
    finally:
        if keep_tmp:
            log(f'fichiers de travail conservés : {tmp}')
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    p = argparse.ArgumentParser(description='Compte rendu de réunion (OVH via LiteLLM)')
    p.add_argument('input_file', nargs='+', help='audio (FLAC multicanal + channels.json, m4a…)')
    p.add_argument('--done-dir', help='dossier de sortie (défaut : <dossier>/Traité)')
    p.add_argument('--self-name', help='nom de « Moi » (défaut : MEETING_SELF_NAME)')
    p.add_argument('--language', help='forcer la langue (défaut : détection par tranche)')
    p.add_argument('--no-llm', action='store_true', help='transcription seule, sans IA')
    p.add_argument('--keep-tmp', action='store_true', help='garder les fichiers de travail')
    a = p.parse_args()
    try:
        res = run(' '.join(a.input_file), a.done_dir, a.self_name, a.language,
                  a.no_llm, a.keep_tmp)
    except Exception as e:                                   # noqa: BLE001 — JSON pour l'appelant
        res = {'status': 'error', 'message': f'{type(e).__name__}: {e}'}
    print(json.dumps(res, ensure_ascii=False))
    return 0 if res['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
