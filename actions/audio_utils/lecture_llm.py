#!/usr/bin/env python3
"""
lecture_llm.py — post-traitement LLM d'un COURS / d'une CONFÉRENCE (mode `--kind lecture`
de meeting_report.py ; DAG Airflow `vanbe_emba_courses`). Réutilise les briques de
meeting_llm (appels, parsing, réécriture) ; les comptes rendus de réunion n'en dépendent pas.

Différences avec une réunion :
  • notes de cours (en anglais par défaut, `MEETING_REPORT_LANGUAGE`) : vue d'ensemble,
    objectifs, notions + définitions, cadres/modèles, exemples/cas, références citées,
    questions-réponses, devoirs/échéances, glossaire (assemblé par le code) ;
  • découpage en SECTIONS de cours, chacune typée : course | qa | logistics | break —
    les passages `break` (micro resté ouvert pendant une pause, bavardages, bruit) sont
    EXCLUS du compte rendu et du transcript rédigé (signalés par leur seule plage horaire) ;
  • cours très longs (≥ 4 h) : découpage par FENÊTRES (≤ 120 k car. chacune), puis fusion ;
  • classement : le cours est choisi dans le CATALOGUE fourni (code EMBA-2527-xxx) — code
    présent dans le nom de fichier d'abord (déterministe), sinon choix du LLM, validé contre
    le catalogue (jamais de cours inventé).
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor

import inference_client as ic
import meeting_llm as ml

KINDS = ('course', 'qa', 'logistics', 'break')
MAX_SECTIONS = 16
MIN_SECTION_SEC = 300           # section plus courte = rattachée à la voisine
WINDOW_CHARS = 120_000          # fenêtre de découpage (cours très longs)

_CODE_RE = re.compile(r'EMBA-\d{4}-\d{3}[A-Z]?', re.I)


def _lang() -> str:
    return ml._env_or_dotenv('MEETING_REPORT_LANGUAGE') or 'English'


# ── 1. Analyse : cours, sujet, professeur ──────────────────────────────────────

def match_course(text: str, catalog: list) -> str | None:
    """Code EMBA-xxxx-xxx présent dans `text` → entrée du catalogue correspondante."""
    for code in _CODE_RE.findall(text or ''):
        hit = next((c for c in catalog if c.upper().startswith(code.upper())), None)
        if hit:
            return hit
    return None


def analyze(lines: list, usage: ml.Usage, *, title_hint: str | None, file_name: str,
            catalog: list, dt) -> dict:
    sample = '\n'.join(lines)
    if len(sample) > 60_000:                      # début + milieu + fin suffisent ici
        n = len(lines)
        sample = '\n'.join(lines[:n // 6] + ['[…]'] + lines[n // 2 - n // 12:n // 2 + n // 12]
                           + ['[…]'] + lines[-n // 8:])[:60_000]
    system = (
        "You analyse the automatic transcript of a university COURSE or lecture (Executive MBA). "
        "Answer ONLY with a JSON object with the keys:\n"
        '- "course": the EXACT entry of the catalogue below that this session belongs to, or '
        'null if none clearly matches\n'
        '- "subject": short title of THIS session (3 to 8 words)\n'
        '- "description": one sentence summarising the session\n'
        '- "professor": the lecturer\'s name ONLY if stated in the transcript (introduction, '
        'someone addresses them), else null\n'
        '- "tags": 2 to 5 lowercase topical keywords\n'
        f"Write subject and description in {_lang()}.\n"
        f"Catalogue: {json.dumps(catalog, ensure_ascii=False)}")
    user = (f"File: {file_name}\n" + (f"Title given by the user: {title_hint}\n" if title_hint else '')
            + f"Date: {ml._date_fr(dt)}\n\nTranscript (excerpts):\n{sample}")
    try:
        d = ml._parse_json(ml._chat(usage, 'summary', system, user, 1500, json_mode=True,
                                    temperature=0))
    except (ValueError, json.JSONDecodeError, ic.InferenceError):
        d = {}
    course = match_course(f'{file_name} {title_hint or ""}', catalog)
    if not course and d.get('course') in catalog:
        course = d['course']
    prof = str(d.get('professor') or '').strip() or None
    full = ml._fold(' '.join(lines))
    if prof and ml._fold(prof.split()[-1]) not in full:   # nom absent du texte → refusé
        prof = None
    return {'meeting_type': 'cours', 'course': course,
            'subject': str(d.get('subject') or title_hint or 'Course session').strip(),
            'description': str(d.get('description') or '').strip() or None,
            'professor': prof, 'tags': [str(t) for t in d.get('tags') or []][:5]}


# ── 2. Sections (cours / questions / logistique / pause) ───────────────────────

def _segment_window(lines: list, offset: int, analysis: dict, usage: ml.Usage) -> list:
    """→ [(start_sec, title, summary, kind)] pour une fenêtre de répliques."""
    system = (
        "You split the transcript of a lecture into SECTIONS in chronological order. A section "
        "is a coherent teaching unit (a concept, a model, a case discussion, an exercise). "
        "Give each section a kind: \"course\" (teaching, case discussion, exercise), \"qa\" "
        "(questions and answers with the class), \"logistics\" (agenda, assignments, deadlines, "
        "exam, practical info) or \"break\" — a BREAK is any passage that is NOT the class: the "
        "microphone left on during a pause, informal chit-chat, private conversations, lunch, "
        "silence or background noise. Be strict: personal conversations are always \"break\". "
        f"Merge digressions shorter than {MIN_SECTION_SEC // 60} minutes into the neighbour. "
        "Answer in JSON: {\"sections\":[{\"start\":\"hh:mm:ss\",\"title\":\"short title "
        "(3-8 words)\",\"summary\":\"one sentence\",\"kind\":\"course\"}]} — the first section "
        f"starts at the first line. Titles and summaries in {_lang()}.")
    user = (f"Course: {analysis.get('course') or 'unknown'} — session: {analysis.get('subject')}\n\n"
            "Transcript:\n" + '\n'.join(lines))
    try:
        d = ml._parse_json(ml._chat(usage, 'summary', system, user, 4000, json_mode=True,
                                    temperature=0))
    except (ValueError, json.JSONDecodeError, ic.InferenceError):
        return []
    out = []
    for s in d.get('sections') or []:
        t = ml._parse_hms(s.get('start'))
        title = str(s.get('title') or '').strip()
        if t is None or not title:
            continue
        kind = s.get('kind') if s.get('kind') in KINDS else 'course'
        out.append((t, title, str(s.get('summary') or '').strip(), kind))
    return sorted(out)


def segment(lines: list, analysis: dict, usage: ml.Usage) -> dict | None:
    """→ même forme que meeting_llm.segment_topics (+ `kind` par section) ou None."""
    if len(lines) < 10:
        return None
    windows = ml._parts(lines, WINDOW_CHARS)
    raw, idx = [], 0
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(lambda w: _segment_window(w, 0, analysis, usage), windows))
    for w, res in zip(windows, results):
        first = ml._line_ts(w[0])
        if not res:                                   # fenêtre inexploitable → 1 section
            res = [(first, f'Part {len(raw) + 1}', '', 'course')]
        res[0] = (min(res[0][0], first),) + res[0][1:]
        raw += res
        idx += len(w)
    if not raw:
        return None
    # Répliques → section (dernière section commencée avant la réplique)
    starts = [r[0] for r in raw]
    line_sec = []
    for l in lines:
        ts = ml._line_ts(l)
        line_sec.append(max(0, max((i for i, s in enumerate(starts) if s <= ts), default=0)))
    # Sections trop courtes → rattachées à la voisine (sauf les pauses : on les garde même
    # courtes pour ne JAMAIS laisser de bavardage dans le compte rendu)
    changed = True
    while changed:
        changed = False
        runs = _runs(line_sec)
        if len(runs) < 2:
            break
        for j, (i0, i1, k) in enumerate(runs):
            if raw[k][3] == 'break':
                continue
            end_ts = ml._line_ts(lines[runs[j + 1][0]]) if j + 1 < len(runs) else ml._line_ts(lines[-1])
            if end_ts - ml._line_ts(lines[i0]) < MIN_SECTION_SEC:
                nk = runs[j - 1][2] if j > 0 else runs[j + 1][2]
                if raw[nk][3] == 'break' and j + 1 < len(runs) and j > 0:
                    nk = runs[j + 1][2]
                line_sec[i0:i1] = [nk] * (i1 - i0)
                changed = True
                break
    # Sections consécutives de même titre (frontière de fenêtre) → fusion
    runs = _runs(line_sec)
    topics, line_topic = [], [0] * len(lines)
    for i0, i1, k in runs:
        title, summary, kind = raw[k][1], raw[k][2], raw[k][3]
        if topics and ml._fold(topics[-1]['title']) == ml._fold(title) and topics[-1]['kind'] == kind:
            topics[-1]['spans'].append((i0, i1))
        else:
            topics.append({'title': title, 'summary': summary, 'kind': kind, 'spans': [(i0, i1)]})
        line_topic[i0:i1] = [len(topics)] * (i1 - i0)
    if len(topics) > MAX_SECTIONS * max(1, len(windows)):
        return None
    # Numérotation CONTINUE des sections enseignées (les pauses ont un id ≥ 900, non affiché) ;
    # `next_start` = début de la section suivante → vraie plage / durée d'une pause (ses rares
    # répliques ne la couvrent pas : le silence a été retiré avant transcription).
    renum, k = {}, 0
    for i, t in enumerate(topics, 1):
        if t['kind'] == 'break':
            renum[i] = 900 + i
        else:
            k += 1
            renum[i] = k
    starts = [ml._line_ts(lines[t['spans'][0][0]]) for t in topics]
    for i, t in enumerate(topics):
        t.update(num=renum[i + 1], id=renum[i + 1], start=starts[i],
                 end=ml._line_ts(lines[t['spans'][-1][1] - 1]),
                 next_start=starts[i + 1] if i + 1 < len(topics) else ml._line_ts(lines[-1]))
    return {'topics': topics, 'line_topic': [renum[x] for x in line_topic]}


def break_label(t: dict) -> str:
    return f"{ml._hm(t['start'])}–{ml._hm(t['next_start'])}"


def _runs(seq: list) -> list:
    out, i0 = [], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[i0]:
            out.append([i0, i, seq[i0]])
            i0 = i
    return out


def whole_as_one(lines: list, analysis: dict) -> dict:
    """Repli si le découpage échoue : une seule section « cours »."""
    return {'topics': [{'num': 1, 'id': 1, 'title': analysis.get('subject') or 'Session',
                        'summary': analysis.get('description') or '', 'kind': 'course',
                        'spans': [(0, len(lines))], 'start': ml._line_ts(lines[0]),
                        'end': ml._line_ts(lines[-1]), 'next_start': ml._line_ts(lines[-1])}],
            'line_topic': [1] * len(lines)}


# ── 3. Notes de cours ──────────────────────────────────────────────────────────

def _section_detail(t: dict, lines: list, analysis: dict, usage: ml.Usage, dt) -> dict:
    body = '\n'.join(lines[i] for i0, i1 in t['spans'] for i in range(i0, i1))
    if t['kind'] == 'logistics':
        schema = ('{"infos":["practical information"],"assignments":[{"what":"assignment / '
                  'reading / deliverable","when":"deadline as stated"}]}')
    else:
        schema = ('{"key_points":["key idea (full sentence)"],"concepts":[{"term":"concept",'
                  '"definition":"definition as explained"}],"frameworks":["model / framework / '
                  'tool and what it is for"],"examples":["example, case or anecdote and its '
                  'lesson"],"references":["book, author, article, company or data cited"],'
                  '"qa":[{"question":"question asked","answer":"answer given"}],'
                  '"assignments":[{"what":"assignment / reading","when":"deadline as stated"}]}')
    system = (
        f"You write the study notes of ONE section of a lecture, in {_lang()}, from its transcript "
        f"only. Answer in JSON: {schema}. Rules: be faithful — never invent content, names, "
        "figures or deadlines; the transcript is automatic and may contain recognition errors: "
        "correct obvious ones (names of authors, models, companies) from context; keep the "
        "lecturer's own definitions; empty lists if nothing. 3 to 10 key points.")
    user = (f"Course: {analysis.get('course') or 'unknown'} — session: {analysis.get('subject')} — "
            f"{ml._date_fr(dt)}\nSection: {t['title']} — {t['summary']}\n\nTranscript:\n{body}")
    try:
        return ml._parse_json(ml._chat(usage, 'summary', system, user, 6000, json_mode=True,
                                       temperature=0))
    except (ValueError, json.JSONDecodeError, ic.InferenceError):
        return {}


def _overview(topics: list, details: list, analysis: dict, usage: ml.Usage) -> dict:
    fiches = [{'section': f"{t['num']}. {t['title']}", 'kind': t['kind'], **d}
              for t, d in zip(topics, details)]
    system = (
        f"From the section notes of a lecture, write in {_lang()} a JSON object: "
        "{\"overview\":\"one paragraph (4-7 sentences): what the session covered and how it "
        "builds up\",\"objectives\":[\"learning objective\"],\"takeaways\":[\"key takeaway to "
        "remember\"]}. 3-6 objectives, 5-10 takeaways. Only from the notes; invent nothing.")
    user = (f"Course: {analysis.get('course') or 'unknown'} — session: {analysis.get('subject')}\n\n"
            + json.dumps(fiches, ensure_ascii=False, indent=1)[:200_000])
    try:
        return ml._parse_json(ml._chat(usage, 'summary', system, user, 3000, json_mode=True,
                                       temperature=0))
    except (ValueError, json.JSONDecodeError, ic.InferenceError):
        return {}


def _items(d: dict, key: str) -> list:
    return [x for x in (d.get(key) or []) if x]


def report(lines: list, seg: dict, analysis: dict, usage: ml.Usage, dt) -> str:
    topics = seg['topics']
    taught = [t for t in topics if t['kind'] != 'break']
    with ThreadPoolExecutor(max_workers=3) as ex:
        details = list(ex.map(lambda t: _section_detail(t, lines, analysis, usage, dt), taught))
    ov = _overview(taught, details, analysis, usage)
    sp = lambda t: ml._span_label(t, lines)
    md = ['### Overview', '', str(ov.get('overview') or analysis.get('description') or '').strip(), '']
    if ov.get('objectives'):
        md += ['**Learning objectives**', ''] + [f'- {x}' for x in ov['objectives']] + ['']
    if ov.get('takeaways'):
        md += ['### Key takeaways', ''] + [f'- {x}' for x in ov['takeaways']] + ['']
    md += ['### Outline', '']
    for t in topics:
        if t['kind'] == 'break':
            mins = max(0, t['next_start'] - t['start']) // 60
            md.append(f"- _Break / off-topic — excluded ({break_label(t)}, {mins} min)_")
        else:
            tag = {'qa': ' · Q&A', 'logistics': ' · practical'}.get(t['kind'], '')
            md.append(f"{t['num']}. **{t['title']}** ({sp(t)}{tag}) — {t['summary']}")
    glossary, assignments, infos = {}, [], []
    for t, d in zip(taught, details):
        assignments += [(t, a) for a in _items(d, 'assignments') if isinstance(a, dict)]
        infos += [str(x) for x in _items(d, 'infos')]
        if t['kind'] == 'logistics':
            continue
        md += ['', f"### {t['num']}. {t['title']} ({sp(t)})", '']
        for key, label in (('key_points', 'Key points'), ('frameworks', 'Models & frameworks'),
                           ('examples', 'Examples & cases'), ('references', 'References cited')):
            if _items(d, key):
                md += [f'**{label}**', ''] + [f'- {x}' for x in _items(d, key)] + ['']
        concepts = [c for c in _items(d, 'concepts') if isinstance(c, dict) and c.get('term')]
        if concepts:
            md += ['**Concepts**', ''] + [f"- **{c['term']}** — {c.get('definition', '')}"
                                          for c in concepts] + ['']
            for c in concepts:
                glossary.setdefault(ml._fold(c['term']), (c['term'], c.get('definition', '')))
        qa = [q for q in _items(d, 'qa') if isinstance(q, dict) and q.get('question')]
        if qa:
            md += ['**Questions & answers**', '']
            for q in qa:
                md += [f"- **Q:** {q['question']}", f"  **A:** {q.get('answer') or '—'}"]
            md.append('')
    if assignments or infos:
        md += ['### Assignments & practical information', '']
        if assignments:
            md += ['| Section | What | When |', '|---|---|---|']
            md += [f"| {t['num']}. {ml._cell(t['title'])} | {ml._cell(a.get('what', ''))} | "
                   f"{ml._cell(a.get('when', ''))} |" for t, a in assignments]
            md.append('')
        md += [f'- {x}' for x in infos]
        md.append('')
    if glossary:
        md += ['### Glossary', '']
        md += [f'- **{term}** — {defn}' for _, (term, defn) in sorted(glossary.items())]
    return '\n'.join(md).strip()
