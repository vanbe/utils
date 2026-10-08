#!/usr/bin/env python3
"""
meeting_llm.py — post-traitement LLM d'un transcript brut de réunion (via la
passerelle LiteLLM perso, alias de rôle) :

  1. analyse    (MEETING_SUMMARY_MODEL)  : type de réunion, sujet, description, prénoms
                                          des participants (JSON) ; l'ATTRIBUTION des noms
                                          aux étiquettes est faite par règles (pas le LLM) ;
  2. compte rendu (MEETING_SUMMARY_MODEL) : Markdown structuré selon le type ;
  3. transcript rédigé FIDÈLE (MEETING_REWRITE_MODEL), par tranches parallèles.

Contexte : un transcript de 2 h ≈ 110 k caractères ≈ 32 k tokens → tient d'un bloc
en entrée (131–262 k). La SORTIE, elle, ne tient pas d'un coup (≈ entrée) → la
réécriture se fait par tranches de ~6 k caractères, chacune avec l'analyse globale
(noms, glossaire implicite du sujet) et la fin brute de la tranche précédente.
Au-delà de ~300 k caractères (> ~9 h), analyse et compte rendu passent en
map-reduce (notes par partie, puis synthèse).

Raisonnement « low » : réglé côté passerelle (`reasoning_effort: low` sur l'alias
de gpt-oss) ; Mistral-Small n'a pas de mode raisonnement.
"""

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import inference_client as ic
from whisper_common import _env_or_dotenv

MEETING_TYPES = ['1-1', "réunion d'équipe", 'comité/gouvernance', 'atelier technique',
                 'entretien de recrutement', 'appel client/fournisseur', 'appel personnel',
                 'autre']

_REPORT_SECTIONS = {
    '1-1': 'Contexte · Sujets abordés · Feedback échangé · Décisions · Actions (qui / quoi / quand) · Points ouverts',
    "réunion d'équipe": 'Contexte · Points abordés · Décisions · Actions (qui / quoi / quand) · Points ouverts',
    'comité/gouvernance': 'Contexte et objet · Points présentés · Décisions prises · Actions (qui / quoi / quand) · Risques et points ouverts',
    'atelier technique': 'Problème / objectif · Options discutées (avantages / inconvénients) · Choix retenus · Risques · Actions (qui / quoi / quand) · Questions ouvertes',
    'entretien de recrutement': 'Poste et contexte · Parcours du candidat · Points forts · Points de vigilance · Questions posées par le candidat · Suite du processus',
    'appel client/fournisseur': 'Contexte · Besoins exprimés · Propositions / offre · Engagements pris · Actions (qui / quoi / quand) · Points ouverts',
    'appel personnel': 'Résumé · Points importants · À retenir / à faire',
    'autre': 'Contexte · Points abordés · Décisions · Actions (qui / quoi / quand) · Points ouverts',
}

REWRITE_CHUNK_CHARS = 6000
SINGLE_PASS_MAX_CHARS = 300_000


def _model(role: str) -> str:
    return _env_or_dotenv(f'MEETING_{role.upper()}_MODEL') or f'meeting-{role}'


def _report_lang() -> str:
    return _env_or_dotenv('MEETING_REPORT_LANGUAGE') or 'français'


def _fmt_ts(t: float) -> str:
    t = int(t)
    return f'{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}'


def transcript_lines(segs: list) -> list:
    """Segments → répliques `[hh:mm:ss] Label : texte` (labels consécutifs fusionnés)."""
    lines, cur = [], None
    for s in sorted(segs, key=lambda r: r['start']):
        if cur and cur['label'] == s['label'] and s['start'] - cur['end'] < 30:
            cur['text'] += ' ' + s['text']
            cur['end'] = s['end']
        else:
            if cur:
                lines.append(cur)
            cur = dict(s)
    if cur:
        lines.append(cur)
    return [f"[{_fmt_ts(l['start'])}] {l['label'] or 'Inconnu'} : {l['text'].strip()}"
            for l in lines]


class Usage:
    def __init__(self):
        self.cost, self.calls, self.tokens_in, self.tokens_out = 0.0, 0, 0, 0
        self._lock = threading.Lock()                   # réécriture en threads

    def add(self, cost: float, usage: dict):
        with self._lock:
            self.cost += cost or 0.0
            self.calls += 1
            self.tokens_in += int(usage.get('prompt_tokens') or 0)
            self.tokens_out += int(usage.get('completion_tokens') or 0)


def _chat(usage: Usage, role: str, system: str, user: str, max_tokens: int,
          json_mode: bool = False, temperature: float = 0.2) -> str:
    text, cost, u = ic.chat([{'role': 'system', 'content': system},
                             {'role': 'user', 'content': user}],
                            model=_model(role), max_tokens=max_tokens,
                            json_mode=json_mode, temperature=temperature)
    usage.add(cost, u)
    return text.strip()


def _parse_json(text: str) -> dict:
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        raise ValueError('pas de JSON dans la réponse')
    return json.loads(m.group(0))


def _parts(lines: list, max_chars: int) -> list:
    out, cur, n = [], [], 0
    for l in lines:
        if cur and n + len(l) > max_chars:
            out.append(cur)
            cur, n = [], 0
        cur.append(l)
        n += len(l) + 1
    if cur:
        out.append(cur)
    return out


# ── 1. Analyse ─────────────────────────────────────────────────────────────────

_WEEKDAYS = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']


def _date_fr(dt) -> str:
    return f"{_WEEKDAYS[dt.weekday()]} {dt:%d/%m/%Y %H:%M}"


def analyze(lines: list, usage: Usage, *, title_hint: str | None, file_name: str,
            self_name: str | None, labels: list, minor_labels: list, dt,
            self_in_labels: bool) -> dict:
    """Type, sujet, description, tags + IDENTIFICATION DES LOCUTEURS.

    Les noms ne sont PAS attribués par le LLM : même avec raisonnement, gpt-oss et Qwen
    confondent « celui qui dit "merci Laurent" » et Laurent (banc 2026-10-07). Le LLM fait
    seulement l'EXTRACTION des mentions (ligne, nom, présentation / interpellation) ; la
    déduction est déterministe (`_assign_names`)."""
    text = '\n'.join(lines)
    if len(text) > SINGLE_PASS_MAX_CHARS:
        text = _notes(lines, usage)
    system = (
        "Tu analyses la transcription automatique d'une réunion. "
        "Réponds UNIQUEMENT par un objet JSON valide, sans texte autour, avec les clés :\n"
        f'- "meeting_type" : une valeur parmi {json.dumps(MEETING_TYPES, ensure_ascii=False)}\n'
        '- "subject" : sujet court et précis (3 à 8 mots, sans date, sans ponctuation finale)\n'
        '- "description" : une phrase résumant la réunion\n'
        '- "languages" : liste des langues parlées (codes ISO 639-1)\n'
        '- "tags" : 2 à 5 mots-clés thématiques en minuscules\n'
        '- "participants" : prénoms des personnes qui PARLENT dans la réunion (pas celles '
        "seulement citées), orthographe la plus probable ; [] si inconnus\n"
        f"Rédige subject et description en {_report_lang()}.")
    user = (f'Fichier : {file_name}\nDate de la réunion : {_date_fr(dt)}\n'
            + (f"Titre donné par l'utilisateur : {title_hint}\n" if title_hint else '')
            + f'Transcription :\n{text}')
    last = None
    for _ in range(2):
        try:
            data = _parse_json(_chat(usage, 'summary', system, user, 4000))
            break
        except (ValueError, json.JSONDecodeError) as e:
            last = e
    else:
        raise ic.InferenceError(f'analyse : JSON invalide ({last})')
    if data.get('meeting_type') not in MEETING_TYPES:
        data['meeting_type'] = 'autre'
    line_labels = [l.split('] ', 1)[1].split(' : ', 1)[0] if '] ' in l else '' for l in lines]
    names = [n for n in (data.get('participants') or []) if isinstance(n, str) and n.strip()]
    if self_name and not self_in_labels:
        names.append(self_name)                        # audio simple : « Moi » est quelque part
    texts = [l.split(' : ', 1)[1] if ' : ' in l else l for l in lines]
    speakers = _assign_names(_find_mentions(texts, names), line_labels, texts, labels,
                             minor_labels, self_name if self_in_labels else None)
    # Audio simple enregistré par l'utilisateur : s'il n'est pas déjà nommé et qu'il reste
    # UN SEUL locuteur principal anonyme, c'est lui (il participe à son propre enregistrement).
    if self_name and not self_in_labels and self_name not in speakers.values():
        left = [l for l in labels if not speakers.get(l) and l not in minor_labels]
        if len(left) == 1:
            speakers[left[0]] = self_name
    data['speakers'] = speakers
    return data


def _fold(t: str) -> str:
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD', t.lower())
                   if unicodedata.category(c) != 'Mn')


_GREET = r"(?:salut|bonjour|bonsoir|hello|hi|hey|coucou|merci|thanks|thank you|bravo|au revoir|bye|ciao|allez)"


def _find_mentions(texts: list, names: list) -> list:
    """Interpellations / présentations, détectées par RÈGLES (déterministe) :
    « salut|merci|… X », « X, tu|vous|est-ce… », « …, X ? » / « … X. » en fin de phrase
    (apostrophe), « je suis X » / « ici X » / « c'est X » en début de réplique.
    Les simples citations (« j'en ai parlé avec X ») ne matchent pas."""
    out = []
    for name in dict.fromkeys(names):
        n = re.escape(_fold(name.strip()))
        addressed = [
            re.compile(rf"\b{_GREET}\s*,?\s+{n}\b"),
            re.compile(rf"(?:^|[.!?]\s+){n}\s*,\s*(?:tu|t'|vous|est-ce|je|on|dis|ecoute|attends)\b"),
            re.compile(rf"(?:,|\b(?:non|oui|ok|okay|d'accord|ta|ton|tes|toi))\s+{n}\s*(?:[?!.]|$)"),
        ]
        selfp = re.compile(rf"^(?:\W*)(?:oui\s*,?\s*)?(?:je suis|moi c'est|ici|c'est)\s+{n}\b")
        for i, t in enumerate(texts):
            f = _fold(t)
            if selfp.search(f):
                out.append({'line': i, 'name': name, 'kind': 'self'})
            elif any(p.search(f) for p in addressed):
                out.append({'line': i, 'name': name, 'kind': 'addressed'})
    return out


_MERCI = re.compile(r'\b(merci|thanks|thank you|bravo|exactement|tout à fait)\b', re.I)


def _assign_names(mentions: list, line_labels: list, line_texts: list, labels: list,
                  minor: list, self_label_name: str | None) -> dict:
    """Votes déterministes étiquette↔nom à partir des mentions extraites :
    - "self" à la ligne i            → étiquette de i (poids 3) ;
    - "addressed" à la ligne i par X → l'interlocuteur = la réplique voisine d'une AUTRE
      étiquette : précédente (poids 2) / suivante (1) pour un remerciement, l'inverse pour
      une salutation ou une question (on s'adresse à celui qui va répondre).
    Puis appariement glouton (meilleur score d'abord), 1 nom ↔ 1 étiquette, seuil 2.
    Les étiquettes mineures ne reçoivent pas de nom (trop peu de matière)."""
    votes: dict = {}

    def add(label, name, w):
        if label and label != 'Moi' and label in labels and label not in minor:
            votes[(label, name)] = votes.get((label, name), 0) + w

    for m in mentions:
        try:
            i, name, kind = int(m.get('line')), (m.get('name') or '').strip(), m.get('kind')
        except (TypeError, ValueError):
            continue
        if not name or not (0 <= i < len(line_labels)):
            continue
        if self_label_name and name.lower() == self_label_name.lower():
            continue                                   # « Moi » est déjà nommé par construction
        me = line_labels[i]
        if kind == 'self':
            add(me, name, 3)
        elif kind == 'addressed':
            prev = next((line_labels[j] for j in range(i - 1, -1, -1) if line_labels[j] != me), None)
            nxt = next((line_labels[j] for j in range(i + 1, len(line_labels))
                        if line_labels[j] != me), None)
            # remerciement → on répond à ce qui vient d'être dit (réplique précédente)
            w_prev, w_next = (2, 1) if _MERCI.search(line_texts[i]) else (1, 2)
            add(prev, name, w_prev)
            add(nxt, name, w_next)
    speakers = {l: None for l in labels}
    if self_label_name and 'Moi' in speakers:
        speakers['Moi'] = self_label_name
    used = set()
    for (label, name), score in sorted(votes.items(), key=lambda kv: -kv[1]):
        if score < 2 or speakers.get(label) or name.lower() in used:
            continue
        speakers[label] = name
        used.add(name.lower())
    return speakers


def _notes(lines: list, usage: Usage) -> str:
    """Map-reduce pour les très longues réunions : notes détaillées par partie."""
    system = ("Prends des notes détaillées et fidèles de cet extrait de réunion : qui dit quoi, "
              "décisions, chiffres, actions, noms propres. Garde les étiquettes de locuteurs "
              f"et les repères horaires. En {_report_lang()}.")
    notes = [_chat(usage, 'summary', system, '\n'.join(p), 6000)
             for p in _parts(lines, SINGLE_PASS_MAX_CHARS // 2)]
    return '\n\n'.join(notes)


# ── 2. Compte rendu ────────────────────────────────────────────────────────────

def report(lines: list, analysis: dict, usage: Usage, dt) -> str:
    text = '\n'.join(lines)
    if len(text) > SINGLE_PASS_MAX_CHARS:
        text = _notes(lines, usage)
    mtype = analysis['meeting_type']
    names = {l: n for l, n in analysis['speakers'].items() if n}
    system = (
        f"Tu rédiges le compte rendu professionnel d'une réunion (type : {mtype}), en "
        f"{_report_lang()}, au format Markdown. Sections (titres de niveau ###, omets une "
        f"section vide) : {_REPORT_SECTIONS[mtype]}. Style : clair, factuel, concis, phrases "
        "complètes ; listes à puces pour les points. Attribue les propos et actions aux "
        "personnes par leur nom quand il est connu. N'invente rien : si une échéance ou un "
        "responsable n'est pas dit, ne le suppose pas. Les dates relatives (« mercredi "
        "prochain », « demain ») se calculent depuis la date de la réunion ; en cas de doute, "
        "garde la formulation d'origine. Pas de titre de niveau 1 ou 2, pas de préambule ni "
        "de conclusion hors sections.")
    user = (f"Sujet : {analysis.get('subject')}\nDate de la réunion : {_date_fr(dt)}\n"
            f"Correspondance des locuteurs : {json.dumps(names, ensure_ascii=False)}\n\n"
            f"Transcription :\n{text}")
    return _chat(usage, 'summary', system, user, 6000)


# ── 2bis. Réunion à plusieurs sujets ───────────────────────────────────────────
#
# Banc 2026-10-08 (3 réunions réelles, 1 h–1 h 42) : le découpage par le LLM tombe aux bons
# endroits (comité d'archi du 06/10 = 3 sujets sans rapport). Chaîne : (1) découpage = le LLM
# donne des sujets + l'heure de début de chaque passage, le CODE range les répliques et impose
# durée minimale / nombre max ; (2) un appel JSON PAR SUJET (en parallèle) sur ses seules
# répliques → points, décisions, actions, points ouverts ; (3) synthèse globale écrite depuis
# les fiches de sujets ; (4) tableau « Actions par sujet » assemblé par le CODE (aucune action
# ne peut être perdue ni reformulée en route). 1 seul sujet / échec → compte rendu classique.

MAX_TOPICS = 8
MIN_TOPIC_SEC = 180            # passage plus court = digression, rattaché au voisin


def _line_ts(line: str) -> int:
    m = re.match(r'^\[(\d+):(\d+):(\d+)\]', line)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else 0


def _parse_hms(s) -> int | None:
    parts = [int(p) for p in re.findall(r'\d+', str(s or ''))]
    if not parts or len(parts) > 3:
        return None
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _hm(sec: int) -> str:
    """Position dans l'enregistrement, format UNIQUE h:mm (0:41, 1:00)."""
    return f'{sec // 3600}:{sec % 3600 // 60:02d}'


def segment_topics(lines: list, analysis: dict, usage: Usage) -> dict | None:
    """→ {'topics': [{id, title, summary, spans:[(i0, i1)], start, end}], 'line_topic': [...]}
    ou None (un seul sujet, transcript trop long, réponse inexploitable)."""
    text = '\n'.join(lines)
    if len(lines) < 10 or len(text) > SINGLE_PASS_MAX_CHARS:
        return None
    system = (
        "Tu découpes la transcription d'une réunion en SUJETS distincts. Un sujet = un thème de "
        "discussion cohérent (objet, décision ou problème différent), pas un simple changement "
        "d'intervenant. Fusionne les digressions courtes (< 3 min) dans le sujet voisin ; "
        "salutations, tour de table et logistique → un sujet « Introduction » ou « Organisation "
        "et prochaines étapes » seulement s'ils durent. Un même sujet peut revenir plus tard : "
        f"réutilise alors son id. Entre 1 et {MAX_TOPICS} sujets ; une réunion à un seul thème "
        "renvoie UN sujet. Réponds en JSON : {\"topics\":[{\"id\":1,\"title\":\"titre court "
        "(3 à 8 mots)\",\"summary\":\"une phrase\"}],\"segments\":[{\"start\":\"hh:mm:ss\","
        "\"topic\":1}]} — segments = début de chaque passage, dans l'ordre chronologique, le "
        f"premier à 00:00:00. Titres et résumés en {_report_lang()}.")
    user = (f"Réunion : {analysis.get('subject')} ({analysis.get('meeting_type')})\n\n"
            f"Transcription :\n{text}")
    try:
        d = _parse_json(_chat(usage, 'summary', system, user, 3000, json_mode=True, temperature=0))
        topics = {int(t['id']): {'id': int(t['id']), 'title': str(t['title']).strip(),
                                 'summary': str(t.get('summary') or '').strip()}
                  for t in d.get('topics') or [] if str(t.get('title') or '').strip()}
        segs = sorted(((_parse_hms(s.get('start')), int(s.get('topic')))
                       for s in d.get('segments') or []
                       if _parse_hms(s.get('start')) is not None and int(s.get('topic')) in topics),
                      key=lambda x: x[0])
    except (ValueError, TypeError, KeyError, ic.InferenceError):
        return None
    if len(topics) < 2 or not segs:
        return None
    # Répliques → sujet (dernier passage commencé avant la réplique)
    starts = [s for s, _ in segs]
    line_topic = []
    for l in lines:
        ts = _line_ts(l)
        k = max(0, max((i for i, s in enumerate(starts) if s <= ts), default=0))
        line_topic.append(segs[k][1])
    # Passages contigus ; ceux trop courts sont rattachés au voisin (précédent, sinon suivant)
    def runs(lt):
        out, i0 = [], 0
        for i in range(1, len(lt) + 1):
            if i == len(lt) or lt[i] != lt[i0]:
                out.append([i0, i, lt[i0]])
                i0 = i
        return out
    changed = True
    while changed:
        changed = False
        rs = runs(line_topic)
        if len(rs) < 2:
            break
        for j, (i0, i1, t) in enumerate(rs):
            end_ts = _line_ts(lines[rs[j + 1][0]]) if j + 1 < len(rs) else _line_ts(lines[-1])
            if end_ts - _line_ts(lines[i0]) < MIN_TOPIC_SEC:
                nt = rs[j - 1][2] if j > 0 else rs[j + 1][2]
                line_topic[i0:i1] = [nt] * (i1 - i0)
                changed = True
                break
    rs = runs(line_topic)
    present = list(dict.fromkeys(t for _, _, t in rs))           # ordre d'apparition
    if len(present) < 2 or len(present) > MAX_TOPICS:
        return None
    out = []
    for n, tid in enumerate(present, 1):
        spans = [(i0, i1) for i0, i1, t in rs if t == tid]
        t = dict(topics[tid])
        t.update(num=n, spans=spans, start=_line_ts(lines[spans[0][0]]),
                 end=_line_ts(lines[spans[-1][1] - 1]))
        out.append(t)
    renum = {t['id']: t['num'] for t in out}
    return {'topics': out, 'line_topic': [renum[t] for t in line_topic]}


def _span_label(t: dict, lines: list) -> str:
    return ', '.join(f'{_hm(_line_ts(lines[i0]))}–{_hm(_line_ts(lines[i1 - 1]))}'
                     for i0, i1 in t['spans'])


def _topic_detail(t: dict, lines: list, analysis: dict, usage: Usage, dt) -> dict:
    names = {l: n for l, n in analysis['speakers'].items() if n}
    system = (
        f"Tu analyses UN SUJET d'une réunion ({analysis.get('meeting_type')}), à partir des seules "
        f"répliques qui le concernent. Réponds en JSON, en {_report_lang()} : "
        "{\"points\":[\"point clé (phrase complète)\"],\"decisions\":[\"décision prise\"],"
        "\"actions\":[{\"qui\":\"nom ou rôle\",\"quoi\":\"action\",\"quand\":\"échéance\"}],"
        "\"ouverts\":[\"question ou point non tranché\"]}. Règles : n'invente rien ; une "
        "DÉCISION ou une ACTION n'est retenue que si elle est explicitement actée ou assignée "
        "(sinon c'est un point ou un point ouvert) ; qui / quand vides (\"\") s'ils ne sont pas "
        "dits ; les dates relatives se calculent depuis la date de la réunion ; nomme les "
        "personnes quand leur nom est connu ; listes vides si rien. 3 à 8 points au plus.")
    body = '\n'.join(lines[i] for i0, i1 in t['spans'] for i in range(i0, i1))
    user = (f"Réunion : {analysis.get('subject')} — date : {_date_fr(dt)}\n"
            f"Sujet : {t['title']} — {t['summary']}\n"
            f"Noms des locuteurs : {json.dumps(names, ensure_ascii=False)}\n\n"
            f"Répliques du sujet :\n{body}")
    try:
        d = _parse_json(_chat(usage, 'summary', system, user, 4000, json_mode=True, temperature=0))
    except (ValueError, json.JSONDecodeError):
        d = {}

    def strs(k):
        return [str(x).strip() for x in d.get(k) or [] if str(x).strip()]
    acts = [{'qui': str(a.get('qui') or '').strip(), 'quoi': str(a.get('quoi') or '').strip(),
             'quand': str(a.get('quand') or '').strip()}
            for a in d.get('actions') or [] if isinstance(a, dict) and str(a.get('quoi') or '').strip()]
    return {'points': strs('points'), 'decisions': strs('decisions'), 'actions': acts,
            'ouverts': strs('ouverts')}


def _synthesis(topics: list, details: list, analysis: dict, usage: Usage, dt) -> str:
    names = sorted({n for n in analysis['speakers'].values() if n})
    fiches = [{'sujet': f"{t['num']}. {t['title']}", 'resume': t['summary'], **d}
              for t, d in zip(topics, details)]
    system = (
        f"Tu rédiges la SYNTHÈSE GLOBALE d'une réunion en {_report_lang()} : un paragraphe de 4 à "
        "7 phrases qui donne le contexte (objet, participants), enchaîne les sujets abordés et "
        "résume les principaux résultats (décisions, suites). Uniquement à partir des fiches "
        "fournies ; n'invente rien. Texte simple, sans titre ni liste.")
    user = (f"Réunion : {analysis.get('subject')} ({analysis.get('meeting_type')}) — "
            f"{_date_fr(dt)} — participants : {', '.join(names) or 'non identifiés'}\n\n"
            f"Fiches des sujets :\n{json.dumps(fiches, ensure_ascii=False, indent=1)}")
    return _chat(usage, 'summary', system, user, 1500)


def _cell(s: str) -> str:
    return (s or '—').replace('|', '\\|').replace('\n', ' ')


def report_by_topics(lines: list, seg: dict, analysis: dict, usage: Usage, dt) -> str:
    topics = seg['topics']
    with ThreadPoolExecutor(max_workers=3) as ex:
        details = list(ex.map(lambda t: _topic_detail(t, lines, analysis, usage, dt), topics))
    synth = _synthesis(topics, details, analysis, usage, dt)
    md = ['### Synthèse globale', '', synth.strip(), '', '### Sujets abordés', '']
    for t in topics:
        md.append(f"{t['num']}. **{t['title']}** ({_span_label(t, lines)}) — {t['summary']}")
    for t, d in zip(topics, details):
        md += ['', f"### {t['num']}. {t['title']} ({_span_label(t, lines)})", '']
        for key, label in (('points', 'Points clés'), ('decisions', 'Décisions'),
                           ('ouverts', 'Points ouverts')):
            if d[key]:
                md += [f'**{label}**', ''] + [f'- {x}' for x in d[key]] + ['']
        if not any(d[k] for k in ('points', 'decisions', 'ouverts')):
            md += ['_Rien de notable._', '']
    md += ['### Actions par sujet', '']
    rows = [(t, a) for t, d in zip(topics, details) for a in d['actions']]
    if rows:
        md += ['| Sujet | Qui | Quoi | Quand |', '|---|---|---|---|']
        md += [f"| {t['num']}. {_cell(t['title'])} | {_cell(a['qui'])} | {_cell(a['quoi'])} | "
               f"{_cell(a['quand'])} |" for t, a in rows]
    else:
        md.append('_Aucune action explicitement assignée._')
    return '\n'.join(md).strip()


# ── 3. Transcript rédigé fidèle ────────────────────────────────────────────────

def rewrite(lines: list, analysis: dict, usage: Usage, workers: int = 3,
            seg: dict | None = None) -> str:
    # Blocs = passages contigus d'un même sujet (intertitre « ### n. Titre ») ; sinon un seul.
    blocks = []
    if seg:
        titles = {t['num']: t['title'] for t in seg['topics']}
        lt, i0 = seg['line_topic'], 0
        for i in range(1, len(lines) + 1):
            if i == len(lines) or lt[i] != lt[i0]:
                blocks.append((f"### {lt[i0]}. {titles[lt[i0]]}", lines[i0:i]))
                i0 = i
    else:
        blocks = [(None, lines)]
    tasks = [(b, p) for b, (_, bl) in enumerate(blocks) for p in _parts(bl, REWRITE_CHUNK_CHARS)]
    parts = [p for _, p in tasks]
    names = {l: n for l, n in analysis['speakers'].items() if n}
    system = (
        "Tu réécris un extrait de transcription automatique de réunion en version RÉDIGÉE "
        "FIDÈLE, dans la langue d'origine des propos.\n"
        "Règles :\n"
        "- garde TOUS les propos, dans le même ordre ; ne résume pas, ne commente pas ;\n"
        "- retire hésitations, répétitions, faux départs et tics de langage ;\n"
        "- corrige les erreurs évidentes de reconnaissance vocale (mots mal entendus, noms "
        "propres, acronymes) d'après le contexte ; ponctuation et phrases propres, registre "
        "professionnel ;\n"
        "- les étiquettes viennent d'une séparation automatique imparfaite : recolle une "
        "phrase coupée à tort entre deux répliques quand c'est évident ;\n"
        "- remplace les étiquettes par les noms fournis ; garde l'étiquette sinon ; une "
        "étiquette marquée « mineure » est souvent une erreur de séparation : attribue la "
        "réplique au locuteur évident d'après le contexte, sinon garde l'étiquette ;\n"
        "- un passage inintelligible devient « [inaudible] » — n'invente jamais de contenu ;\n"
        "- format : un paragraphe par réplique, « **Nom** : texte ». Rien d'autre (pas de "
        "titre, pas de repère horaire, pas de note).")
    ctx = (f"Réunion : {analysis.get('subject')} ({analysis.get('meeting_type')}) — "
           f"{analysis.get('description')}\n"
           f"Noms des locuteurs : {json.dumps(names, ensure_ascii=False)}\n"
           + (f"Étiquettes mineures : {', '.join(analysis.get('minor_labels') or [])}\n"
              if analysis.get('minor_labels') else ''))

    def one(i: int) -> str:
        prev = ('\n'.join(parts[i - 1][-4:]) if i else '')
        user = (ctx
                + (f"\nFin de l'extrait précédent (contexte seulement, NE PAS réécrire) :\n"
                   f"{prev}\n" if prev else '')
                + "\nExtrait à réécrire :\n" + '\n'.join(parts[i]))
        n_in = sum(len(l) for l in parts[i])
        return _chat(usage, 'rewrite', system, user, max(1500, min(8000, n_in // 2)))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(one, range(len(parts))))
    out, cur = [], None
    for (b, _), o in zip(tasks, outs):
        if b != cur:
            cur = b
            if blocks[b][0]:
                out.append(blocks[b][0])
        if o.strip():
            out.append(o.strip())
    return '\n\n'.join(out)
