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


def _chat(usage: Usage, role: str, system: str, user: str, max_tokens: int) -> str:
    text, cost, u = ic.chat([{'role': 'system', 'content': system},
                             {'role': 'user', 'content': user}],
                            model=_model(role), max_tokens=max_tokens)
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


# ── 3. Transcript rédigé fidèle ────────────────────────────────────────────────

def rewrite(lines: list, analysis: dict, usage: Usage, workers: int = 3) -> str:
    parts = _parts(lines, REWRITE_CHUNK_CHARS)
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
    return '\n\n'.join(o.strip() for o in outs if o.strip())
