#!/usr/bin/env python3
"""
inference_client.py — client OpenAI-compatible MINIMAL (stdlib seule : urllib)
pour la transcription (Whisper) et le chat (LLM), via la passerelle LiteLLM perso
ou directement OVH AI Endpoints.

Pourquoi stdlib : tourne tel quel dans l'image `utils:latest` (Airflow) et sur un
laptop sans GPU, sans `openai`/`requests` à installer.

Configuration (env, ou `.env` projet) :
  INFERENCE_BASE_URL   ex. http://192.168.1.15:4000/v1   (LiteLLM perso)
  INFERENCE_API_KEY    clé virtuelle LiteLLM
  OVH_AI_BASE_URL      défaut https://oai.endpoints.kepler.ai.cloud.ovh.net/v1
  OVH_AI_API_KEY       token OVH — UNIQUEMENT pour la diarisation : l'endpoint
                       OpenAI de LiteLLM re-modélise la réponse et perd le champ
                       `diarization` (extension OVH). Tout le reste passe par
                       LiteLLM (routage local/OVH, coûts, clés).

Coût : LiteLLM le renvoie dans l'en-tête `x-litellm-response-cost` ; en direct
OVH on le calcule (`usage.seconds` × prix/seconde).
"""

import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid

from whisper_common import _env_or_dotenv

OVH_DEFAULT_BASE = 'https://oai.endpoints.kepler.ai.cloud.ovh.net/v1'
OVH_WHISPER_EUR_PER_SEC = 0.00004083           # whisper-large-v3, catalogue OVH 2026-10

_RETRY_HTTP = {429, 500, 502, 503, 504}


class InferenceError(RuntimeError):
    pass


def _cfg(key: str, default: str = '') -> str:
    return _env_or_dotenv(key) or default


def gateway() -> tuple[str, str]:
    """(base_url, api_key) de la passerelle LiteLLM."""
    base = _cfg('INFERENCE_BASE_URL').rstrip('/')
    key = _cfg('INFERENCE_API_KEY')
    if not base or not key:
        raise InferenceError('INFERENCE_BASE_URL / INFERENCE_API_KEY non configurés')
    return base, key


def ovh_direct() -> tuple[str, str]:
    """(base_url, api_key) OVH direct (diarisation seulement)."""
    key = _cfg('OVH_AI_API_KEY')
    if not key:
        raise InferenceError('OVH_AI_API_KEY non configuré (requis pour la diarisation)')
    return _cfg('OVH_AI_BASE_URL', OVH_DEFAULT_BASE).rstrip('/'), key


def _request(url: str, key: str, body: bytes, content_type: str,
             timeout: int, retries: int = 4) -> tuple[dict, dict]:
    """POST avec retries exponentiels sur 429/5xx/erreurs réseau → (json, headers)."""
    delay = 5.0
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, method='POST', headers={
            'Authorization': f'Bearer {key}', 'Content-Type': content_type})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode('utf-8')), dict(r.headers)
        except urllib.error.HTTPError as e:
            detail = e.read()[:500].decode('utf-8', 'replace')
            if e.code not in _RETRY_HTTP or attempt == retries:
                raise InferenceError(f'HTTP {e.code} sur {url} : {detail}') from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries:
                raise InferenceError(f'réseau sur {url} : {e}') from None
        time.sleep(delay)
        delay *= 2
    raise InferenceError('unreachable')


def _multipart(fields: dict, file_field: str, file_path: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        if v is None:
            continue
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; '
                      f'name="{k}"\r\n\r\n{v}\r\n').encode('utf-8'))
    ctype = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
    with open(file_path, 'rb') as f:
        data = f.read()
    parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
                  f'filename="{os.path.basename(file_path)}"\r\n'
                  f'Content-Type: {ctype}\r\n\r\n').encode('utf-8') + data + b'\r\n')
    parts.append(f'--{boundary}--\r\n'.encode('utf-8'))
    return b''.join(parts), f'multipart/form-data; boundary={boundary}'


def transcribe(audio_path: str, language: str | None = None, diarize: bool = False,
               model: str | None = None, timeout: int = 1800) -> tuple[dict, float]:
    """Transcrit un fichier audio → (verbose_json, coût €).

    diarize=False : passerelle LiteLLM, alias `ASR_MODEL` (défaut `asr`).
    diarize=True  : OVH direct, `whisper-large-v3` + `diarize=true`.
    `chunking_strategy=auto` (VAD serveur) est TOUJOURS envoyé : sans lui, Whisper
    hallucine sur les fenêtres silencieuses (« Sous-titrage ST' 501 »)."""
    fields = {'response_format': 'verbose_json', 'chunking_strategy': 'auto',
              'language': language or None}
    if diarize:
        base, key = ovh_direct()
        fields.update(model=model or 'whisper-large-v3', diarize='true')
    else:
        base, key = gateway()
        fields['model'] = model or _cfg('ASR_MODEL', 'asr')
    body, ctype = _multipart(fields, 'file', audio_path)
    data, headers = _request(f'{base}/audio/transcriptions', key, body, ctype, timeout)
    cost = _header_cost(headers)
    if cost is None:
        secs = ((data.get('usage') or {}).get('seconds') or data.get('duration') or 0.0)
        cost = float(secs) * OVH_WHISPER_EUR_PER_SEC
    return data, cost


def chat(messages: list, model: str, max_tokens: int = 4000, temperature: float = 0.2,
         timeout: int = 600, json_mode: bool = False) -> tuple[str, float, dict]:
    """Chat completion via la passerelle → (texte, coût €, usage). json_mode : JSON garanti."""
    base, key = gateway()
    payload = {'model': model, 'messages': messages, 'max_tokens': max_tokens,
               'temperature': temperature}
    if json_mode:
        payload['response_format'] = {'type': 'json_object'}
    body = json.dumps(payload).encode('utf-8')
    data, headers = _request(f'{base}/chat/completions', key, body,
                             'application/json', timeout)
    try:
        choice = data['choices'][0]
        text = choice['message'].get('content') or ''
    except (KeyError, IndexError):
        raise InferenceError(f'réponse inattendue : {str(data)[:300]}') from None
    if choice.get('finish_reason') == 'length':
        raise InferenceError(f'réponse tronquée (max_tokens={max_tokens}) pour {model}')
    return text, _header_cost(headers) or 0.0, data.get('usage') or {}


def _header_cost(headers: dict) -> float | None:
    for k, v in headers.items():
        if k.lower() == 'x-litellm-response-cost':
            try:
                return float(v)
            except ValueError:
                return None
    return None
