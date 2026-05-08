#!/usr/bin/env python3
"""
engine.py — Backend abstraction for local AI inference.

Supported engines:
  lmstudio  — LMStudio OpenAI-compatible API  (default port 1234)
  ollama    — Ollama REST API                  (default port 11434)

Both expose an OpenAI-compatible /v1/ endpoint, so the same openai.OpenAI
client works for both. Differences are in model listing, lifecycle management,
and context-size detection, which this module abstracts away.

Usage:
  from engine import create_backend, resolve_host

  backend = create_backend('ollama', resolve_host('http://localhost:11434'))
  client  = backend.create_client()
  model   = backend.get_loaded()       # currently active model id
  backend.load('llama3.2:latest', 8192)
  backend.unload('llama3.2:latest')
"""

import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

_LOAD_TIMEOUT = 600   # 10 min — large models (26 B+) can take several minutes to load
_DEFAULT_CTX  = 8192


# ---------------------------------------------------------------------------
# WSL2 host resolution  (shared by both backends)
# ---------------------------------------------------------------------------

def resolve_host(host: str) -> str:
    """
    If host references localhost/127.0.0.1, verify connectivity first.
    Falls back to the Windows host IP from /etc/resolv.conf when running in
    WSL2 NAT mode (engine runs on the Windows side, not inside WSL).
    If the engine runs natively inside WSL2 (Ollama), localhost works fine.
    """
    if 'localhost' not in host and '127.0.0.1' not in host:
        return host
    try:
        port = int(urllib.parse.urlparse(host).port or 80)
    except Exception:
        port = 80
    try:
        with socket.create_connection(('localhost', port), timeout=2):
            return host   # localhost reachable — use it directly
    except OSError:
        pass
    # NAT mode: resolve Windows host IP via nameserver entry
    try:
        with open('/etc/resolv.conf', encoding='utf-8') as f:
            for line in f:
                if line.startswith('nameserver'):
                    ip = line.split()[1].strip()
                    return host.replace('localhost', ip).replace('127.0.0.1', ip)
    except OSError:
        pass
    return host


# ---------------------------------------------------------------------------
# Backend base class
# ---------------------------------------------------------------------------

class Backend:
    """Abstract interface for a local AI inference backend."""

    ENGINE       = ''
    DEFAULT_HOST = ''
    _API_KEY     = ''

    def __init__(self, host: str):
        self.host = host   # pre-resolved API base URL (no trailing slash)

    # -- Client ---------------------------------------------------------------

    def create_client(self):
        """Return an openai.OpenAI client for this backend's /v1/ endpoint.

        The read timeout is set to 30 minutes to accommodate large vision models
        that can take several minutes per page on complex layouts.
        """
        from openai import OpenAI
        return OpenAI(
            base_url=f"{self.host}/v1",
            api_key=self._API_KEY,
            timeout=1800.0,   # 30 min — large vision models can be slow
        )

    # -- Inference options ----------------------------------------------------

    def get_inference_extra_body(self) -> dict:
        """
        Extra fields to inject into every OpenAI-compat inference call.

        Ollama reloads the model if inference options (num_ctx, num_gpu) differ
        from what was used at load time.  Returning the original load options here
        prevents that silent reload.
        LMStudio uses the standard OpenAI API with no extra fields.
        """
        return {}

    # -- Model discovery ------------------------------------------------------

    def list_available(self) -> list:
        """
        List models available (installed) on this backend.
        Returns [{'id': str, 'ctx': int|None}, ...]
        """
        raise NotImplementedError

    def get_loaded(self) -> 'str | None':
        """Return the model_id currently in VRAM/active, or None."""
        raise NotImplementedError

    def get_ctx(self, model_id: str) -> 'int | None':
        """Return the configured context-window size for a model, or None."""
        return None

    # -- Lifecycle ------------------------------------------------------------

    def load(self, model_id: str, ctx: int, num_gpu: 'int | None' = None) -> bool:
        """
        Request the backend to load `model_id` with `ctx` token context.

        num_gpu: number of model layers to offload to the GPU (Ollama: num_gpu
                 option; LMStudio: ignored, always uses max GPU offload).
                 None means 'let the backend decide'.

        Returns True if the request was accepted (may not be ready yet —
        call wait_ready() to confirm, except for Ollama which is synchronous).
        """
        raise NotImplementedError

    def unload(self, model_id: str) -> bool:
        """Evict `model_id` from VRAM. Returns True on success."""
        raise NotImplementedError

    def wait_ready(self, model_id: str, timeout: int = _LOAD_TIMEOUT) -> bool:
        """Poll until `model_id` is fully loaded. Returns True on success."""
        raise NotImplementedError

    # -- Internal HTTP helpers ------------------------------------------------

    def _get(self, path: str, timeout: int = 30) -> dict:
        req = urllib.request.Request(f"{self.host}{path}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _post(self, path: str, body: dict, timeout: int = 30) -> dict:
        data = json.dumps(body).encode()
        req  = urllib.request.Request(
            f"{self.host}{path}", data=data, method='POST',
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())


# ---------------------------------------------------------------------------
# LMStudio backend
# ---------------------------------------------------------------------------

class LMStudioBackend(Backend):
    """
    LMStudio local inference server (https://lmstudio.ai).
    Uses LMStudio's native /api/v0/ endpoints for model lifecycle and
    the standard /v1/ OpenAI-compatible endpoint for inference.
    Default port: 1234.
    """

    ENGINE       = 'lmstudio'
    DEFAULT_HOST = 'http://localhost:1234'
    _API_KEY     = 'lm-studio'

    def list_available(self) -> list:
        # Prefer /api/v0/models — it includes context_length
        try:
            data = self._get('/api/v0/models')
            return [
                {'id': m['id'], 'ctx': m.get('context_length')}
                for m in data.get('data', [])
            ]
        except Exception:
            pass
        try:
            data = self._get('/v1/models')
            return [{'id': m['id'], 'ctx': None} for m in data.get('data', [])]
        except Exception:
            return []

    def get_loaded(self) -> 'str | None':
        try:
            data = self._get('/api/v0/models')
            for m in data.get('data', []):
                if m.get('state') in ('loaded', 'loading'):
                    return m['id']
        except Exception:
            pass
        # Fallback: OpenAI-compat /v1/models (e.g. older LMStudio versions)
        try:
            data = self._get('/v1/models')
            models = data.get('data', [])
            if models:
                return models[0]['id']
        except Exception:
            pass
        return None

    def get_ctx(self, model_id: str) -> 'int | None':
        try:
            data = self._get('/api/v0/models')
            for m in data.get('data', []):
                if m['id'] == model_id:
                    return m.get('context_length')
        except Exception:
            pass
        return None

    def load(self, model_id: str, ctx: int, num_gpu: 'int | None' = None) -> bool:
        try:
            self._post('/api/v0/models/load', {
                'identifier':     model_id,
                'context_length': ctx,
                'gpu_offload':    {'ratio': 'max'},   # always max for LMStudio
            })
            return True
        except urllib.error.HTTPError as e:
            e.read()   # drain body to avoid resource leak
            return False
        except Exception:
            return False

    def unload(self, model_id: str) -> bool:
        try:
            self._post('/api/v0/models/unload', {'identifier': model_id})
            return True
        except Exception:
            return False

    def wait_ready(self, model_id: str, timeout: int = _LOAD_TIMEOUT) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                data = self._get('/api/v0/models')
                for m in data.get('data', []):
                    if m['id'] == model_id and m.get('state') == 'loaded':
                        return True
            except Exception:
                pass
            time.sleep(2)
        return False


# ---------------------------------------------------------------------------
# Native Ollama client wrapper
#
# Mirrors the small subset of the OpenAI SDK surface that this codebase uses
# (`client.chat.completions.create(...)`) but talks to Ollama's native
# /api/chat endpoint instead of /v1/chat/completions.
#
# Why: Ollama's OpenAI-compat endpoint silently drops `keep_alive` and
# `options` (num_ctx / num_gpu / etc.) from the JSON body. Any inference call
# therefore arrives at the runner with the modelfile defaults, causing the
# loaded model to be reset:
#   - num_ctx flips back to the modelfile default (often 4096), undoing the
#     large context we configured at load() time.
#   - keep_alive drops from `-1` (Forever) to the 5 min default, so the
#     runner can be evicted between calls and reloaded with stale settings.
# /api/chat respects both fields, so we use it directly.
# ---------------------------------------------------------------------------

class _OllamaMsg:
    """OpenAI message-shaped object (for both .message and .delta access)."""
    __slots__ = ('content', 'role')
    def __init__(self, content: str = '', role: str = 'assistant'):
        self.content = content
        self.role    = role


class _OllamaChoice:
    __slots__ = ('message', 'delta', 'finish_reason', 'index')
    def __init__(self, *, message=None, delta=None, finish_reason=None):
        self.message       = message
        self.delta         = delta
        self.finish_reason = finish_reason
        self.index         = 0


class _OllamaResponse:
    """Mimics openai ChatCompletion (for stream=False)."""
    __slots__ = ('choices',)
    def __init__(self, content: str, finish_reason: str = 'stop'):
        self.choices = [_OllamaChoice(
            message=_OllamaMsg(content=content),
            finish_reason=finish_reason,
        )]


class _OllamaStreamChunk:
    """Mimics openai ChatCompletionChunk (for stream=True)."""
    __slots__ = ('choices',)
    def __init__(self, content: str = '', finish_reason=None):
        self.choices = [_OllamaChoice(
            delta=_OllamaMsg(content=content),
            finish_reason=finish_reason,
        )]


def _to_native_messages(messages: list) -> list:
    """
    Translate OpenAI-style messages (incl. multimodal `image_url` parts)
    into Ollama's native /api/chat format with text content + base64
    `images` field.
    """
    out = []
    for msg in messages:
        content = msg.get('content', '')
        role    = msg.get('role', 'user')
        if isinstance(content, list):
            text_parts: list = []
            images:     list = []
            for part in content:
                ptype = part.get('type')
                if ptype == 'text':
                    text_parts.append(part.get('text', ''))
                elif ptype == 'image_url':
                    url = part.get('image_url', {}).get('url', '')
                    if url.startswith('data:') and ',' in url:
                        images.append(url.split(',', 1)[1])
                    elif url:
                        images.append(url)   # raw base64 fallback
            native = {'role': role, 'content': '\n'.join(text_parts)}
            if images:
                native['images'] = images
            out.append(native)
        else:
            out.append({'role': role, 'content': content})
    return out


class _OllamaCompletionsAPI:
    def __init__(self, host: str):
        self._host = host

    def create(self, *, model, messages, stream=False, timeout=180,
               temperature=None, max_tokens=None, top_p=None,
               frequency_penalty=None, presence_penalty=None,
               extra_body=None, **_ignored):
        # Build options dict — start from request-level params, then overlay
        # anything the caller set in extra_body['options'] (those win, so
        # the load-time num_ctx / num_gpu always make it through).
        opts: dict = {}
        if temperature       is not None: opts['temperature']       = temperature
        if max_tokens        is not None: opts['num_predict']       = max_tokens
        if top_p             is not None: opts['top_p']             = top_p
        if frequency_penalty is not None: opts['frequency_penalty'] = frequency_penalty
        if presence_penalty  is not None: opts['presence_penalty']  = presence_penalty

        keep_alive = -1
        if extra_body:
            if 'options' in extra_body and isinstance(extra_body['options'], dict):
                opts.update(extra_body['options'])
            if 'keep_alive' in extra_body:
                keep_alive = extra_body['keep_alive']

        body = {
            'model':      model,
            'messages':   _to_native_messages(messages),
            'stream':     bool(stream),
            'keep_alive': keep_alive,
            'options':    opts,
        }
        req = urllib.request.Request(
            f"{self._host}/api/chat",
            data=json.dumps(body).encode(),
            method='POST',
            headers={'Content-Type': 'application/json'},
        )
        resp = urllib.request.urlopen(req, timeout=timeout)

        if stream:
            return _ollama_stream_iter(resp)
        try:
            data    = json.loads(resp.read())
            content = (data.get('message') or {}).get('content', '') or ''
            finish  = 'stop' if data.get('done') else 'length'
            return _OllamaResponse(content, finish_reason=finish)
        finally:
            resp.close()


def _ollama_stream_iter(resp):
    """Yield _OllamaStreamChunk objects for each NDJSON line until done."""
    try:
        for raw in resp:
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = (data.get('message') or {}).get('content', '') or ''
            done    = bool(data.get('done'))
            if content or done:
                yield _OllamaStreamChunk(
                    content=content,
                    finish_reason='stop' if done else None,
                )
            if done:
                break
    finally:
        try: resp.close()
        except Exception: pass


class _OllamaChatAPI:
    def __init__(self, host: str):
        self.completions = _OllamaCompletionsAPI(host)


class OllamaNativeClient:
    """Drop-in replacement for openai.OpenAI used against Ollama.

    Implements only `client.chat.completions.create(...)`. Routes calls to
    the native /api/chat endpoint so that `options` and `keep_alive` survive
    — the OpenAI-compat /v1/chat/completions endpoint silently drops them.
    """
    def __init__(self, host: str):
        self.chat = _OllamaChatAPI(host)


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaBackend(Backend):
    """
    Ollama local inference server (https://ollama.com).
    Uses Ollama's native REST API for everything — model lifecycle AND
    inference. Default port: 11434.

    Inference goes through /api/chat (not the OpenAI-compatible /v1/) because
    Ollama's compat layer silently drops `options` and `keep_alive` from the
    request body, undoing the load-time num_ctx / num_gpu / keep_alive=-1 on
    every call. The native client wrapper (OllamaNativeClient) keeps the same
    `client.chat.completions.create(...)` API surface as the OpenAI SDK so
    consumer code is unchanged.

    Key behaviours vs LMStudio:
    - load()  is synchronous: it blocks until the model is in VRAM.
    - unload() uses keep_alive=0 on /api/generate.
    - Context size is read from POST /api/show (num_ctx parameter).
    - Model IDs use name:tag format (e.g. 'llama3.2:latest').
    """

    ENGINE       = 'ollama'
    DEFAULT_HOST = 'http://localhost:11434'
    _API_KEY     = 'ollama'

    def __init__(self, host: str):
        super().__init__(host)
        self._active_options: dict = {}   # options used at last load()

    def create_client(self):
        """Override: native /api/chat client to keep options + keep_alive sticky."""
        return OllamaNativeClient(self.host)

    def get_inference_extra_body(self) -> dict:
        """Return Ollama options + keep_alive to re-use in every inference call.

        Without `options`: Ollama compares the inference request's options (empty
        = defaults, e.g. num_ctx=262144 or whatever the modelfile bakes in)
        against the loaded runner's options and reloads the model from scratch,
        wiping out num_ctx and num_gpu settings.

        Without `keep_alive`: Ollama applies its default 5 min keep_alive to the
        runner on every request. The first inference flips the runner from our
        load()-time `keep_alive=-1` (Forever) to a 5 min timer; subsequent gaps
        between requests can then evict the model and trigger a fresh load with
        modelfile defaults. Sending `keep_alive=-1` on every call pins it.
        """
        if not self._active_options:
            return {}
        return {
            'options':    dict(self._active_options),
            'keep_alive': -1,
        }

    def list_available(self) -> list:
        """GET /api/tags — list installed models."""
        try:
            data = self._get('/api/tags')
            return [{'id': m['name'], 'ctx': None} for m in data.get('models', [])]
        except Exception:
            return []

    def get_loaded(self) -> 'str | None':
        """GET /api/ps — list models currently held in VRAM."""
        try:
            data = self._get('/api/ps')
            models = data.get('models', [])
            if models:
                return models[0]['name']
        except Exception:
            pass
        return None

    def get_loaded_ctx(self) -> 'int | None':
        """Return the context size of the running model, or None if unknown.

        Sources checked in priority order:
        1. _active_options (set by this process when it called load())
        2. /api/ps response (Ollama exposes num_ctx / context_length in newer versions)
        """
        if self._active_options.get('num_ctx'):
            return self._active_options['num_ctx']
        try:
            data = self._get('/api/ps')
            for m in data.get('models', []):
                ctx = (m.get('context_length') or m.get('num_ctx')
                       or m.get('details', {}).get('context_length'))
                if ctx:
                    return int(ctx)
        except Exception:
            pass
        return None

    def get_loaded_num_gpu(self) -> 'int | None':
        """Return the num_gpu of the running model as tracked by this process."""
        return self._active_options.get('num_gpu')

    def get_ctx(self, model_id: str) -> 'int | None':
        """
        POST /api/show — retrieve model details.
        Checks `parameters` string for user-configured num_ctx, then
        `model_info` for architecture-default context_length.
        """
        try:
            data = self._post('/api/show', {'name': model_id})
            # User-configured num_ctx (highest priority)
            for line in data.get('parameters', '').splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0].lower() == 'num_ctx':
                    return int(parts[1])
            # Architecture default from model_info dict
            for key, val in data.get('model_info', {}).items():
                if 'context_length' in key.lower():
                    return int(val)
        except Exception:
            pass
        return None

    def load(self, model_id: str, ctx: int, num_gpu: 'int | None' = None) -> bool:
        """
        Warm the model into VRAM via POST /api/generate with keep_alive=-1.
        Ollama processes this synchronously — the call blocks until ready.

        options.num_ctx sets the KV-cache / context window size.
        options.num_gpu sets how many transformer layers to offload to GPU
        (e.g. 7 for a 26 B model that fills ~1.6 GB VRAM per layer).
        If num_gpu is None, Ollama decides automatically.
        """
        try:
            options: dict = {'num_ctx': ctx}
            if num_gpu is not None:
                options['num_gpu'] = num_gpu
            self._active_options = options   # save for inference calls
            body = json.dumps({
                'model':      model_id,
                'prompt':     '',
                'keep_alive': -1,         # -1 = keep loaded indefinitely
                'options':    options,
            }).encode()
            req = urllib.request.Request(
                f"{self.host}/api/generate", data=body, method='POST',
                headers={'Content-Type': 'application/json'},
            )
            with urllib.request.urlopen(req, timeout=_LOAD_TIMEOUT) as resp:
                resp.read()   # consume NDJSON response stream
            return True
        except Exception:
            return False

    def unload(self, model_id: str) -> bool:
        """Evict model from VRAM via keep_alive=0."""
        try:
            self._post('/api/generate', {
                'model':      model_id,
                'prompt':     '',
                'keep_alive': 0,
            })
            self._active_options = {}
            return True
        except Exception:
            return False

    def wait_ready(self, model_id: str, timeout: int = _LOAD_TIMEOUT) -> bool:
        """
        For Ollama, load() is synchronous — the model is already in /api/ps
        when load() returns. This polls briefly as a safety confirmation.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                data = self._get('/api/ps')
                for m in data.get('models', []):
                    if m['name'] == model_id:
                        return True
            except Exception:
                pass
            time.sleep(1)
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_backend(engine: str, host: str) -> Backend:
    """
    Instantiate the appropriate backend for `engine` pointed at `host`.

    Args:
        engine: 'lmstudio' or 'ollama'
        host:   Resolved API base URL (pass through resolve_host() first).

    Returns:
        A Backend subclass instance ready to use.
    """
    engine = (engine or 'lmstudio').lower().strip()
    if engine == 'ollama':
        return OllamaBackend(host)
    if engine in ('lmstudio', 'lm-studio', 'lm_studio'):
        return LMStudioBackend(host)
    raise ValueError(
        f"Unknown AI engine: {engine!r}. Valid values: 'lmstudio', 'ollama'."
    )
