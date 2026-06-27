"""Thin, uniform wrappers over OpenAI / Mistral / Anthropic + a MockClient for tests.

Each client exposes `.name`, `.complete(system, user) -> str`, and token-usage counters
(`.in_tok`, `.out_tok`, `.calls`) accumulated thread-safely so a run can report real cost.
SDKs are imported lazily inside `complete()`, so importing this module (and using MockClient)
needs no SDKs and no keys. Model ids are env-overridable via OC14_<PROVIDER>_MODEL.
"""

from __future__ import annotations

import os
import threading

_DEFAULT_MODEL = {"openai": "gpt-4o-mini", "mistral": "mistral-small-latest",
                  "anthropic": "claude-3-5-haiku-latest"}


class _Usage:
    """Mixin: thread-safe, cache-aware token accounting shared by the real provider clients.

    in_full = uncached input (full price); in_cread = cache reads (~0.1x);
    in_cwrite = cache writes (~1.25x, Anthropic only); out_tok = output.
    """

    def _init_usage(self) -> None:
        self.in_full = self.in_cread = self.in_cwrite = self.out_tok = self.calls = 0
        self._lock = threading.Lock()

    def _add_usage(self, in_full, out, cread=0, cwrite=0) -> None:
        with self._lock:
            self.in_full += int(in_full or 0)
            self.in_cread += int(cread or 0)
            self.in_cwrite += int(cwrite or 0)
            self.out_tok += int(out or 0)
            self.calls += 1

    @property
    def in_tok(self) -> int:  # total input across all tiers (for display)
        return self.in_full + self.in_cread + self.in_cwrite


class OpenAIClient(_Usage):
    name = "openai"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("OC14_OPENAI_MODEL", _DEFAULT_MODEL["openai"])
        self._init_usage()

    def complete(self, system: str, user: str) -> str:
        from openai import OpenAI  # lazy
        r = OpenAI().chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        u = r.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        self._add_usage((getattr(u, "prompt_tokens", 0) or 0) - cached,
                        getattr(u, "completion_tokens", 0), cread=cached)
        return r.choices[0].message.content or ""


class MistralClient(_Usage):
    """Mistral via its REST endpoint (stdlib urllib) — the `mistralai` SDK (2.5.0) imports as an
    empty namespace package, so we skip it. The API is OpenAI-compatible."""

    name = "mistral"
    URL = "https://api.mistral.ai/v1/chat/completions"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("OC14_MISTRAL_MODEL", _DEFAULT_MODEL["mistral"])
        self._init_usage()

    def complete(self, system: str, user: str) -> str:
        import json
        import time
        import urllib.error
        import urllib.request
        body = json.dumps({
            "model": self.model, "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }).encode("utf-8")
        last: Exception | None = None
        for attempt in range(4):  # urllib has no retry of its own; back off on 429/5xx
            req = urllib.request.Request(self.URL, data=body, headers={
                "Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}",
                "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    d = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 500, 502, 503, 529) and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise
        else:  # pragma: no cover — loop always breaks or raises
            raise last  # type: ignore[misc]
        u = d.get("usage", {})
        self._add_usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        return d["choices"][0]["message"]["content"] or ""


class AnthropicClient(_Usage):
    name = "anthropic"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("OC14_ANTHROPIC_MODEL", _DEFAULT_MODEL["anthropic"])
        self._init_usage()

    def complete(self, system: str, user: str) -> str:
        import anthropic  # lazy
        r = anthropic.Anthropic().messages.create(
            model=self.model, max_tokens=512, temperature=0,
            # cache the (stable) rubric prefix; no-op below the model's min cacheable size.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        u = r.usage
        self._add_usage(
            getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0),
            cread=getattr(u, "cache_read_input_tokens", 0) or 0,
            cwrite=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        return "".join(getattr(b, "text", "") for b in r.content)


class MockClient:
    """Returns canned answers — for unit tests and key-free dry-runs."""

    def __init__(self, name: str, answer: str | list[str]):
        self.name = name
        self._answers = answer if isinstance(answer, list) else None
        self._fixed = answer if isinstance(answer, str) else None
        self._i = 0
        self.in_full = self.in_cread = self.in_cwrite = self.in_tok = self.out_tok = self.calls = 0

    def complete(self, system: str, user: str) -> str:
        if self._fixed is not None:
            return self._fixed
        a = self._answers[self._i % len(self._answers)]
        self._i += 1
        return a


_KEY_ENV = {"openai": "OPENAI_API_KEY", "mistral": "MISTRAL_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
_CTORS = {"openai": OpenAIClient, "mistral": MistralClient, "anthropic": AnthropicClient}

# Sync $/1M tokens (input, output) for the configured frontier models — for cost reporting only.
PRICES = {"openai": (2.50, 15.00), "mistral": (1.50, 7.50), "anthropic": (3.00, 15.00)}


def available_clients() -> list:
    """Real clients for every provider whose API key is present in the environment."""
    return [_CTORS[name]() for name, env in _KEY_ENV.items() if os.environ.get(env)]


def missing_keys() -> list[str]:
    return [env for env in _KEY_ENV.values() if not os.environ.get(env)]
