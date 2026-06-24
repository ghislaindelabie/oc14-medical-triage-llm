"""Thin, uniform wrappers over OpenAI / Mistral / Anthropic + a MockClient for tests.

Each client: a `.name` and `.complete(system, user) -> str`. SDKs are imported lazily inside
`complete()`, so importing this module (and using MockClient) needs no SDKs and no keys.
Model ids are env-overridable; defaults are mid-tier-strong (cheap). Bump to frontier via env
for the eval-gold pass. The orchestrator uses whichever providers have a key set.
"""

from __future__ import annotations

import os

# Default models (mid-tier-strong = cheap; override with env for a frontier eval-gold pass).
OPENAI_MODEL = os.environ.get("OC14_OPENAI_MODEL", "gpt-4o-mini")
MISTRAL_MODEL = os.environ.get("OC14_MISTRAL_MODEL", "mistral-small-latest")
ANTHROPIC_MODEL = os.environ.get("OC14_ANTHROPIC_MODEL", "claude-3-5-haiku-latest")


class OpenAIClient:
    name = "openai"

    def __init__(self, model: str = OPENAI_MODEL):
        self.model = model

    def complete(self, system: str, user: str) -> str:
        from openai import OpenAI  # lazy
        client = OpenAI()  # reads OPENAI_API_KEY
        r = client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return r.choices[0].message.content or ""


class MistralClient:
    name = "mistral"

    def __init__(self, model: str = MISTRAL_MODEL):
        self.model = model

    def complete(self, system: str, user: str) -> str:
        from mistralai import Mistral  # lazy
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        r = client.chat.complete(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return r.choices[0].message.content or ""


class AnthropicClient:
    name = "anthropic"

    def __init__(self, model: str = ANTHROPIC_MODEL):
        self.model = model

    def complete(self, system: str, user: str) -> str:
        import anthropic  # lazy
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        r = client.messages.create(
            model=self.model, max_tokens=512, temperature=0,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(getattr(b, "text", "") for b in r.content)


class MockClient:
    """Returns canned answers — for unit tests and key-free dry-runs."""

    def __init__(self, name: str, answer: str | list[str]):
        self.name = name
        self._answers = answer if isinstance(answer, list) else None
        self._fixed = answer if isinstance(answer, str) else None
        self._i = 0

    def complete(self, system: str, user: str) -> str:
        if self._fixed is not None:
            return self._fixed
        a = self._answers[self._i % len(self._answers)]
        self._i += 1
        return a


_KEY_ENV = {"openai": "OPENAI_API_KEY", "mistral": "MISTRAL_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
_CTORS = {"openai": OpenAIClient, "mistral": MistralClient, "anthropic": AnthropicClient}


def available_clients() -> list:
    """Real clients for every provider whose API key is present in the environment."""
    return [_CTORS[name]() for name, env in _KEY_ENV.items() if os.environ.get(env)]


def missing_keys() -> list[str]:
    return [env for env in _KEY_ENV.values() if not os.environ.get(env)]
