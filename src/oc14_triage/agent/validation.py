"""Deterministic input-sanity guardrail — the rules half of the hybrid LLM+rules design.

Symmetric to the red-flag safety override: where that turns a detected danger cue into an
escalation ("catch the emergency the model missed"), this turns CLEAR gibberish into an
HONEST refusal ("don't confabulate a verdict from noise"). It fixes an observed bug where
"ddsd dsfdsx dfd dsfd" produced a confident *urgence maximale* with hallucinated findings.

Design principles (in order of importance):
  1. CONSERVATIVE — err toward LETTING input THROUGH to the model / clinician. Only clear
     gibberish is rejected; anything with real linguistic content passes.
  2. Never reject short but real symptoms ("fièvre", "toux", "mal au ventre").
  3. SIMPLE + EXPLAINABLE — a handful of pure, thresholded heuristics, no ML, returns a
     human-readable reason so the refusal notice can be audited.

The primary signal is the fraction of alphabetic tokens that look like real words (contain
a vowel, plausible length). On the calibration data this separates cleanly: the gibberish
bug scores 0.0 while every real case — 300 gold clinical vignettes and terse one-word
symptoms alike — scores >= 0.58.
"""

from __future__ import annotations

import re

# Vowels across FR/EN (accented + y). A token with no vowel at all is almost never a word.
_VOWELS = frozenset("aeiouyàâäéèêëîïôöùûüÿœæ")
# Alphabetic tokens only (drops digits/punctuation); Unicode-aware so accents survive.
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)

# A text passes when at least this fraction of its alphabetic tokens are word-like. Chosen
# well below the real-input minimum (~0.58 on gold + terse symptoms) and well above the
# gibberish score (0.0) — a wide, conservative margin that favours letting input through.
_MIN_WORDLIKE_FRACTION = 0.4
# A single alphabetic token this long with no vowel (e.g. "zzzzzzzzzz") is keyboard-mash.
_MAX_CONSONANT_RUN = 5


def _is_wordlike(token: str) -> bool:
    """A token looks like a real word: at least 2 chars and containing at least one vowel."""
    return len(token) >= 2 and any(c in _VOWELS for c in token)


def _longest_consonant_run(token: str) -> int:
    run = best = 0
    for c in token:
        if c in _VOWELS:
            run = 0
        else:
            run += 1
            best = max(best, run)
    return best


def intelligibility_reason(text: str, lang: str = "fr") -> str | None:
    """Return a short human-readable reason the input is unintelligible, or None if it is fine.

    Conservative: only CLEAR gibberish yields a reason. Anything carrying real linguistic
    content (including terse symptoms and real phrases buried in noise) returns None.
    """
    stripped = (text or "").strip()
    if not stripped:
        return "entrée vide"

    tokens = _TOKEN.findall(stripped.lower())
    if not tokens:
        # No alphabetic content at all (pure punctuation/digits) — nothing to triage on.
        return "aucun contenu textuel exploitable"

    wordlike = [t for t in tokens if _is_wordlike(t)]
    fraction = len(wordlike) / len(tokens)
    if fraction < _MIN_WORDLIKE_FRACTION:
        return "trop peu de mots intelligibles"

    # Single-token inputs: guard the terse-symptom case ("fièvre") against keyboard-mash.
    # A lone token is not a real word when it is one character repeated ("aaaa", "zzzz") or
    # carries a very long consonant run ("zzzzzzzzzz"). Real symptoms use >= 2 distinct letters
    # and stay well under the run threshold, so this never rejects "fièvre" / "toux".
    if len(tokens) == 1:
        only = tokens[0]
        if len(set(only)) == 1 and len(only) >= 3:
            return "caractère répété — pas un mot"
        if _longest_consonant_run(only) > _MAX_CONSONANT_RUN:
            return "suite de consonnes non prononçable"

    return None


def is_intelligible(text: str, lang: str = "fr") -> bool:
    """True if the input carries enough linguistic content to triage on (conservative)."""
    return intelligibility_reason(text, lang) is None
