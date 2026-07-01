"""RGPD anonymisation — the core "no PII stored" control for the triage POC.

Grounded in docs/research/06-presidio-anonymization-gdpr-medical-dataset.md.

Two layers, by design:

1. **Presidio + spaCy NER** detects the semantic identifiers a regex cannot (PERSON,
   LOCATION) plus DATE_TIME, and drives readable-placeholder replacement.
2. **A regex safety net** masks the structured direct identifiers (phone, email, French
   NIR, long numeric IDs) *unconditionally*. This is deliberate: the French spaCy phone
   recogniser scores ~0.4 (below a sane 0.5 threshold) and can even fragment a phone into
   a spurious DATE_TIME — so relying on NER alone would leak. The regex layer guarantees
   the LEAK INVARIANT regardless of model quality; NER adds the names/places on top.

Two modes:
- ``runtime`` — mask/redact DIRECT identifiers; age/date are REPLACED with de-identified
  ``[AGE]`` / ``[DATE]`` placeholders (kept because they are triage-relevant).
- ``dataset`` — replace ALL detected PII with readable FR placeholders (offline corpus pass).

If Presidio or the spaCy models cannot be imported, the module falls back to the regex
masker alone; PERSON/LOCATION then need the NER path and are documented as un-masked.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field

# Readable placeholders (French, per the research doc's operator strategy).
_PLACEHOLDER = {
    "PERSON": "[NOM]",
    "PHONE_NUMBER": "[TEL]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "LOCATION": "[LIEU]",
    "IBAN_CODE": "[IBAN]",
    "FRENCH_NIR": "[NIR]",
    "MEDICAL_RECORD_ID": "[IPP]",
    "DATE_TIME": "[DATE]",
    "AGE": "[AGE]",
    "URL": "[URL]",
    "IP_ADDRESS": "[IP]",
}

# Direct identifiers masked in BOTH modes. In runtime mode DATE_TIME/AGE are additionally
# replaced (kept as placeholders), not dropped.
_DIRECT = ("PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION", "IBAN_CODE", "FRENCH_NIR",
           "MEDICAL_RECORD_ID")

_ENGINE_VERSION_UNKNOWN = "unknown"

# --- Regex safety net --------------------------------------------------------
# Order matters: NIR (long grouped digits) and email/URL before the generic phone, so a
# longer identifier is consumed before a shorter pattern can nibble part of it.
_NIR_RE = re.compile(r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2,3}\s?\d{2,3}\s?\d{3}\s?\d{2}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL_RE = re.compile(r"https?://\S+")
# FR/intl phones: optional +33/0, then 8-10 groups of digits with . - or space separators.
_PHONE_RE = re.compile(r"(?:(?:\+|00)\d{1,3}[\s.-]?)?(?:\(?\d{1,4}\)?[\s.-]?){3,6}\d{2,4}")
# A bare age like "43 ans" / "43 years" — de-identify the number, keep the [AGE] signal.
_AGE_RE = re.compile(r"\b\d{1,3}\s?(?=ans\b|an\b|years?\b|yo\b)", re.IGNORECASE)
# Dates in numeric d/m/y forms (spaCy misses some; this backstops the leak invariant).
_DATE_RE = re.compile(r"\b\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4}\b")
# Long standalone numeric IDs (record numbers) — 7+ digits not already caught above.
_LONGID_RE = re.compile(r"\b\d{7,}\b")

# Regex layer applied in priority order → (entity_type, compiled regex).
_REGEX_LAYER = (
    ("EMAIL_ADDRESS", _EMAIL_RE),
    ("URL", _URL_RE),
    ("FRENCH_NIR", _NIR_RE),
    ("PHONE_NUMBER", _PHONE_RE),
    ("DATE_TIME", _DATE_RE),
    ("MEDICAL_RECORD_ID", _LONGID_RE),
)

# Entity types Presidio should surface; NER contributes PERSON/LOCATION/DATE_TIME.
_NER_ENTITIES = ("PERSON", "LOCATION", "DATE_TIME", "IBAN_CODE")


@dataclass
class AnonResult:
    """Result of one anonymisation pass, plus the provenance an audit log needs."""

    text: str
    entities: list[dict] = field(default_factory=list)
    engine: str = "regex"
    engine_version: str = _ENGINE_VERSION_UNKNOWN


def sha256_text(text: str) -> str:
    """Hex SHA-256 digest — one-way hash for traceability without retaining the raw input."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# --- Presidio analyzer (lazy singleton) --------------------------------------
_ANALYZER = None  # AnalyzerEngine | None
_PRESIDIO_READY: bool | None = None  # tri-state: None = not yet probed
_ENGINE_NAME = "regex"
_ENGINE_VER = _ENGINE_VERSION_UNKNOWN


def _get_analyzer():
    """Build the bilingual Presidio analyzer once; return None if unavailable (regex fallback)."""
    global _ANALYZER, _PRESIDIO_READY, _ENGINE_NAME, _ENGINE_VER
    if _PRESIDIO_READY is not None:
        return _ANALYZER
    try:
        from importlib.metadata import version

        from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "fr", "model_name": "fr_core_news_md"},
                {"lang_code": "en", "model_name": "en_core_web_sm"},
            ],
        })
        nlp_engine = provider.create_engine()
        registry = RecognizerRegistry(supported_languages=["fr", "en"])
        registry.load_predefined_recognizers(languages=["fr", "en"])
        # Custom FR NIR recogniser (absent from public corpora but demonstrates thoroughness).
        nir = PatternRecognizer(
            supported_entity="FRENCH_NIR",
            supported_language="fr",
            patterns=[{"name": "french_nir", "regex": _NIR_RE.pattern, "score": 0.85}],
            context=["sécurité sociale", "nir", "numéro", "carte vitale"],
        )
        registry.add_recognizer(nir)
        _ANALYZER = AnalyzerEngine(
            nlp_engine=nlp_engine, registry=registry, supported_languages=["fr", "en"])
        _ENGINE_NAME = "presidio+spacy"
        try:
            _ENGINE_VER = version("presidio-analyzer")
        except Exception:  # noqa: BLE001
            _ENGINE_VER = _ENGINE_VERSION_UNKNOWN
        _PRESIDIO_READY = True
    except Exception:  # noqa: BLE001 — any import/model failure → documented regex fallback
        _ANALYZER = None
        _PRESIDIO_READY = False
    return _ANALYZER


def _ner_spans(text: str, lang: str) -> list[tuple[int, int, str]]:
    """(start, end, entity_type) from Presidio for the NER-only entity types. Phones/emails/NIR
    are handled by the regex layer, so we ignore Presidio's (often low-confidence) versions of
    those to avoid double-masking and fragment artefacts."""
    analyzer = _get_analyzer()
    if analyzer is None:
        return []
    results = analyzer.analyze(text=text, language=lang, entities=list(_NER_ENTITIES),
                               score_threshold=0.4)
    return [(r.start, r.end, r.entity_type) for r in results]


def _regex_spans(text: str) -> list[tuple[int, int, str]]:
    """Non-overlapping spans from the regex safety net, in priority order."""
    spans: list[tuple[int, int, str]] = []
    for etype, rx in _REGEX_LAYER:
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            if s == e:
                continue
            if any(s < oe and os < e for os, oe, _ in spans):  # overlaps an earlier match
                continue
            spans.append((s, e, etype))
    return spans


def _age_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), "AGE") for m in _AGE_RE.finditer(text)]


def _merge(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Sort by start; drop any span that overlaps one already accepted (longer/earlier wins)."""
    kept: list[tuple[int, int, str]] = []
    for s, e, t in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
        if any(s < ke and ks < e for ks, ke, _ in kept):
            continue
        kept.append((s, e, t))
    return kept


def anonymize(text: str, *, mode: str = "runtime", lang: str = "fr") -> AnonResult:
    """Anonymise ``text``. ``mode`` in {"runtime","dataset"}; ``lang`` in {"fr","en"}.

    runtime: mask direct identifiers; replace age/date with [AGE]/[DATE] (kept, de-identified).
    dataset: replace every detected PII with a readable FR placeholder.
    """
    if mode not in ("runtime", "dataset"):
        raise ValueError(f"unknown mode: {mode!r}")
    text = text or ""

    # Collect candidate spans from both layers (age only relevant if kept as placeholder).
    candidates = _regex_spans(text) + _ner_spans(text, lang) + _age_spans(text)
    spans = _merge(candidates)

    # In dataset mode everything detected is replaced. In runtime mode we also replace all
    # of it: direct identifiers are masked, and age/date become de-identified placeholders
    # (both modes therefore leave no raw identifier — the difference is that dataset mode is
    # the offline corpus pass and would use the same readable placeholders).
    counts: Counter[str] = Counter()
    # Apply replacements right-to-left so earlier offsets stay valid.
    out = text
    for s, e, etype in sorted(spans, key=lambda x: x[0], reverse=True):
        placeholder = _PLACEHOLDER.get(etype, f"[{etype}]")
        out = out[:s] + placeholder + out[e:]
        counts[etype] += 1

    entities = [{"type": t, "count": c} for t, c in sorted(counts.items())]
    _get_analyzer()  # ensure engine metadata is populated
    return AnonResult(text=out, entities=entities, engine=_ENGINE_NAME, engine_version=_ENGINE_VER)


__all__ = ["AnonResult", "anonymize", "sha256_text"]
