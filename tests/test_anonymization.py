"""Tests for the RGPD anonymisation module — the core "no PII stored" control.

The load-bearing assertions are LEAK INVARIANTS: after anonymisation the output must not
contain the original direct identifiers, while triage-relevant clinical content survives.
These are written test-first (strict TDD). The Presidio+spaCy NER path is required for the
PERSON/LOCATION detections; if the models cannot be imported the NER-dependent tests skip
with a clear reason and the regex-guaranteed masking (phone/email/date/long-id) still runs.
"""

from __future__ import annotations

import importlib.util

import pytest

from oc14_triage.anonymization import AnonResult, anonymize, sha256_text

# NER-backed detection (PERSON / LOCATION) needs the spaCy models. Regex masking does not.
_HAS_NER = (
    importlib.util.find_spec("fr_core_news_md") is not None
    and importlib.util.find_spec("en_core_web_sm") is not None
)
_NER_REASON = "spaCy models (fr_core_news_md / en_core_web_sm) unavailable — NER path skipped"


def test_sha256_deterministic_and_input_sensitive() -> None:
    assert sha256_text("douleur thoracique") == sha256_text("douleur thoracique")
    assert sha256_text("a") != sha256_text("b")
    # hex digest shape
    digest = sha256_text("x")
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_returns_anonresult_shape() -> None:
    res = anonymize("appel du 06.12.34.56.78", mode="runtime", lang="fr")
    assert isinstance(res, AnonResult)
    assert isinstance(res.text, str)
    assert isinstance(res.entities, list)
    assert res.engine and res.engine_version  # non-empty provenance for the audit log


def test_phone_masked_even_without_ner_fr() -> None:
    """Phone masking must be guaranteed by regex, independent of the NER model."""
    res = anonymize("Rappeler le patient au 06.12.34.56.78 svp", mode="runtime", lang="fr")
    assert "06.12.34.56.78" not in res.text
    assert "[TEL]" in res.text


def test_email_masked_even_without_ner() -> None:
    res = anonymize("contact pierre.dupont@example.fr pour le suivi", mode="runtime", lang="fr")
    assert "pierre.dupont@example.fr" not in res.text
    assert "[EMAIL]" in res.text


@pytest.mark.skipif(not _HAS_NER, reason=_NER_REASON)
def test_leak_invariant_fr() -> None:
    """CORE RGPD control: direct identifiers gone, clinical signal kept."""
    text = "Jean Dupont, né le 3/2/1980, tél 06.12.34.56.78 — douleur thoracique aiguë"
    res = anonymize(text, mode="runtime", lang="fr")
    assert "Jean Dupont" not in res.text
    assert "1980" not in res.text
    assert "06.12.34.56.78" not in res.text
    assert "douleur thoracique" in res.text  # triage-relevant content survives


@pytest.mark.skipif(not _HAS_NER, reason=_NER_REASON)
def test_entities_report_person_and_phone_fr() -> None:
    text = "Jean Dupont, né le 3/2/1980, tél 06.12.34.56.78 — douleur thoracique aiguë"
    res = anonymize(text, mode="runtime", lang="fr")
    types = {e["type"] for e in res.entities}
    assert "PERSON" in types
    assert "PHONE_NUMBER" in types
    # each entity carries a count for the audit log
    assert all("count" in e and e["count"] >= 1 for e in res.entities)


@pytest.mark.skipif(not _HAS_NER, reason=_NER_REASON)
def test_leak_invariant_en() -> None:
    text = "John Smith, phone 555-123-4567, chest pain"
    res = anonymize(text, mode="runtime", lang="en")
    assert "John Smith" not in res.text
    assert "555-123-4567" not in res.text
    assert "chest pain" in res.text


def test_runtime_keeps_age_and_date_as_placeholders_fr() -> None:
    """runtime mode de-identifies age/date but KEEPS a triage-relevant placeholder."""
    res = anonymize("patient de 43 ans, admis le 14/03/1978", mode="runtime", lang="fr")
    assert "14/03/1978" not in res.text
    assert "[DATE]" in res.text or "[AGE]" in res.text


@pytest.mark.skipif(not _HAS_NER, reason=_NER_REASON)
def test_dataset_mode_replaces_all_pii_fr() -> None:
    """dataset mode: every detected PII becomes a readable FR placeholder (no raw value left)."""
    text = "Mme Martin habite à Lyon, tél 06.11.22.33.44"
    res = anonymize(text, mode="dataset", lang="fr")
    assert "Martin" not in res.text
    assert "Lyon" not in res.text
    assert "06.11.22.33.44" not in res.text


def test_french_nir_masked() -> None:
    """FR social-security number (NIR) is highly identifying — must be masked."""
    res = anonymize("NIR 1 85 07 75 108 071 42, douleur abdominale", mode="runtime", lang="fr")
    assert "1 85 07 75 108 071 42" not in res.text
    assert "douleur abdominale" in res.text
