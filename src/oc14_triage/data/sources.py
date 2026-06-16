"""Registry of the source datasets, with provenance, licence and intended role.

Verified facts come from docs/research/04-oc14-dataset-construction-recipe.md; the
actual schema/row-count is re-confirmed at download time and written to
data/raw/_inventory.json (the Day-1 load-smoke from the plan).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str  # local slug → data/raw/<name>/
    hf_id: str
    language: str  # "fr" | "en"
    license: str
    role: str  # "sft_fr" | "sft_en" | "dpo" | "triage_fr"
    url: str
    configs: tuple[str, ...] = ()  # HF config names ("" / () means default config)
    splits: tuple[str, ...] = ()  # () means auto-discover available splits
    notes: str = ""
    enabled: bool = True  # disabled sources are recorded for provenance but not built


# Order matters only for readability. Roles drive the build_* scripts.
SOURCES: list[Source] = [
    Source(
        name="mediqal",
        hf_id="ANR-MALADES/MediQAl",
        language="fr",
        license="CC-BY-4.0",
        role="sft_fr",
        url="https://huggingface.co/datasets/ANR-MALADES/MediQAl",
        configs=("mcqu", "mcqm", "oeq"),  # config names are LOWERCASE on the Hub
        splits=(),  # auto-discover per config (oeq is test-only)
        notes="French ECN exam QA. oeq=open-ended (best SFT signal); mcqu=single-answer; "
        "rows with medical_subject=='Urgences' seed the FR triage slice.",
    ),
    Source(
        name="frenchmedmcqa",
        hf_id="qanastek/LLaMaInstructionsFrenchMedMCQA",
        language="fr",
        license="Apache-2.0",
        role="sft_fr",
        url="https://huggingface.co/datasets/qanastek/LLaMaInstructionsFrenchMedMCQA",
        splits=(),  # auto-discover
        enabled=False,  # both qanastek variants ship a loader script modern `datasets` rejects
        notes="DISABLED: French pharmacy DES exam MCQA. Both qanastek variants ship a Python "
        "loader script that datasets>=3 refuses; MediQAl alone covers French amply, so we skip "
        "this rather than pin a fragile old datasets. Pharmacology-vocab coverage is the only loss.",
    ),
    Source(
        name="medquad",
        hf_id="keivalya/MedQuad-MedicalQnADataset",
        language="en",
        license="CC-BY-4.0 (per original Ben Abacha 2019; HF card omits it)",
        role="sft_en",
        url="https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset",
        splits=("train",),
        notes="NIH patient-education QA. Filter qtype in {symptoms, treatment, exams and "
        "tests, complications, prevention}. Fallback: lavita/MedQuAD (explicit card).",
    ),
    Source(
        name="ultramedical_pref",
        hf_id="TsinghuaC3I/UltraMedical-Preference",
        language="en",
        license="MIT",
        role="dpo",
        url="https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference",
        splits=("train", "validation"),  # NB: 'test' has a schema mismatch — do not load
        notes="GPT-4-scored medical preference pairs (prompt/chosen/rejected). English. "
        "Framed as DPO technique demo, not a clinical-quality signal.",
    ),
]

SOURCES_BY_NAME = {s.name: s for s in SOURCES}


def by_role(role: str) -> list[Source]:
    return [s for s in SOURCES if s.role == role and s.enabled]
