# OC14 — Metadata schema (dataset + runtime dossier)

_Étape-1 deliverable: "Définir le schéma des métadonnées (symptômes, antécédents, constantes, source, niveau de confiance)." This documents both the **training-data** record shapes and the **runtime dossier** written by the agent — the latter is the same shape the mock SIH record is built from._

## 1. Clinical metadata schema (the five brief-named fields)

| Field (brief) | Key | Meaning | Provenance in this POC |
|---|---|---|---|
| **Symptômes** | `symptoms` | Chief complaint + described symptoms (free-text, anonymised) | Runtime: questionnaire `motif` + red-flag `followup`. Dataset: the MediQAl clinical vignette / case text. |
| **Antécédents** | `medical_history` | Relevant history | Runtime: optional free-text (not a core questionnaire field in the POC — documented limitation). Absent from the exam-derived corpus. |
| **Constantes** | `vitals` | Vital signs (T°, FC, TA, SpO₂…) | **Not collected in the POC** (no device integration) — a scale-up item in the roadmap. Field reserved in the schema. |
| **Source** | `source` | Data / interaction provenance | Dataset: `mediqal_mcqu` · `mediqal_oeq` · `medquad` · `llm_triage` · `vignette`. Runtime: `chsa-triage-poc`. |
| **Niveau de confiance** | `confidence_level` | Label confidence | Dataset: derived from the **3-LLM consensus** — unanimous (3/3) → `high`, majority (2/3) → `medium`. Runtime: n/a (single model). |

Plus, on every record: `anonymization_applied` (bool) + `anonymization_engine` / `_version` — see the RGPD note.

## 2. Training-data record shapes (`data/kaggle_upload/`)

- **SFT** (`sft_train.jsonl`, `sft_val.jsonl`): `{ "messages": [ {role: system}, {role: user}, {role: assistant} ], "lang": "fr"|"en", "kind": "triage"|"qa", "source": <source> }`
- **DPO** (`dpo_train.jsonl`, `dpo_val.jsonl`): `{ "prompt": str, "chosen": str, "rejected": str, "lang": ..., "source": "dpo_under"|"dpo_mod"|"dpo_over"|"safety" }`
- **Eval-gold** (`triage_eval_gold.jsonl`): `{ "case_id": str, "user": str, "gold_esi": int (1–5), "gold_urgency": <3-level> }` — held-out, stratified 100/100/100.

## 3. Runtime dossier schema (persisted per interaction)

Written by the agent's `persistance` node to SQLite (`Store`) and projected into the FHIR-shaped SIH record. **Every free-text field here is post-anonymisation by construction.**

```
session_id (uuid)          interaction_id (uuid)      timestamp_utc
model_version              symptoms_anon              antecedents_anon
constantes                 urgency (3-level)          justification
recommandation_anon        source = "chsa-triage-poc" confidence_level
input_sha256               disclaimer_present         latency_ms          deleted (erasure flag)
```

## 4. RGPD note (traceability without retention)

- Direct identifiers (name, DOB, phone, email, address, NIR) are **removed by Presidio + a regex safety net** before anything is stored — see `anonymization_audit.json` for the measured findings across the corpus.
- The raw input is **never persisted or logged**; it survives only as a one-way `input_sha256`, so an interaction can be audited without retaining personal data — *"hash for traceability, anonymise for storage."*
- `session_id` / `interaction_id` are synthetic UUIDs (not derived from patient attributes), so they are not personal data.
