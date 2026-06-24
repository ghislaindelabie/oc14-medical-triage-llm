# OC14 — Triage criteria (the medical core to review)

> **Status:** intermediary, 2026-06-24. **Review this first** — the entire triage dataset (training +
> eval gold) rests on these criteria. Canonical source = `src/oc14_triage/labeling/rubric.py`.

## What this is
The rubric three frontier LLMs (OpenAI + Mistral + Anthropic) apply to **real MediQAl French clinical
vignettes** to assign a triage urgency. Each model returns a **3-level** label *and* its **ESI 1–5**
equivalent in the same call (free cross-check + comparability to ESI datasets). We keep the **consensus**.

## The 3-level scale (with ESI mapping)
The 5-level boundaries are collapsed **once, here, transparently** (standard monotonic mapping):

| Level | ESI | Meaning | Red-flag examples (any one ⇒ this level) |
|---|---|---|---|
| **urgence maximale** | 1–2 | life/function threat, cannot wait | cardiac arrest/unconscious; resp. distress / SpO₂<90%; chest pain (ACS); stroke signs (FAST); severe haemorrhage; anaphylaxis; sepsis/shock (SBP<90); GCS<13; active seizure; major trauma; suicidal w/ plan; critical vitals (HR>130/<40, RR>30) |
| **urgence modérée** | 3 | symptomatic, assess promptly, stable | febrile focal infection; moderate pain (abdo w/o guarding); mild-moderate dehydration; suspected fracture able to bear weight; stable chronic decompensation |
| **urgence différée** | 4–5 | non-urgent / ambulatory / administrative | minor sore throat; minor sprain walking; prescription renewal; medical certificate; prevention advice; stable chronic follow-up; information request |

**Decision order:** (a) any red flag → maximale; (b) else acute, needs prompt care → modérée; (c) else
minor/administrative → différée. **When in doubt, over-triage** (choose the more urgent level). Non-clinical
text (exam/mechanism questions) is flagged `is_triage_case=false` and excluded.

## Sources (distilled; verify exact editions before publishing)
- **ESI** — Emergency Severity Index, AHRQ handbook (Gilboy et al.): 5-level, the 4-decision algorithm + vital danger zones.
- **MTS** — Manchester Triage System (Mackway-Jones et al.): presentational red-flag discriminators.
- **FRENCH** — French Emergency Nurses Classification in Hospital (Taboulet et al., SFMU): FR 5-level scale.
- **CIMU** — Classification Infirmière des Malades aux Urgences (Fourestié et al.): FR 5-level scale.
Red flags are universal across these; the French scales fix the taxonomy. (CCMU is a *physician severity*
grade, not front-door nurse triage, so it's not used as the taxonomy.)

## How labels are made trustworthy (without a clinician)
- **3-model consensus.** Keep **unanimous + ESI-consistent** cases as **gold** (held-out eval); majority cases → training; report **Fleiss' κ** (inter-model agreement) as the credibility metric.
- **Self-consistency check.** Each answer's ESI must bucket to its stated 3-level; mismatches are flagged.
- **Labeler calibration (floor check).** Each model first answers ~200 **real MediQAl MCQU** exam questions vs the **real answer key** → a measured French-medical-competence number. *Necessary, not sufficient* (knowledge ≠ triage), but a poor score disqualifies a labeler.
- **Cross-method triangulation.** Compare the consensus to `medical-triage-500`'s independent rule-based labels — agreement across *independent* methods is the closest thing to validity we have.

## Honest limitations (state in the report)
- This is a **silver standard**, not clinical validation — no clinician, no validated FR triage gold exists.
- LLM triage agreement with clinicians is only **moderate** in the literature (κ≈0.47 at 5-level; higher at 3-level); we mitigate via consensus + clear-case gold + over-triage default.
- Source cases are real French **exam** vignettes (good provenance) but exam-style, not raw ED notes.
- The deployed model has **no RAG** by design; the grounded LLMs are *teachers*, the small model is the *student* (distillation).
