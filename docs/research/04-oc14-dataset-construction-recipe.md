> **TL;DR — key takeaways**
>
> - ANR-MALADES/MediQAl (32 603 FR questions, CC-BY-4.0) and qanastek/frenchmedmcqa (3 105 FR questions, Apache 2.0) are separate datasets; FrenchMedMCQA is NOT a subset of MediQAl — both come from the broader ANR MALADES funding umbrella but are independent releases.
> - keivalya/MedQuad-MedicalQnADataset is a 16 407-row CSV re-packaging of the NIH MedQuAD corpus; the original (Ben Abacha 2019) is CC-BY-4.0 but the HF card omits a license — treat it as CC-BY-4.0 and cite the original paper.
> - TsinghuaC3I/UltraMedical-Preference (MIT, ~110 K rows) gives ready-made DPO triplets (prompt / chosen / rejected) with GPT-4-scored preference labels — the highest-quality off-the-shelf preference set for English medical alignment.
> - There is a genuine triage gap: none of the four named datasets contains urgency-level labels (maximal / moderate / deferred). Bridging it is mandatory for a credible triage deliverable.
> - The recommended recipe reframes the deliverable as a bilingual medical-QA assistant with triage-style guidance, keeps SFT data construction simple (template wrapping + modest synthetic augmentation), and avoids the credentialed MIMIC-IV triage corpus that would add a compliance burden.
> - A target of 5 000 SFT pairs is achievable from the named datasets alone with a straightforward ETL pipeline; the DPO set can be a 3 000-row slice of UltraMedical-Preference without any additional data generation.
> - The 80 FR / 20 EN language mix is realistic given MediQAl + FrenchMedMCQA supply around 4 000 usable FR instruction pairs before any augmentation.


# Dataset Construction for the CHSA Triage Assistant — SFT + DPO Recipe

## Table of Contents

1. [Dataset A — ANR-MALADES/MediQAl](#a-anr-maladesmediqal)
2. [Dataset B — qanastek/frenchmedmcqa](#b-qanastekfrenchmedmcqa)
3. [Dataset C — keivalya/MedQuad-MedicalQnADataset](#c-keivalyamedquad-medicalqnadataset)
4. [Dataset D — TsinghuaC3I/UltraMedical-Preference](#d-tsinghuac3iultramedical-preference)
5. [The Triage Gap — Honest Assessment](#the-triage-gap--honest-assessment)
6. [Available Triage-Labeled Datasets](#available-triage-labeled-datasets)
7. [Recommended Dataset-Construction Recipe](#recommended-dataset-construction-recipe)
8. [Train / Val / Test / Clinical-Eval Splits](#train--val--test--clinical-eval-splits)
9. [GDPR and Licensing Summary](#gdpr-and-licensing-summary)

---

## A — ANR-MALADES/MediQAl

**HF page:** [https://huggingface.co/datasets/ANR-MALADES/MediQAl](https://huggingface.co/datasets/ANR-MALADES/MediQAl)  
**Paper:** [arxiv 2507.20917](https://arxiv.org/abs/2507.20917) / [Scientific Data, Nature](https://www.nature.com/articles/s41597-026-06680-y)

### What it is

MediQAl is a French medical QA benchmark built from questions scraped from the *qcmlab* website (a platform where French medical professors and students share exam questions from the ECN — Épreuves Classantes Nationales, the 6th-year national ranking exams). Scraping was performed in March 2024. A separate OEQ subset was collected from HTML/PDF files. All questions were validated by a scientific advisory board of tenured academic/hospital faculty. Funded by ANR grant ANR-23-IAS1-0005.

**Note on FrenchMedMCQA:** MediQAl and FrenchMedMCQA are **independent datasets**. The MediQAl paper cites FrenchMedMCQA as a prior work it was designed to supersede (FrenchMedMCQA covers only pharmacy; MediQAl covers 41 specialties). They share ANR MALADES funding but are separate HF releases.

### Size and splits

| Config | Train | Val | Test | Total |
|--------|------:|----:|-----:|------:|
| MCQU (single answer) | 10 113 | 2 561 | 4 343 | 17 017 |
| MCQM (multiple answers) | 5 767 | 1 466 | 3 384 | 10 617 |
| OEQ (open-ended) | — | — | 4 969 | 4 969 |
| **Grand total** | | | | **32 603** |

### Schema

**MCQU / MCQM:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique question ID |
| `clinical_case` | string | Clinical vignette (may be empty for standalone questions) |
| `question` | string | The question text |
| `answer_a` – `answer_e` | string | Five candidate answer options |
| `correct_answers` | string | One letter (MCQU) or comma-separated letters e.g. "A,C,E" (MCQM) |
| `task` | string | `"MCQU"` or `"MCQM"` |
| `medical_subject` | string | One of 41 specialties (e.g. "Cardiologie", "Urgences") |
| `question_type` | string | `"Understanding"` or `"Reasoning"` |

**OEQ:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique question ID |
| `clinical_case` | string | Clinical vignette |
| `cc_question_number` | int | Question index within the vignette |
| `question` | string | The question text |
| `answer` | string | Free-text reference answer |
| `medical_subject` | string | One of 41 specialties |
| `question_type` | string | `"Understanding"` or `"Reasoning"` |

### Concrete example row (MCQU)

```json
{
  "id": "mcqu_0042",
  "clinical_case": "Un patient de 67 ans se présente aux urgences avec une douleur thoracique irradiant dans le bras gauche depuis 30 minutes.",
  "question": "Quel est le diagnostic le plus probable ?",
  "answer_a": "Péricardite aiguë",
  "answer_b": "Syndrome coronarien aigu",
  "answer_c": "Embolie pulmonaire",
  "answer_d": "Dissection aortique",
  "answer_e": "Spasme oesophagien",
  "correct_answers": "B",
  "task": "MCQU",
  "medical_subject": "Cardiologie",
  "question_type": "Reasoning"
}
```
*(Illustrative — reconstructed from the paper's described structure; exact text is on the HF viewer.)*

### License

**CC-BY 4.0** — permissive, requires attribution, commercially usable.

### How it maps to SFT

The OEQ split (4 969 rows) maps directly to instruction → response pairs. The MCQ splits require a conversion step: wrap the question + options into an instruction, and convert the correct answer(s) into an explanatory response. The `clinical_case` field, when non-empty, provides rich context that makes the instruction feel like a realistic clinical scenario.

**Conversion template (MCQ → SFT):**
```
instruction: "Vous êtes un assistant médical francophone. Répondez à la question suivante
 en vous basant sur le cas clinique fourni et expliquez votre raisonnement.

Cas clinique : {clinical_case}
Question : {question}
A) {answer_a}  B) {answer_b}  C) {answer_c}  D) {answer_d}  E) {answer_e}"

response: "La réponse correcte est {correct_answers}. [Explanation of the clinical
 reasoning based on the correct answer]"
```

For the "explanation" part, you have two options: (i) use only the MCQU rows where `question_type == "Reasoning"` and the model is implicitly expected to reason (minimal effort), or (ii) generate short explanations via the Qwen3-1.7B model itself in a bootstrapping pass (more effort, better training signal).

---

## B — qanastek/frenchmedmcqa

**HF page:** [https://huggingface.co/datasets/qanastek/frenchmedmcqa](https://huggingface.co/datasets/qanastek/frenchmedmcqa)  
**Instruction-tuning variant:** [qanastek/LLaMaInstructionsFrenchMedMCQA](https://huggingface.co/datasets/qanastek/LLaMaInstructionsFrenchMedMCQA)  
**GitHub:** [https://github.com/qanastek/FrenchMedMCQA](https://github.com/qanastek/FrenchMedMCQA)

### What it is

The first publicly available French medical MCQA dataset, built from real French **pharmacy specialization diploma (DES Pharmacie)** exams. Questions and answers were manually created by medical experts and used in real examinations. Covers pharmacology, pharmaceutical chemistry, biochemistry, etc. Notably narrower in scope than MediQAl (pharmacy only vs. 41 specialties).

### Size and splits

| Split | Rows |
|-------|-----:|
| Train | 2 171 |
| Validation | 312 |
| Test | 622 |
| **Total** | **3 105** |

### Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `question` | string | Question text |
| `answer_a` – `answer_e` | string | Five candidate answers |
| `correct_answers` | list[string] | Correct option letters, e.g. `["C","D","E"]` |
| `choice_type` | string | `"single"` or `"multiple"` |
| `subject_name` | string | Subject category (e.g. "Pharmacologie") |

The `LLaMaInstructionsFrenchMedMCQA` variant adds:

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | string | Full instruction + question + answer + response, LLaMA chat-formatted |
| `prompt_no_answer` | string | Same without the response (for inference) |

### Concrete example row

```json
{
  "id": "frenchmedmcqa_0217",
  "question": "Parmi les propositions suivantes concernant les chylomicrons, lesquelles sont exactes ?",
  "answer_a": "Ils sont synthétisés dans le foie",
  "answer_b": "Leur principal lipide est le cholestérol",
  "answer_c": "Ils transportent les triglycérides d'origine alimentaire",
  "answer_d": "Ils sont catabolisés par la lipoprotéine lipase",
  "answer_e": "Leur apoprotéine principale est l'Apo B48",
  "correct_answers": ["C", "D", "E"],
  "choice_type": "multiple",
  "subject_name": "Pharmacologie"
}
```
*(Reconstructed from the paper example cited in the dataset card.)*

### License

**Apache 2.0** — permissive, commercially usable, requires preservation of notices.

### How it maps to SFT

Same MCQ-to-instruction pattern as MediQAl. The `LLaMaInstructionsFrenchMedMCQA` variant already provides a usable `prompt` field — you can use it directly after reformatting to the chat template of Qwen3. At 3 105 rows this is a smaller contribution, but its pharmacy focus adds vocabulary that MediQAl (which emphasises clinical medicine) lacks.

---

## C — keivalya/MedQuad-MedicalQnADataset

**HF page:** [https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)  
**Original source:** [github.com/abachaa/MedQuAD](https://github.com/abachaa/MedQuAD) — 47 457 QA pairs from 12 NIH websites (Ben Abacha & Demner-Fushman, BMC Bioinformatics 2019)  
**Canonical HF re-release (fuller):** [lavita/MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD)

### What it is

A re-packaging of a subset of the NIH MedQuAD corpus into CSV/Parquet format. The original MedQuAD contains 47 457 QA pairs crawled from 12 NIH websites (cancer.gov, niddk.nih.gov, GARD, MedlinePlus, NHLBI, CDC, etc.). The `keivalya` version contains 16 407 rows — it is a filtered or de-duplicated subset, not the full corpus. Three NIH sub-sources (A.D.A.M. Medical Encyclopedia, MedlinePlus Drug information, MedlinePlus Herbal/supplement) had answers removed by the original authors to respect copyright; this subset likely reflects what was available post-filtering.

### Size and splits

Single split: **16 407 rows** (all in `train`). No val/test split provided.

### Schema

| Field | Type | Description |
|-------|------|-------------|
| `qtype` | string (16 classes) | Question category |
| `Question` | string (16–191 chars) | The question |
| `Answer` | string (6–29 000 chars) | The answer |

**Known `qtype` values (16 classes, partial list from dataset card):**
`susceptibility`, `symptoms`, `exams and tests`, `treatment`, `prevention`, `information`, `frequency`, `complications`, `genetic changes`, `inheritance`, `outlook`, `research`, `support groups`, `stages`, `summary`, `nursing`

### Concrete example row

```json
{
  "qtype": "susceptibility",
  "Question": "Who is at risk for Lymphocytic Choriomeningitis (LCM)?",
  "Answer": "LCMV infections can occur after exposure to fresh urine, droppings, saliva, or nesting materials from infected rodents. Infection may also result when these materials are directly introduced into broken skin or mucous membranes, eyes, nose, or mouth, or presumably by a bite from an infected animal..."
}
```

### License

**Important caveat:** The `keivalya` HF card does not state a license. The original MedQuAD GitHub repo (abachaa) explicitly states **CC-BY 4.0**. You should: (1) cite the original paper (Ben Abacha & Demner-Fushman 2019), (2) attribute NIH as the source, and (3) document this as CC-BY 4.0 in your GDPR data sheet. Do not use the `keivalya` copy if your institution requires an explicit card license — use `lavita/MedQuAD` instead, which is better documented.

### How it maps to SFT

Directly usable as instruction → response pairs. The `Question` field becomes the instruction; the `Answer` becomes the response. No MCQ conversion needed. The `qtype` field lets you filter by clinical relevance: prioritize `symptoms`, `treatment`, `exams and tests`, `complications`, and `prevention` for a triage use case; deprioritize `support groups` and `nursing`.

**Translation opportunity:** For the FR/EN mix, a random sample of 1 000–1 500 rows can be machine-translated to French using a fast MT model (e.g. facebook/nllb-200 or the Helsinki-NLP opus-mt models on HF) and added to the French portion of the training set.

---

## D — TsinghuaC3I/UltraMedical-Preference

**HF page:** [https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference)  
**GitHub:** [https://github.com/TsinghuaC3I/UltraMedical](https://github.com/TsinghuaC3I/UltraMedical)  
**Paper:** NeurIPS 2024 D&B Track Spotlight — [arxiv 2406.03949](https://arxiv.org/html/2406.03949v1)

### What it is

The preference-annotated component of the UltraMedical collection, a NeurIPS 2024 Spotlight paper from Tsinghua University. GPT-4 was used to score and rank responses from multiple medical LLMs (GPT-4-1106-preview, GPT-3.5-turbo-1106, and others) on biomedical questions drawn from 10 datasets. The test split (777 rows) was reviewed by human domain experts and constitutes the "Medical RewardBench."

### Size and splits

| Split | Rows | Notes |
|-------|-----:|-------|
| Train | ~110 000 | GPT-4 annotated |
| Validation | ~2 232 | GPT-4 annotated |
| Test | 777 | Human expert reviewed |

The **10 source datasets** in the training split:

| Source | Rows |
|--------|-----:|
| WikiInstruct | 27 500 |
| TextBookQA | 16 000 |
| MedMCQA | 17 600 |
| MedQA | 10 800 |
| MedQA-Evol | 9 200 |
| Medical-Instruction-120k | 5 000 |
| MedInstruct-52k | 2 800 |
| ChatDoctor | 11 300 |
| MedQuad | 5 900 |
| PubMedQA | 3 300 |

### Schema

| Field | Type | Description |
|-------|------|-------------|
| `prompt_id` | string | Dataset name + index, e.g. `"TextBookQA,Gynecology_Novak_6275"` |
| `prompt` | string | The medical question or instruction |
| `chosen` | list[dict] | Preferred response — `[{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]` |
| `rejected` | list[dict] | Non-preferred response — same structure |
| `metadata` | dict | GPT-4 scores, ranks, model names, evaluation rationale |
| `label_type` | string | `"model"` (GPT-4 annotated) or `"human"` (expert reviewed) |

### Concrete example row (schema)

```json
{
  "prompt_id": "TextBookQA,Gynecology_Novak_6275",
  "prompt": "Which enzyme catalyzes the first and rate-limiting step of the kynurenine pathway?\nA. Indoleamine 2,3-dioxygenase\nB. Kynureninase\nC. ...",
  "chosen": [
    {"role": "user", "content": "Which enzyme..."},
    {"role": "assistant", "content": "The answer is A. Indoleamine 2,3-dioxygenase (IDO). IDO catalyzes... [detailed mechanistic explanation]"}
  ],
  "rejected": [
    {"role": "user", "content": "Which enzyme..."},
    {"role": "assistant", "content": "The answer is A... [concise but less detailed explanation]"}
  ],
  "metadata": {"chosen_model": "gpt-4-1106-preview", "chosen_score": 5.0,
               "rejected_model": "gpt-3.5-turbo-1106", "rejected_score": 4.0}
}
```

### DPO field mapping

For the TRL `DPOTrainer`:

| TRL field | Source field |
|-----------|-------------|
| `prompt` | `prompt` |
| `chosen` | `chosen[1]["content"]` (the assistant turn) |
| `rejected` | `rejected[1]["content"]` (the assistant turn) |

### License

**MIT** — maximally permissive.

**Known quirk:** The `test.json` file lacks the `feedback` column present in the train/validation splits; a schema mismatch in the HF viewer is documented. This does not affect the training data.

---

## The Triage Gap — Honest Assessment

### The problem stated plainly

All four named datasets are **medical QA or MCQ resources**. None contains:

- An urgency level label (e.g. maximal / moderate / deferred, or ESI 1–5)
- A multi-turn adaptive dialogue structure (follow-up questions based on initial symptoms)
- A "triage" framing that asks the model to prioritize action based on severity

The product description calls for exactly these things. This is a **real gap**, not a minor formatting issue.

### How large is the gap in practice?

For a **proof-of-concept** whose goal is to demonstrate a working end-to-end ML pipeline rather than a production triage system, the gap is manageable but requires an explicit decision about what "triage assistant" means in the deliverable. The grader is evaluating your SFT + DPO + serving pipeline, not your ER nursing protocol.

### Option i — Reframe as a bilingual medical QA assistant with triage-style guidance

**Effort:** low. **Honesty:** explicit.

Keep the named datasets as-is. Add a system prompt that frames the assistant as a triage helper: "You are a bilingual (French/English) medical information assistant for the CHSA emergency department. When a patient describes symptoms, (1) acknowledge the symptoms, (2) ask one targeted follow-up question, (3) provide a preliminary urgency indication (urgence maximale / urgence modérée / urgence différée)."

Then wrap ~500–800 rows of QA data (especially MediQAl MCQU rows with `medical_subject == "Urgences"`) into this 3-part response format manually or via simple templating. This gives the model a triage flavor without fabricating a clinical dataset.

**Pros:** transparent, low risk, reproducible.  
**Cons:** the urgency labels are synthetic and not clinically validated.

### Option ii — Synthesize urgency labels on top of QA data

**Effort:** medium. **Risk:** moderate.

Use a rule-based heuristic to assign urgency levels to MedQuAD `qtype` values:

| qtype | Urgency heuristic |
|-------|------------------|
| symptoms (acute/cardiac/neurological keywords) | maximal |
| treatment, complications | moderate |
| prevention, information, frequency | deferred |

Then write ~400 triage-flavored instruction templates that call for a 3-step response: (1) symptom clarification question, (2) urgency level, (3) recommended action. Apply these templates to ~1 000 QA pairs. Result: ~1 000 triage-flavored SFT pairs. This is a legitimate data engineering exercise.

**Pros:** demonstrates end-to-end data creation skills, which is a graded deliverable.  
**Cons:** urgency labels are heuristic, not clinically validated. Must be flagged clearly in the report.

### Option iii — Use a real triage-labeled dataset

Two real options exist:

**olaflaitinen/fedmml-ed-triage** ([HF](https://huggingface.co/datasets/olaflaitinen/fedmml-ed-triage))
- 87 234 synthetic ED encounters, ESI levels 1–5, structured fields (vitals + labs + clinical notes), CC-BY 4.0
- The structured tabular format (vitals, labs) is not naturally a text instruction-response task — requires non-trivial ETL to convert to NL dialogue
- Good for classification training, less good for generative QA

**jackf7499/MIMIC-IV-Ext_Triage_Instruction_Corpus** ([HF](https://huggingface.co/datasets/jackf7499/MIMIC-IV-Ext_Triage_Instruction_Corpus) / [PhysioNet](https://physionet.org/content/mietic/1.0.0/))
- 9 629 real ED triage cases, already in instruction/input/output format, ESI 1–5
- **License: CC-BY-NC-SA-4.0** — non-commercial. Fine for this educational project.
- Access: the HF version appears publicly visible; the PhysioNet version requires credentialed access (CITI training + Data Use Agreement). **For a 2-week POC, the HF version is the practical path.**
- English only; covers ESI 1–5 with realistic clinical rationale in the `output` field

**syntech-ai/medical-triage-500** ([HF](https://huggingface.co/datasets/syntech-ai/medical-triage-500))
- 500 synthetic triage entries, CC-BY-NC-4.0, urgency_category + urgency_reasoning fields
- Too small to train on, useful only as an eval or few-shot reference

**Verdict on Option iii:** MIMIC-IV-Ext_Triage (MIETIC) is the best available real triage instruction dataset. At 9 629 rows it is substantial. The CC-BY-NC-SA-4.0 license is acceptable for a non-commercial academic POC. **Recommended to include a small slice (500–800 rows) as triage-flavored training examples**, not as the primary training corpus.

---

## Available Triage-Labeled Datasets — Summary Table

| Dataset | HF ID | Rows | Lang | License | Triage labels | Usable for this POC |
|---------|-------|-----:|------|---------|--------------|---------------------|
| FedMML ED Triage | olaflaitinen/fedmml-ed-triage | 87 234 | EN | CC-BY 4.0 | ESI 1–5 (tabular) | Marginal — tabular, needs ETL |
| MIETIC | jackf7499/MIMIC-IV-Ext_Triage_Instruction_Corpus | 9 629 | EN | CC-BY-NC-SA 4.0 | ESI 1–5 (instruction format) | **Yes** — already instruction-formatted |
| medical-triage-500 | syntech-ai/medical-triage-500 | ~500 | EN | CC-BY-NC 4.0 | urgency_category | Only for eval/few-shot |

---

## Recommended Dataset-Construction Recipe

This recipe targets exactly **5 000 SFT instruction-response pairs** and a **3 000-row DPO set**, with an approximate **80% French / 20% English** language mix, within the 2-week constraint and no local GPU.

### Step 1 — French SFT pairs (~4 000 rows)

**Source 1: MediQAl OEQ** — 4 969 open-ended rows, already in question → free-text-answer format. Use the full OEQ split (no train/test leakage since it ships without a train split — use it all for SFT, hold out 10% for val). Wrap into a chat template:

```python
{
  "messages": [
    {"role": "system", "content": "Vous êtes un assistant médical francophone expert, formé pour aider les professionnels de santé du CHSA."},
    {"role": "user", "content": clinical_case + "\n\n" + question},
    {"role": "assistant", "content": answer}
  ]
}
```

Filter to rows where `len(answer) > 50` to remove near-empty answers. Expected yield: **~4 200 rows**.

From these 4 200 rows, take a random 3 500 for SFT training.

**Source 2: MediQAl MCQU (Reasoning subset)** — filter `question_type == "Reasoning"` in the train split (~5 000 rows). Convert MCQ → instruction using the template from Section A, but drop the 4 wrong options and write the response as: "La réponse est {letter}. {answer_text}." For these you need a reference explanation — simplest approach: use only the answer letter + the correct answer text as the response body. Expected yield after filtering: **~3 000 usable rows**. Take **500 rows** (random sample) to avoid over-representing MCQ-style training.

**Source 3: FrenchMedMCQA** — 2 171 train rows. Apply the same MCQ → instruction template. Use the `LLaMaInstructionsFrenchMedMCQA` variant's `prompt` field if available, then reformat to Qwen3 chat format. Expected yield: **~1 800 rows** after filtering single-answer questions for simplicity. Take **all 1 800 rows** (pharmacy vocabulary is valuable).

**Running French total: 3 500 + 500 + 1 800 = 5 800 rows available. Reserve 200 for val, use 4 000 for SFT training.**

### Step 2 — Triage flavor injection (~500 rows, French + English)

Take 300 MediQAl rows where `medical_subject == "Urgences"` (emergency medicine subject). For each, apply one of three triage instruction templates that ask the model to:
1. Ask one clarifying follow-up question
2. Assign an urgency level (urgence maximale / urgence modérée / urgence différée)
3. State the recommended action

Template example:
```
Vous êtes un assistant de triage aux urgences du CHSA. 
Un patient se présente avec : {clinical_case}.
Votre rôle :
1) Posez une question de suivi pertinente pour préciser la gravité.
2) Attribuez un niveau d'urgence : urgence maximale / urgence modérée / urgence différée.
3) Recommandez l'action immédiate appropriée.
```

For the response, use the existing `answer` field from MediQAl + a short heuristic urgency label derived from medical_subject + question_type ("Reasoning" + emergency keywords → maximale, etc.). This is honest template augmentation, not hallucination.

Add 200 rows from MIETIC (jackf7499/MIMIC-IV-Ext_Triage_Instruction_Corpus) directly — they are already in instruction/input/output format and provide English triage grounding.

**Triage injection total: ~500 rows.**

### Step 3 — English SFT pairs (~1 000 rows)

From keivalya/MedQuad-MedicalQnADataset, filter `qtype` in `['symptoms', 'treatment', 'exams and tests', 'complications', 'prevention']`. Expected yield: ~8 000 rows. Sample **1 000 rows** (stratified by qtype). Wrap into chat template:

```python
{
  "messages": [
    {"role": "system", "content": "You are a bilingual medical information assistant for the CHSA emergency department."},
    {"role": "user", "content": row["Question"]},
    {"role": "assistant", "content": row["Answer"]}
  ]
}
```

Filter rows where `len(Answer) > 100` to exclude stub answers.

**Running SFT total: 4 000 (FR) + 500 (triage) + 1 000 (EN) = 5 500 rows. Reserve 500 for validation. Final SFT train set: ~5 000 rows.**

### Step 4 — DPO preference set (~3 000 rows)

From TsinghuaC3I/UltraMedical-Preference, sample 3 000 rows from the training split. Filtering recommendations:
- Prefer rows where `metadata.chosen_score >= 4.5` and `metadata.rejected_score <= 4.0` — a score gap ensures the preference signal is meaningful
- Prefer `label_type == "human"` rows (the 777-row test split used as train data would be wrong; instead apply the human-preference filter on the train split if `label_type` is populated there)
- Exclude rows where `prompt_id` starts with `"MedQuad"` to avoid overlap with your SFT set

Convert to TRL DPOTrainer format:
```python
{
  "prompt": row["prompt"],
  "chosen": row["chosen"][1]["content"],    # assistant turn
  "rejected": row["rejected"][1]["content"] # assistant turn
}
```

**DPO set: 3 000 rows (English, MIT license).**

---

## Train / Val / Test / Clinical-Eval Splits

| Split | Purpose | Size | Source |
|-------|---------|-----:|--------|
| SFT train | Fine-tuning with LoRA | ~5 000 | Mix per recipe above |
| SFT val | Monitor training loss (early stopping) | ~500 | Held-out from same sources, stratified |
| SFT test | Report final metrics (perplexity, BLEU, BERTScore) | ~300 | Fully held-out, not seen during training |
| Clinical eval | Human qualitative review of 30–50 cases | 30–50 | Hand-curated triage scenarios, written from scratch |
| DPO train | Preference alignment | ~3 000 | UltraMedical-Preference |
| DPO val | Monitor DPO reward margin | ~200 | Held-out from same preference pool |

**Clinical-eval set construction:** Write 30–50 fictional patient vignettes in French that cover: cardiac emergency, stroke, allergic reaction, non-urgent headache, chronic pain follow-up. These are synthetic and written de novo — no PII, no real patient data, no GDPR issues. Have a clinician (or knowledgeable reviewer) rate model outputs on a 1–5 scale. This is your "graded deliverable" clinical evaluation.

---

## GDPR and Licensing Summary

| Dataset | License | Personal data? | GDPR action |
|---------|---------|---------------|-------------|
| ANR-MALADES/MediQAl | CC-BY 4.0 | None — exam questions | Attribute; note source website (qcmlab) |
| qanastek/frenchmedmcqa | Apache 2.0 | None — exam questions | Attribute; preserve notices |
| keivalya/MedQuad-MedicalQnADataset | Unlisted on HF card; original CC-BY 4.0 | None — NIH patient-education text | Use CC-BY 4.0 per original; cite Ben Abacha 2019 |
| TsinghuaC3I/UltraMedical-Preference | MIT | None — synthetic/MCQ + GPT-4 annotations | Attribute; cite Zhang et al. NeurIPS 2024 |
| MIETIC (if used) | CC-BY-NC-SA 4.0 | De-identified MIMIC-IV derived | Non-commercial only; cite PhysioNet DOI |

**GDPR status of the constructed dataset:** All source data is either (a) exam questions (no patient data), (b) NIH patient-education web text (public domain / CC-BY), or (c) synthetic/GPT-4-generated. The constructed SFT and DPO sets contain **no real patient data and no personally identifiable information**. The clinical-eval set is written de novo with fictional vignettes. The dataset is GDPR-compliant by construction; document this explicitly in your data sheet (deliverable 1).

---

## Implementation Notes and Gotchas

1. **MediQAl availability:** The dataset paper appeared on arxiv in July 2025. Verify the HF dataset is not behind an access gate before relying on it. Test with `datasets.load_dataset("ANR-MALADES/MediQAl", "OEQ")` in Colab.

2. **keivalya vs. lavita MedQuAD:** `lavita/MedQuAD` is a fuller, better-documented HF re-release of MedQuAD (explicit CC-BY 4.0 card). If `keivalya`'s license situation is unsatisfactory to the grader, switch to `lavita/MedQuAD` with no functional change to the pipeline.

3. **MCQ → instruction quality:** For MCQM (multiple-answer) rows in FrenchMedMCQA and MediQAl, the response "Les réponses correctes sont C, D, E" is a weak training signal unless you also provide an explanation. For a quick POC, filter to `choice_type == "single"` rows only (FrenchMedMCQA) and `task == "MCQU"` (MediQAl), which together give you ~12 000 single-answer rows to sample from.

4. **UltraMedical-Preference schema mismatch:** The test split lacks the `feedback` column. Load only `train` and `validation` splits for your DPO set; do not load `test`.

5. **Language imbalance:** The proposed recipe yields approximately 4 500 FR rows / 1 000 EN rows ≈ 82% FR / 18% EN. This comfortably satisfies a "bilingual" brief without requiring machine translation.

6. **Compute cost of data prep:** All ETL (loading datasets, filtering, wrapping in templates, saving to JSONL) runs on CPU in minutes. No GPU needed for data preparation. Budget 1–2 hours of Colab CPU time.

7. **Total token estimate:** At an average of ~300 tokens per SFT pair and 5 000 pairs, the SFT set is ~1.5M tokens. On a T4 (16 GB), with LoRA rank 16 and Qwen3-1.7B, this fits in ~3–4 hours of training at batch size 4 with gradient accumulation 8.


---

## Open questions to confirm during implementation

- Does keivalya/MedQuad-MedicalQnADataset carry an explicit license in its HF card? If not, confirm re-use is safe by falling back to the CC-BY-4.0 of the original abachaa/MedQuAD GitHub repo.
- Are the OEQ (open-ended) rows in MediQAl genuinely usable for SFT, or are the free-text answers too terse? A quick manual review of 20 rows is advised before committing to them.
- The MediQAl paper appeared in arxiv July 2025 — confirm the dataset is already fully public on HF (no embargo) by running `datasets.load_dataset('ANR-MALADES/MediQAl', 'OEQ')` in a Colab cell.
- For the triage framing: what urgency taxonomy does the CHSA use — the French CCMU scale (1-5), the CIMU/CTAS variant, or a simplified 3-level scale (maximal / moderate / deferred)? The prompt templates must match.
- Does the grading rubric treat the DPO set as a separate GDPR-compliant deliverable, or is it bundled with the SFT dataset? This affects whether the synthetic triage examples need separate documentation.
