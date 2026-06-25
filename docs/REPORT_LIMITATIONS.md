# OC14 — Limitations & honest caveats (report-ready)

This POC demonstrates a **method** (specialise a small bilingual LLM for triage + an honest eval) and a
**progress signal**, **not a deployable autonomous triager**. The caveats below are load-bearing and
should appear in the report; several were surfaced by an adversarial audit (`docs/KNOWN_ISSUES.md`).

## 1. Silver-standard labels, not clinical ground truth
No clinician was available; the triage labels are a **3-LLM consensus** (GPT-5.4 + Mistral-Medium-3.5 +
Sonnet-4.6) over real MediQAl vignettes. LLM↔clinician triage agreement is only *moderate* in the
literature. Mitigations: 3-model consensus, unanimous-only "gold", over-triage default, MCQU calibration
floor-check. It remains a **silver standard**.

## 2. Circularity + easy-subset — the key eval caveat
The held-out gold = the **3-LLM unanimous** label, and the model is SFT'd on the **same 3-LLM labels**.
So the headline metric measures **imitation fidelity to the teachers**, not clinical accuracy. Worse, the
gold is the **unanimous (easy) slice** — ambiguous/disagreement cases are excluded from **both** train and
eval, so the number is **optimistic vs the full population**.
- The tempting "score predicted urgency vs the stored `gold_esi` bucket" check **does not work**: gold cases
  are ESI-consistent *by construction* (`is_gold` requires `all_consistent`), so it is identical to scoring
  vs `gold_urgency` — zero independent signal.
- **Genuine semi-independent checks (planned):** eval on `syntech-ai/medical-triage-500` (independent
  dataset, different label source; English + synthetic) and the 6 hand-labelled vignettes (n=6). The gap
  between in-distribution gold and the independent set bounds the imitation inflation.

## 3. Conservative (over-triage) bias — safe but imprecise *(measured)*

| | gold → predicted | recall (95% CI) |
|---|---|---|
| urgence maximale | 91 ✓ · 9→modérée · **0→différée** | 0.91 [0.84, 0.95] |
| urgence modérée | 85 ✓ · 15→maximale · **0→différée** | 0.85 [0.77, 0.91] |
| urgence différée | 28 ✓ · **72→modérée** · 0→maximale | 0.28 [0.20, 0.38] |

The model **never under-triages** (zero maximale/modérée → différée) but **systematically over-triages
low-acuity**: *différée* recall collapses to 0.28 and *modérée* becomes a dumping ground (**precision 0.51**
— half of "modérée" predictions are *différée* pushed up). This is clinically the *safe* direction
(under-triage is the dangerous error) but is **bought with precision/efficiency**; at the limit, a triager
that calls everything moderate-or-worse has no triage value. The **3-level** scale concentrates the bias in
the middle class (no granularity to express "low-but-not-lowest"). Causes: corpus skew + over-triage rubric
rule (~2 pp, ablation-measured) + *différée*-starved training. Headline **macro-F1 = 0.65** (greedy,
leak-free, stratified n=300); an earlier **0.81** was inflated by an eval→train leak + sampled decoding +
noisy labels and is **retracted**.

## 4. Language — French-primary; bilingual only weakly met
Train is **79% FR / 21% EN**, but the EN is almost entirely **general medical QA** — EN **triage** is only
**4 rows**, and the **eval is 100% French**. So "mostly French" is satisfied and the model is FR-primary,
but the **triage task and its evaluation are effectively FR-only**. To meet a bilingual requirement
properly: grow EN triage (oversample the 4 EN vignettes + author more) and add an EN eval slice (or use the
English `medical-triage-500` as the EN check).

## 5. Corpus representativeness
Training/eval cases are real French **medical-exam** vignettes → they over-represent serious pathology
relative to a real ED (~47% maximale vs ~25–30%). A production system would need a **representative,
prospectively-collected ED triage dataset**; this is a PoC proxy.

## 6. Statistical power
n = 100 per class → wide CIs (e.g. maximale recall 0.91 → **[0.84, 0.95]**). Report CIs; the conservative
safety figure is the **lower bound (0.84)**; per-class deltas (e.g. before/after DPO) must exceed ~±0.04.

## 7. Safety bar
0.91 maximale recall (floor **0.84**) means **≥1-in-10 emergencies missed at the lower bound** —
**unacceptable for autonomous ICU/ED triage**. Reaching ≥99% recall would force precision to collapse (a
false-alarm flood needing human review anyway). Position the system as **decision-support / human-in-the-loop**.
