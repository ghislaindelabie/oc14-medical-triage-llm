# OC14 — Known issues & pre-GPU audit (2026-06-25)

Adversarial code+methodology audit of the SFT-on-LLM-labels result (macro-F1 0.813) and the whole
labelling/training/eval setup, run before spending more Kaggle GPU. 29 candidate findings → **16
confirmed** after independent verification (23 agents). This is the fix list; it also feeds the
report's *limitations / methodology* section.

## What this means for the headline 0.813
It is **mildly optimistic and not yet reproducible**, for four reasons below (E2, E3, M1, M2). After
the fixes the post-clean number is expected to differ by roughly the ±0.04 confidence band. **Report it
with Wilson CIs and the imitation-fidelity framing, never as a bare clinical-accuracy point estimate.**

## (1) Errors / bugs — fix before the next Kaggle run
- **E1 [HIGH] Non-consensus rows leak into SFT training with arbitrary labels.** `run.py:135`
  `train_src = [urgency and not is_gold]` admits 1-1-1 three-way splits (flagged) and ESI-inconsistent
  rows → meaningless tie-winner labels on the hardest cases, poisoning `triage_sft_train.jsonl`. Fix:
  persist `n_agree` in `cmd_label`'s JSONL; gate `... and not flagged and n_agree>=2`.
- **E2 [HIGH] Held-out eval-gold leaks into training (~30/300) via the QA pipeline.** `run.py` +
  `data/build_sft.py:57-88` + `build_retrain_sft.py`. Disjointness is enforced only by `case_id`
  *within* the triage pipeline; the same MediQAl `clinical_case` is independently reshaped into QA
  rows that the retrain keeps → ~8-12% of gold *presentations* are seen in training (input-only leak,
  partial inflation). My exact-string check missed it (QA framing differs). Fix: shared "used
  clinical_case" registry; exclude gold cases whose text matches a kept train row; assert `gold ∩
  train == 0`. **Moves the headline down slightly.**
- **E3 [MEDIUM] Urgency extraction reads the earliest level substring, not the verdict line.**
  `metrics.py:24-28` + notebook mirror. "ce n'est pas une *urgence maximale* … Niveau : *modérée*" →
  scores maximale (wrong). Can inflate maximale recall + pollute confusion. Fix: anchor to the
  `Niveau d'urgence :` line; report silent-fallback rate.
- **E4 [MEDIUM] Val/train cross-task contamination** (11 val-triage cases as train QA). Same registry
  fix as E2. Harmless today (no early-stopping) but a real leak once `eval_strategy` is on.
- **E5 [MEDIUM] 4/11 hand-written vignettes randomly dropped** by triage-pool truncation
  (`build_sft.py:135-145`), including the only EN *modérée* triage example. Fix: concatenate the 11
  vignettes *after* truncation so they're always kept.
- **E6/E7/E8 [LOW]** Mistral retry too narrow (HTTPError-only); OpenAI cache-read priced at Anthropic's
  0.1× (cost-report only); SFT sanity-eyeball cell omits `eos_token_id`. Defer / cheap.

## (2) Methodology — eval trustworthiness
- **M1 [foreground this] Circularity + easy-subset.** gold = 3-LLM *unanimous* consensus; model SFT'd
  on the same labels → 0.813 = **imitation fidelity to the teachers on the easy (unanimous) slice**,
  not clinical accuracy; ambiguous cases never enter eval. Disclose on every mention. Semi-independent
  signal available: score predicted urgency vs the stored `gold_esi` bucket (currently unused).
- **M2 Non-deterministic eval** (`do_sample=True, temp=0.3`, no seed) → not reproducible; Base-vs-SFT
  -vs-DPO deltas carry unreported sampling noise. Fix: **greedy** (`do_sample=False`).
- **M3 n=100/class → wide CIs.** maximale recall 0.93 → Wilson 95% CI **[0.86, 0.97]**; macro-F1 ≈ ±0.04.
  Report CIs; cite ~0.86 as the conservative safety floor; SFT→DPO deltas must beat the CI band.
- **M4 Fleiss κ on complete-data subset only** → optimistic, undisclosed N. Report `n_kappa_items`.
- **Safety bar (user):** 0.93 maximale recall = ~7% missed emergencies → **unacceptable for autonomous
  ICU/ED triage**; ≥99% recall would force precision to collapse (false-alarm flood) → needs human
  review anyway. Position as **decision-support / human-in-the-loop**, not autonomous; the value is the
  *method* + the *progress signal* (naive-Base → SFT → DPO), not a deployable autonomous triager.

## (3) Optimizations (for the upcoming baseline + DPO evals)
- **O1 [dominant lever] Eval runs 300 generations at batch size 1** (~84 min). Left-pad + batch 16-32/call
  → est. **~6-12 min** (8-15×), reused across baseline/SFT/DPO evals. Pairs with M2 (greedy).
- **O2** `max_new_tokens` 256→128 (answer ~100 tok). **O3** DPO loads a 150-row val but sets no
  `eval_strategy` → add `eval_strategy='steps'` for a cheap "is DPO separating chosen>rejected" signal.
- **O4** inference-only kernels run the full training install (trl) — trim for baseline/eval/merge.

## (4) Simplifications
- **S1** `triage_report`/`extract_urgency` copy-pasted inline into the eval cells (drift risk;
  `language_match_rate` never inlined). Ship `eval/metrics.py` in the Kaggle dataset and `import` it.
- **S2** tests assert on the stale `processed/sft_train.jsonl`, not the shipped `kaggle_upload/*` → the
  real trained artifact is untested (no test would have caught E2/E4/E5). Repoint + add asserts:
  0 gold∩train, 0 dup (user,assistant), all 11 vignettes present.

## Execution order before GPU
1. **E1 + E2 + E4** (shared registry / filter) → rebuild the data. 2. **S2** test guards. 3. **E5, E3, M2**
into `EVAL_CELLS`/`build_sft.py`. 4. **O1 + O2 + S1** (one eval-cell pass). 5. **M1 + M3** reporting,
**O3**. Defer E6/E7, M4/M5, O4.

Then GPU: rebuild clean → re-run SFT + (batched greedy) eval → naive-Base baseline (same harness) → DPO.
