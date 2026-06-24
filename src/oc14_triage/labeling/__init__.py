"""Triage labelling pipeline: 3 LLMs apply a cited triage rubric to real MediQAl
clinical vignettes, emit a 3-level urgency + an ESI-1-5 bonus, and we keep the
high-agreement consensus as a (silver-standard) gold/train set. Deterministic
parts (rubric, parsing, consistency, consensus, Fleiss' kappa) are unit-tested
with a mock client; only the API calls need keys.
"""
