"""Agentic triage POC: the LangGraph orchestration around the fine-tuned model.

The graph nodes ARE the CHSA patient-journey chain:
collecte → anonymisation (Presidio) → prétraitement/validation → triage (LLM) →
explication → persistance (traçabilité) → intégration SIH.

Backend-agnostic: the triage node calls an OpenAI-compatible endpoint (local Ollama,
or RunPod vLLM) selected by env; a STUB mode lets the whole chain run with no model.
"""
