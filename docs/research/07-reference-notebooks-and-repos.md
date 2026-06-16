> **TL;DR — key takeaways**
>
> - The single most important SFT reference is the Unsloth Qwen3 notebook collection (unsloth.ai/docs + GitHub notebooks repo) — it already targets Kaggle T4 and Qwen3-4B, and the 1.7B variant runs with even more headroom.
> - A Kaggle community notebook (kaggle.com/code/vaibhavbarala26/fine-tuning-qwen-3-1-7b-with-lora) fine-tunes exactly Qwen3-1.7B with LoRA — this is the closest off-the-shelf match to the project's model and hardware target.
> - The TRL DPO Trainer docs now use Qwen3-0.6B as their canonical quick-start example, making the code directly portable to Qwen3-1.7B with a one-line model ID swap.
> - For the bilingual medical dataset requirement, MedInjection-FR (github.com/ikram28/MedInjection-FR) is the only large-scale open French biomedical instruction dataset and was validated specifically with a Qwen-4B-Instruct LoRA setup.
> - FreedomIntelligence/medical-o1-reasoning-SFT (HF Hub) is the best English-language clinical reasoning SFT dataset: 90K pairs with verified chain-of-thought, Apache-2.0 licensed, and already used to fine-tune Qwen-class models.
> - RunPod serverless deployment via the official runpod-workers/worker-vllm Docker image requires only setting MODEL_NAME and HF_TOKEN env vars — no custom server code needed for a POC.
> - The Unsloth vLLM guide shows the critical save_method='merged_16bit' pattern: merge LoRA adapters into base weights before pushing to HF Hub, which avoids PEFT-at-runtime complexity in the vLLM server.


# Reference Notebooks and Repos for the CHSA Medical Triage LLM POC

This document catalogs **13 concrete, linkable resources** organized by the four pipeline stages: SFT+LoRA, DPO alignment, medical-domain fine-tuning, and vLLM deployment. Each entry states what to copy directly and what to skip or adapt for your single-GPU, two-week POC.

---

## Category 1 — Qwen3 SFT + LoRA Fine-Tuning with Unsloth

Unsloth is a drop-in library that wraps `transformers` and `TRL`, adds hand-written Triton kernels, and achieves roughly 2× training speed at 60–70 % lower VRAM compared to a plain HuggingFace loop. It runs on Kaggle T4 and Colab free tiers. All Unsloth notebooks follow the same skeleton: load model in 4-bit (`FastLanguageModel.from_pretrained`), attach LoRA adapters, format the dataset, call `SFTTrainer.train()`, and optionally push to HF Hub.

### Resource 1 — Unsloth official Qwen3 how-to-run-and-fine-tune doc

**URL:** https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune

**What it demonstrates:** Step-by-step SFT of Qwen3 (from 4B to 14B) with 4-bit QLoRA in Colab and Kaggle. Explains "thinking mode" vs "non-thinking mode" (Qwen3 can toggle chain-of-thought at inference), and how to disable it for production to reduce token count. Includes the recommended LoRA target modules, context length settings, and the `chat_template` format.

**Copy directly:**
- The `FastLanguageModel.from_pretrained(model_name, max_seq_length, load_in_4bit=True)` call and the `get_peft_model(...)` block with `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`.
- The prompt formatting function using `tokenizer.apply_chat_template`.
- The `SFTTrainer` + `TrainingArguments` configuration (batch size, gradient accumulation, learning rate, warmup ratio).

**Skip / adapt:**
- Thinking-mode examples are for reasoning-heavy tasks; turn it off (`enable_thinking=False`) for your triage assistant so answers are concise.
- The 14B notebook requires an A100; use the 4B notebook for free Kaggle T4.

---

### Resource 2 — Unsloth Qwen3 (4B) Instruct Colab Notebook

**URL:** https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-Instruct.ipynb

**What it demonstrates:** End-to-end SFT on a conversational instruction dataset in a single Colab cell sequence. You can open it directly in Colab (free T4) or import it into Kaggle via the Kaggle notebook welcome URL. The notebook includes W&B / no-logging options and GGUF export at the end.

**Copy directly:**
- The entire training cell block — it is self-contained and can be repurposed by swapping the dataset variable.
- The `model.save_pretrained_merged(...)` cell for exporting to vLLM-compatible merged weights.

**Skip / adapt:**
- The dataset used is a generic alpaca-style dataset. Replace with your medical instruction dataset.
- Qwen3-4B is larger than needed. For a T4 16 GB target, you can run 4B at 4-bit, but Qwen3-1.7B leaves more headroom.

---

### Resource 3 — Unsloth Kaggle Qwen3 (14B) Notebook import link

**URL:** https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Qwen3_%2814B%29.ipynb

**What it demonstrates:** Same Unsloth SFT workflow but pre-configured for Kaggle's T4×2 or P100 environment, including Kaggle-specific pip install lines. The notebook handles the 14B model — relevant because it shows the multi-GPU gradient accumulation settings you would only reduce (not increase) for a 1.7B model.

**Copy directly:**
- The Kaggle-specific pip install block (`!pip install unsloth`).
- The `max_seq_length=2048` and `per_device_train_batch_size` recommendations for T4.

**Skip / adapt:**
- 14B model won't fit on a single T4 without significant tricks; scale down to Qwen3-1.7B or 4B.

---

### Resource 4 — Kaggle community notebook: "Fine tuning Qwen 3 1.7b with Lora"

**URL:** https://www.kaggle.com/code/vaibhavbarala26/fine-tuning-qwen-3-1-7b-with-lora

**What it demonstrates:** Exactly your target model (Qwen3-1.7B) fine-tuned with LoRA on Kaggle. This is the closest off-the-shelf reference to your pipeline.

**Copy directly:**
- The model ID (`Qwen/Qwen3-1.7B`) and any confirmed LoRA hyperparameters (the notebook uses standard Unsloth defaults).
- Confirmation that this model trains within T4 VRAM budget.

**Skip / adapt:**
- Verify the dataset — the notebook may use a generic English dataset. Swap in your bilingual medical instruction set.
- Check the Kaggle kernel was executed recently; re-run to verify compatibility with current library versions.

> Confidence note: The page loaded as a Kaggle title-only in the fetcher (authentication wall), so the exact LoRA rank and dataset inside the notebook could not be confirmed remotely. Open it directly on Kaggle to verify.

---

## Category 2 — DPO Preference Alignment

DPO (Direct Preference Optimization) is a simplified alternative to PPO-based RLHF. Instead of training a separate reward model, it directly nudges the model to assign higher probability to "chosen" responses and lower probability to "rejected" responses, using a single classification-like loss. You need a preference dataset with three fields per row: `prompt`, `chosen`, and `rejected`.

### Resource 5 — HuggingFace TRL DPO Trainer official documentation

**URL:** https://huggingface.co/docs/trl/dpo_trainer

**What it demonstrates:** The canonical `DPOTrainer` API. The quick-start example trains `Qwen/Qwen3-0.6B` on `trl-lib/ultrafeedback_binarized` with 4 lines of Python. The page also shows the PEFT+LoRA integration pattern, the expected dataset schema (conversational format with `prompt`/`chosen`/`rejected` message lists), and all key `DPOConfig` hyperparameters (`beta`, `loss_type`, `learning_rate`).

**Copy directly:**
```python
from trl import DPOTrainer
from peft import LoraConfig
from datasets import load_dataset

trainer = DPOTrainer(
    "Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
    peft_config=LoraConfig(),
)
trainer.train()
```
- The conversational dataset format schema (the `prompt`/`chosen`/`rejected` structure with `role`/`content` dicts).
- Default hyperparameters: `beta=0.1`, `learning_rate=1e-6`, `gradient_checkpointing=True`.

**Skip / adapt:**
- `ultrafeedback_binarized` is a general English dataset. For your medical DPO step, you will need to construct or source a medical preference dataset (e.g., generate two responses per prompt with your SFT model and rate them, or use an existing one like `lavita/medical-preference-data` if it exists).
- The 0.6B quick-start model is for illustration; substitute `Qwen/Qwen3-1.7B` (or your SFT checkpoint).

---

### Resource 6 — Unsloth DPO/ORPO/KTO preference optimization guide

**URL:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto

**What it demonstrates:** Unsloth's wrapper around `DPOTrainer`, showing the same LoRA model loading pattern as SFT but swapped for preference data. Shows the Zephyr-7B DPO Colab notebook as the canonical example, along with ORPO and KTO alternatives. Key addition vs. raw TRL: `use_gradient_checkpointing = "unsloth"` saves additional VRAM.

**Direct DPO notebook link:** https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb

**Copy directly:**
- The `ref_model = None` pattern (Unsloth handles the reference model internally when PEFT is active, avoiding loading a second full model).
- `DPOConfig(beta=0.1, max_length=1024, max_prompt_length=512)` — adjust max lengths for your medical prompts which tend to be long.

**Skip / adapt:**
- The example uses Zephyr-7B; swap in your Qwen3-1.7B SFT checkpoint.
- ORPO and KTO are alternative alignment methods; DPO is simpler and better supported for a POC — stick with DPO.

---

### Resource 7 — HuggingFace smol-course, Unit 2 — Preference Alignment with DPO

**URL:** https://huggingface.co/learn/smol-course/unit2/2

**What it demonstrates:** A pedagogically-structured walkthrough of DPO with TRL, aimed at engineers learning the domain (exactly your audience). Uses SmolLM3-3B but all code is framework-level and model-agnostic. Explains DPO theory inline, then shows the full training script. The companion exercise at https://huggingface.co/learn/smol-course/unit2/3 has a runnable notebook.

**Copy directly:**
- The conceptual explanation of `beta` (controls how much the model can deviate from the reference — low beta = conservative, high beta = more aggressive preference enforcement).
- The dataset preprocessing function that converts arbitrary preference data into the `{prompt, chosen, rejected}` schema.

**Skip / adapt:**
- SmolLM3 is not Qwen3; the tokenizer chat template will differ. Use `tokenizer.apply_chat_template` with the Qwen3 template.
- The course targets serverless HF Jobs for training; adapt to Kaggle.

---

## Category 3 — Medical-Domain LLM Fine-Tuning

### Resource 8 — Kaggle notebook: "Fine-tune Gemma using LoRA for Medical Q&A task"

**URL:** https://www.kaggle.com/code/gpreda/fine-tune-gemma-using-lora-for-medical-q-and-a-task

**What it demonstrates:** End-to-end LoRA SFT of a small LLM on a medical QA dataset (MedQuAD or equivalent), running on Kaggle T4. Even though the base model is Gemma (not Qwen3), the dataset formatting, QLoRA loading, and `SFTTrainer` boilerplate are directly transferable.

**Copy directly:**
- Dataset loading and formatting for medical QA (instruction-input-output format for the `SFTTrainer` `dataset_text_field`).
- The evaluation cell that runs sample inference to check answer quality before committing to full training.

**Skip / adapt:**
- Replace Gemma model ID with `Qwen/Qwen3-1.7B`.
- The Gemma chat template differs from Qwen3's; replace the formatting function.

---

### Resource 9 — lavita Medical Instruction Tuning Datasets collection (HuggingFace)

**URL:** https://huggingface.co/collections/lavita/medical-instruction-tuning-datasets

**What it demonstrates:** A curated collection of four ready-to-use medical instruction datasets:

| Dataset ID | Examples | Description |
|---|---|---|
| `axiong/pmc_llama_instructions` | 514k | Medical literature-based instructions from PubMed Central |
| `lavita/ChatDoctor-HealthCareMagic-100k` | 112k | Doctor–patient conversation data |
| `lavita/AlpaCare-MedInstruct-52k` | 52k | Healthcare instructions (AlpaCare style) |
| `xz97/MedInstruct` | 216 | Curated medical instruction pairs |

**Copy directly:**
- `lavita/ChatDoctor-HealthCareMagic-100k` is the most practical for a triage assistant: real doctor–patient Q&A with clinical context. Load with `load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")` and reformat to instruction-response pairs.
- Use `axiong/pmc_llama_instructions` for a larger, higher-quality English-language SFT dataset (subsample to ~5000 examples for your POC to stay within the graded deliverable scope).

**Skip / adapt:**
- These are English-only. For the bilingual French requirement, supplement with translated or native French data (see Resource 10).
- `xz97/MedInstruct` has only 216 examples — too small for SFT on its own; use as eval set.

---

### Resource 10 — MedInjection-FR: French biomedical instruction dataset + GitHub repo

**URL (paper):** https://arxiv.org/html/2603.06905
**URL (GitHub):** https://github.com/ikram28/MedInjection-FR

**What it demonstrates:** The only currently available large-scale French biomedical instruction dataset (571K pairs). Built by translating English medical datasets (via Gemini 2.0 Flash / GPT-4o-mini) plus native French medical QA pairs. The paper fine-tuned **Qwen-4B-Instruct** using DoRA (a LoRA variant), making this the closest architectural match to your project. The repo includes the data pipeline scripts and evaluation prompts.

**Copy directly:**
- The translation pipeline pattern (translate English medical QA to French at scale using a cheap LLM API) — relevant for building your own bilingual dataset.
- The DoRA/LoRA configuration used for Qwen-4B-Instruct (rank 8–16, alpha 16–32, targeting q/v projections).
- The data format: `{instruction, input, output}` triples covering open-ended QA and multiple-choice.

**Skip / adapt:**
- The full 571K dataset is far more than the ~5000 examples your deliverable requires. Sample a balanced subset (e.g., 3000 EN + 2000 FR pairs).
- CC BY-NC-ND 4.0 license: not for commercial use, fine for a student POC.
- DoRA is a refinement of LoRA; plain LoRA (standard Unsloth) works fine for a POC.

---

### Resource 11 — FreedomIntelligence/medical-o1-reasoning-SFT dataset (HuggingFace)

**URL:** https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT

**What it demonstrates:** 90K medical instruction-response pairs generated by GPT-4o on clinical reasoning problems, validated by a medical verifier. Includes English, Chinese, and mixed versions. The English split (`medical_o1_sft.json`) contains ~20K examples with three-field structure: `Question`, `Complex_CoT` (chain-of-thought reasoning), and `Response`. This dataset was specifically designed for Qwen-class models (the HuatuoGPT-o1 model was fine-tuned with it on Qwen-7B).

**Copy directly:**
- The dataset loading and reformatting: use `Question` as the instruction, `Response` as the output. Optionally prepend `Complex_CoT` as a `<think>` block if you want chain-of-thought behavior.
- The `medical_o1_sft_mix.json` variant (English + Chinese) as a proxy for bilingual data.

**Skip / adapt:**
- The CoT content is clinical reasoning steps. For a triage assistant that needs short triage decisions, you may want to strip the CoT or place it behind a `<think>` tag and only expose the `Response` field at inference.
- 90K examples is far too many for a T4 training run (OOM / timeout). Subsample to 5000–8000.

---

## Category 4 — vLLM Deployment

vLLM (Virtual LLM) is an inference engine that uses PagedAttention to dramatically increase GPU throughput. It exposes an OpenAI-compatible REST API (`/v1/chat/completions`) out of the box. For a fine-tuned model, you either (a) load the base model + LoRA adapter at serve time, or (b) merge the adapter into the base weights before serving. Option (b) is simpler and more reliable for a POC.

### Resource 12 — Unsloth vLLM deployment guide

**URL:** https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide

**What it demonstrates:** How to go from an Unsloth-trained model directly to a running vLLM server. The key step is saving with the right format:

```python
# In the training notebook, after training:
model.save_pretrained_merged("finetuned_model", tokenizer, save_method="lora")
# Then push to HF Hub:
model.push_to_hub_merged("your-hf-username/chsa-triage", tokenizer, save_method="merged_16bit")
```

Then serve with:
```bash
vllm serve your-hf-username/chsa-triage
```

**Copy directly:**
- The `save_method="merged_16bit"` pattern to merge LoRA weights into the base model before serving — this avoids vLLM needing to handle PEFT adapters at runtime, which is simpler.
- The `push_to_hub_merged` call to upload merged weights to HF Hub for RunPod to pull.

**Skip / adapt:**
- The guide also shows `save_method="lora"` for smaller upload size. For a POC, merged is fine.
- GGUF export (also shown) is for CPU/llama.cpp inference — not needed for vLLM.

---

### Resource 13 — runpod-workers/worker-vllm GitHub repo + RunPod vLLM docs

**URL (GitHub):** https://github.com/runpod-workers/worker-vllm
**URL (RunPod docs):** https://docs.runpod.io/serverless/vllm/get-started

**What it demonstrates:** The official RunPod serverless template for deploying any HuggingFace-hosted model behind vLLM with scale-to-zero billing. You point it at your HF Hub model ID via an environment variable and it handles the rest.

**Minimal deployment config:**
```
Docker image: runpod/worker-v1-vllm:<latest-version>
Environment variables:
  MODEL_NAME=your-hf-username/chsa-triage
  HF_TOKEN=<your-hf-token>
  MAX_MODEL_LEN=2048
  DTYPE=float16
  GPU_MEMORY_UTILIZATION=0.90
```

**Copy directly:**
- The environment variable schema above — copy it verbatim into your RunPod serverless endpoint configuration.
- The `HF_TOKEN` secret handling: set it as a RunPod secret, not a plain env var.

**Skip / adapt:**
- The `worker-vllm` repo contains a custom handler for non-standard request shapes. For a POC, use the pre-built image directly without modifying the worker code.
- CUDA >= 13.0 requirement: verify the current image tag on the GitHub releases page before deploying.

---

### Supplementary reference — vLLM + FastAPI + Docker standalone article with code

**URL:** https://medium.com/@wpan36/deploy-your-own-lightweight-llm-inference-api-with-vllm-fastapi-docker-on-your-laptop-220a74ead5b7
**GitHub:** https://github.com/wpan36/vllm_in_docker

**What it demonstrates:** A minimal working Dockerfile + FastAPI wrapper around vLLM, using `Qwen2.5-1.5B-Instruct` — architecturally close to your Qwen3-1.7B target. Shows the exact `vllm.AsyncLLMEngine` initialization and a `/generate` POST endpoint. Useful if you want to add a thin FastAPI layer instead of relying on vLLM's built-in OpenAI-compatible server.

**Copy directly:**
- The `Dockerfile` base image (`FROM pytorch/pytorch:<cuda>`) and pip install lines.
- The `gpu_memory_utilization=0.90` and `max_model_len=2048` settings.

**Skip / adapt:**
- Building a custom FastAPI wrapper adds complexity. For your POC, vLLM's built-in OpenAI-compatible server (`vllm serve`) is sufficient and avoids maintaining extra code.

---

## Summary Table

| # | Resource | Category | Platform | Directly copy |
|---|---|---|---|---|
| 1 | Unsloth Qwen3 fine-tune guide | SFT+LoRA | Doc | LoRA config, chat template |
| 2 | Unsloth Qwen3-4B Instruct Colab notebook | SFT+LoRA | Colab | Full training cell block |
| 3 | Unsloth Kaggle Qwen3-14B notebook | SFT+LoRA | Kaggle | Kaggle pip install, T4 config |
| 4 | Kaggle "Fine tuning Qwen 3 1.7b with Lora" | SFT+LoRA | Kaggle | Model ID, T4 fit confirmation |
| 5 | TRL DPO Trainer official docs | DPO | Doc | DPOTrainer + LoRA code, dataset schema |
| 6 | Unsloth DPO/ORPO guide + Zephyr notebook | DPO | Colab | `ref_model=None` pattern, DPOConfig |
| 7 | HF smol-course Unit 2 DPO | DPO | HF Course | Beta explanation, dataset preprocess |
| 8 | Kaggle Gemma medical Q&A LoRA notebook | Medical SFT | Kaggle | Dataset formatting, eval cell |
| 9 | lavita medical instruction datasets | Medical SFT | HF Hub | ChatDoctor-100k, AlpaCare-52k |
| 10 | MedInjection-FR paper + GitHub | Medical SFT | GitHub | French data pipeline, Qwen LoRA config |
| 11 | FreedomIntelligence medical-o1-SFT | Medical SFT | HF Hub | 20K medical EN QA pairs |
| 12 | Unsloth vLLM deployment guide | Deployment | Doc | `save_method=merged_16bit`, vllm serve |
| 13 | runpod-workers/worker-vllm + RunPod docs | Deployment | GitHub/RunPod | Env var schema for serverless |

---

## Recommended Execution Order

1. **Open Resource 4** (Kaggle Qwen3-1.7B LoRA notebook) to confirm T4 fit, then use **Resource 2** (Unsloth Qwen3-4B Colab) as the code skeleton — it is cleaner and more up-to-date.
2. **Assemble your SFT dataset** by combining a sample from **Resource 11** (medical-o1-SFT, English) and translated/native French pairs from **Resource 10** (MedInjection-FR). Target 3000–5000 examples total.
3. **Run SFT** using the Unsloth skeleton (Resources 1–3). Export merged weights to HF Hub with `push_to_hub_merged`.
4. **Construct a small DPO preference set** (~500–1000 pairs) by generating two candidate answers per prompt with your SFT model, then rating them (rule-based on safety/completeness is fine for a POC). Format using the schema from **Resource 5**.
5. **Run DPO** using the TRL pattern from **Resource 5** + Unsloth memory tricks from **Resource 6**. This is a short run (few hundred steps).
6. **Deploy** using **Resource 13** (RunPod + worker-vllm) with merged weights from HF Hub. Use **Resource 12** for the Unsloth export command.


---

## Open questions to confirm during implementation

- Does the Kaggle Qwen3-1.7B LoRA notebook (Resource 4) actually train end-to-end without OOM on a T4? The page was behind a Kaggle auth wall — needs manual verification by opening the notebook.
- What is the current stable Docker image tag for runpod/worker-v1-vllm? The README references specific version numbers that change frequently — check github.com/runpod-workers/worker-vllm/releases before deploying.
- MedInjection-FR is CC BY-NC-ND 4.0 — confirm this is acceptable for a student POC before building the dataset pipeline on it. The NC clause forbids commercial use; ND forbids redistribution of derivatives.
- Qwen3-1.7B is a 'Base' (pre-trained, not instruction-tuned) model. Confirm that the project intends to SFT from the Base rather than from an Instruct variant — starting from Base requires more SFT data to learn instruction-following behavior.
- For the DPO step, no publicly available medical preference dataset (chosen/rejected pairs) was found. Will the preference set be constructed synthetically (e.g., generate two answers per prompt and score them automatically), or is there an existing dataset the project plans to use?
- The Unsloth vLLM guide mentions CUDA >= 13.0 as a requirement for the latest worker image. Verify that the RunPod GPU instances available at the time of deployment satisfy this, especially for budget-tier L4 or T4 pods.
