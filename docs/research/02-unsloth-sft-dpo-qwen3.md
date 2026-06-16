> **TL;DR — key takeaways**
>
> - Unsloth delivers roughly 2x faster training and 70% less VRAM than a vanilla HuggingFace+FlashAttention-2 baseline, making Qwen3-1.7B fine-tuning trivially comfortable on a free T4 16 GB.
> - Qwen3-1.7B is fully supported: Unsloth publishes a pre-quantized 4-bit checkpoint (unsloth/Qwen3-1.7B-unsloth-bnb-4bit) and the same FastLanguageModel API used for 14B models works unchanged for 1.7B.
> - The SFT pipeline is three function calls: FastLanguageModel.from_pretrained → get_peft_model (adds LoRA adapters) → TRL SFTTrainer.train(). No custom loops needed.
> - DPO requires a single extra line before TRL's DPOTrainer: PatchDPOTrainer(). Data must have three string columns — prompt, chosen, rejected — with chat-template formatting already applied.
> - For vLLM serving, save with save_method='merged_16bit' to produce a plain HuggingFace SafeTensors directory; never serve the raw 4-bit QLoRA checkpoint directly through vLLM.
> - Version pinning is critical and breaks frequently: the official Qwen3 notebooks pin transformers==4.56.2 and trl==0.22.2 as of June 2026 — always copy these pins from the latest official notebook, not from memory.
> - Kaggle free tier (T4 x2, 30 h/week) is the better free-GPU choice over Colab for long runs: no 90-minute idle-disconnect, more reliable GPU allocation, and 1-2 GBps download from HuggingFace. Store HF_TOKEN via Kaggle Secrets, not in the notebook source.


# Unsloth for SFT+LoRA and DPO — Practical Guide for the CHSA Medical Triage POC

> Audience: a solo learner building a bilingual medical triage assistant on free Kaggle/Colab GPUs, targeting Qwen3-1.7B-Base, with vLLM serving on RunPod.
> Current as of June 2026. Version-sensitive items are flagged explicitly.

---

## 1. What Unsloth Is

**Unsloth** ([unsloth.ai](https://unsloth.ai)) is an open-source Python library that wraps HuggingFace Transformers and TRL (the library that provides SFTTrainer, DPOTrainer, etc.) with hand-written Triton kernels and memory-management tricks to make LoRA and QLoRA fine-tuning significantly faster and cheaper. It does **not** replace the Transformers/PEFT/TRL ecosystem — it accelerates it transparently.

Unsloth is **not** a hosted service. You install it in your notebook, it patches the model layers at load time, and everything else (datasets, TRL trainers, HuggingFace Hub) continues to work as normal.

**Licence:** Apache 2.0 for the pip core library; AGPL-3.0 only for the optional Unsloth Studio web UI. The `pip install unsloth` package is Apache 2.0, safe for project use. ([GitHub](https://github.com/unslothai/unsloth))

---

## 2. Concrete Performance Gains vs. Vanilla HuggingFace

The gains come from three stacked techniques:

| Technique | What it does |
|---|---|
| Custom Triton kernels | Re-implements attention, RoPE embeddings, SwiGLU/GeGLU activations, and the cross-entropy loss as fused GPU ops that avoid repeated HBM round-trips. The QK-RoPE kernel alone is 2.3x faster than the stock implementation. |
| Smart gradient checkpointing (`use_gradient_checkpointing="unsloth"`) | Stores only a small set of activations, recomputing the rest on the backward pass. Unsloth's version is more selective than vanilla `torch.utils.checkpoint`, saving more VRAM without extra compute. |
| Uncontaminated packing | Concatenates multiple short training examples into one long sequence before feeding them to the GPU, eliminating padding waste. Attention masks prevent cross-contamination. Results in 2.1–2.5x more tokens processed per second at 50% less VRAM vs. padded batches. |

### Benchmark numbers

These are Unsloth's own published figures, which align with third-party comparisons ([Red Hat Developer, April 2026](https://developers.redhat.com/articles/2026/04/01/unsloth-and-training-hub-lightning-fast-lora-and-qlora-fine-tuning); [The AI Engineer Substack](https://theaiengineer.substack.com/p/unsloth-vs-axolotl-vs-llama-factory)):

| Comparison | Speed | VRAM |
|---|---|---|
| Unsloth LoRA vs. HF + FlashAttention 2 | ~2x faster | ~70% less |
| Unsloth QLoRA (4-bit) vs. full fine-tune | ~2x faster | ~75% less |
| Unsloth packing enabled vs. padded baseline | 2.1–3x faster | ~50% less |
| Unsloth vs. Axolotl (A100 40 GB, Llama-3.1 8B QLoRA) | 3.2 h vs. 5.8 h | — |
| GRPO/RL training | ~10% faster | ~80% less vs. standard |

**Practical consequence for this project:** Qwen3-1.7B QLoRA on a T4 16 GB fits well within Unsloth's published 3.5 GB minimum for a 3B model in 4-bit mode. You will have headroom to spare.

---

## 3. Qwen3-1.7B Support — Confirmed

Unsloth officially supports the full Qwen3 family including 1.7B, 4B, 8B, 14B, 30B, and the MoE variants.

- Pre-quantized 4-bit checkpoint: [`unsloth/Qwen3-1.7B-unsloth-bnb-4bit`](https://huggingface.co/unsloth/Qwen3-1.7B-unsloth-bnb-4bit) — 1.4B active parameters, 32k context, GQA architecture.
- Full collection (4-bit, GGUF, 16-bit): [Unsloth Qwen3 collection on HuggingFace](https://huggingface.co/collections/unsloth)
- Official doc page: [Qwen3 — How to Run & Fine-tune](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune)

**For the base model**, use `unsloth/Qwen3-1.7B` (16-bit) or `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` (4-bit). The `-Base` suffix (no instruct tuning) is what you want for SFT from scratch.

---

## 4. SFT + LoRA Workflow

The workflow is three sequential steps.

### 4.1 Install

Copy the exact install block from the current official notebook each time you start a project — version pins change frequently. As of June 2026 the Qwen3 14B notebook uses:

```python
%%capture
import os, re

# On Colab
if "COLAB_" in "".join(os.environ.keys()):
    import torch
    v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
    xformers = 'xformers==' + {
        '2.10': '0.0.34',
        '2.9':  '0.0.33.post1',
        '2.8':  '0.0.32.post2',
    }.get(v, "0.0.34")
    !pip install sentencepiece protobuf "datasets==4.3.0" \
        "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate \
        {xformers} peft trl triton unsloth
    !pip install --no-deps --upgrade "torchao>=0.16.0"
else:
    # Kaggle / local
    !pip install torch torchvision torchaudio xformers \
        --index-url https://download.pytorch.org/whl/cu128
    !pip install unsloth
    !pip install --no-deps --upgrade "torchao>=0.16.0"

# Version pins — copy from the latest official notebook
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
```

> **Version pitfall:** `transformers` and `trl` have frequent breaking changes. The `==4.56.2` / `==0.22.2` pins are what the official notebook uses as of June 2026. Always copy from the [current notebook source](https://raw.githubusercontent.com/unslothai/notebooks/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb) rather than relying on these notes.

### 4.2 Load model with FastLanguageModel

`FastLanguageModel` is Unsloth's drop-in replacement for `AutoModelForCausalLM`. It patches the model layers with custom kernels at load time.

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    # For the base (non-instruction) model use "unsloth/Qwen3-1.7B"
    max_seq_length = 2048,   # Up to 32768 for Qwen3; start small
    load_in_4bit = True,     # QLoRA — recommended for T4 16 GB
    load_in_8bit = False,
    full_finetuning = False,
)
```

**When to use 4-bit vs 8-bit:**
- `load_in_4bit=True` → QLoRA, ~4 GB VRAM for 1.7B, train faster, slight accuracy trade-off.
- `load_in_8bit=True` → more accurate, roughly double the VRAM (~8 GB for 1.7B), slower.
- For a T4 with 16 GB, either works for 1.7B. 4-bit is the right default for this POC.

### 4.3 Add LoRA adapters with get_peft_model

LoRA (Low-Rank Adaptation) inserts small trainable matrices (rank `r`) alongside the frozen base weights. Only these tiny matrices are updated during training — for 1.7B at r=16 that is around 5 million trainable parameters out of 1.7 billion, or ~0.3%.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                   # Rank. 8–32 is the usual range. 16 is a good default for 1.7B.
    lora_alpha = 16,          # Scaling factor — keep equal to r as a starting point.
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention projections
        "gate_proj", "up_proj", "down_proj",       # feed-forward (MLP) projections
    ],
    lora_dropout = 0,         # Unsloth recommends 0 for its optimised kernels.
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # Critical — use Unsloth's version, not "True".
    random_state = 3407,
    use_rslora = False,       # RSLoRA normalises alpha/sqrt(r) — useful at high ranks.
    loftq_config = None,
)
```

**Key parameters explained:**
- `r` (rank): controls capacity. Higher r = more parameters trained = more expressive but more VRAM. For 1.7B on T4, r=16 is safe and sufficient for a POC.
- `lora_alpha`: a scaling multiplier. Setting it equal to r is the common practice and gives a net scale factor of 1.0.
- `target_modules`: which weight matrices get LoRA adapters. The seven listed above cover all attention and MLP projections and are standard for Qwen3.
- `use_gradient_checkpointing="unsloth"`: use exactly this string, not `True`. Unsloth's implementation is both faster and uses less VRAM than PyTorch's default.

### 4.4 Train with SFTTrainer

SFT (Supervised Fine-Tuning) teaches the model to respond to prompts by showing it many (prompt, response) pairs. `SFTTrainer` from the TRL library handles this loop.

**Dataset format:** your dataset must have a column (default name: `"text"`) where each row is a fully formatted string including the chat template. For Qwen3 that looks like:

```
<|im_start|>system
You are a medical triage assistant.<|im_end|>
<|im_start|>user
J'ai une forte fièvre depuis deux jours.<|im_end|>
<|im_start|>assistant
Voici les informations importantes...<|im_end|>
```

Apply the template in bulk using the tokenizer:

```python
def format_example(example):
    messages = [
        {"role": "system",    "content": example["system"]},
        {"role": "user",      "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )}

dataset = dataset.map(format_example)
```

Then train:

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,   # effective batch size = 8
        warmup_steps = 10,
        num_train_epochs = 3,              # or use max_steps for quick experiments
        learning_rate = 2e-4,
        logging_steps = 10,
        optim = "adamw_8bit",              # 8-bit Adam saves ~2 GB VRAM vs fp32 Adam
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",               # change to "wandb" if you want tracking
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
    ),
)
trainer.train()
```

---

## 5. DPO Workflow

DPO (Direct Preference Optimization) is a second training stage that uses pairs of responses — one "chosen" (preferred) and one "rejected" — to nudge the model toward better outputs without a separate reward model. It is run **after** SFT, on the SFT-trained model.

### 5.1 Data format

Each row must have exactly three string columns after chat-template formatting:

```python
{
  "prompt":   "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n",
  "chosen":   "The better answer<|im_end|>\n",
  "rejected": "The worse answer<|im_end|>\n"
}
```

The `prompt` column must end with the assistant-turn opening tag so the trainer knows where the response begins. The `chosen` and `rejected` columns contain only the assistant's response text (plus closing token).

### 5.2 Code

```python
from unsloth import PatchDPOTrainer   # Must come BEFORE importing DPOTrainer
PatchDPOTrainer()                      # Patches TRL's DPOTrainer with Unsloth kernels

from trl import DPOTrainer, DPOConfig

dpo_trainer = DPOTrainer(
    model = model,        # the SFT-trained model (with LoRA adapters)
    ref_model = None,     # None = use the frozen LoRA base as reference (standard PEFT-DPO)
    args = DPOConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 1,          # DPO typically needs fewer epochs than SFT
        learning_rate = 5e-6,          # Much lower LR than SFT — DPO is a fine adjustment
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "dpo_outputs",
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        report_to = "none",
    ),
    beta = 0.1,              # KL-divergence penalty strength. 0.1 is the standard default.
    train_dataset = dpo_dataset,
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
)
dpo_trainer.train()
```

**Key DPO parameters explained:**
- `ref_model=None`: when the model has LoRA adapters, TRL automatically uses the frozen base weights as the reference distribution. This avoids loading a second full model copy, saving ~half the VRAM.
- `beta`: controls how far the model is allowed to drift from the reference. Lower beta (0.05) = more aggressive preference learning; higher (0.5) = more conservative. 0.1 is the standard starting point.
- Learning rate: must be much lower than SFT (5e-6 vs 2e-4) because DPO moves the distribution rather than learning new knowledge.

**Call `PatchDPOTrainer()` before any TRL import** — it monkey-patches the class, and calling it after instantiation will have no effect.

---

## 6. Saving the Model for vLLM Serving

### Options at a glance

| Format | Command | Size (1.7B) | Use case |
|---|---|---|---|
| LoRA adapters only | `model.save_pretrained("adapter/")` | ~50–100 MB | Lightweight; requires base model at serve time |
| Merged 16-bit | `model.save_pretrained_merged(..., save_method="merged_16bit")` | ~3.4 GB | **Recommended for vLLM** |
| Merged 4-bit | `model.save_pretrained_merged(..., save_method="merged_4bit")` | ~0.9 GB | Not recommended for vLLM |
| GGUF Q4_K_M | `model.save_pretrained_gguf(..., quantization_method="q4_k_m")` | ~1.1 GB | llama.cpp, Ollama |

### Recommended approach for this project: merged 16-bit

```python
# Save locally
model.save_pretrained_merged(
    "qwen3-1.7b-chsa-merged",
    tokenizer,
    save_method = "merged_16bit",
)

# Or push directly to HuggingFace Hub
model.push_to_hub_merged(
    "your-hf-username/qwen3-1.7b-chsa",
    tokenizer,
    save_method = "merged_16bit",
    token = "YOUR_HF_TOKEN",
)
```

**Why merged 16-bit for vLLM:**
- vLLM expects a standard HuggingFace model directory. Merged 16-bit produces exactly that: SafeTensors files + `config.json` + tokenizer files.
- vLLM can then be invoked simply as `vllm serve ./qwen3-1.7b-chsa-merged`.
- Serving a raw 4-bit QLoRA checkpoint (the `.bnb` quantized format) directly with vLLM is not straightforward and is officially discouraged by Unsloth.
- Serving LoRA adapters separately requires the `--enable-lora` flag and vLLM's LoRA serving mode, which adds complexity without benefit for a single-model endpoint.

**GGUF is for llama.cpp/Ollama** — not vLLM. Do not confuse the two.

---

## 7. Running on Kaggle and Colab Free Tiers

### 7.1 Comparison

| Factor | Google Colab Free | Kaggle Free |
|---|---|---|
| GPU | Tesla T4, 16 GB VRAM | T4 x1 or T4 x2, 16 GB (x2 = 32 GB combined) |
| Weekly quota | ~15–30 GPU hours (dynamic) | 30 GPU hours/week (fixed) |
| Session idle disconnect | ~90 minutes of inactivity | No idle disconnect |
| Hard session cap | ~12 hours | ~12 hours per session |
| Internet access | Yes | Yes (must be toggled ON in settings) |
| HuggingFace download speed | Good | 1–2 GBps (very fast) |
| Secrets management | Colab secrets or `userdata` | Kaggle Secrets panel → `UserSecretsClient` |
| Persistent disk | No (session only) | Datasets + output quota |
| GPU availability | Not guaranteed at peak | More reliable |

**Recommendation for this project:** use Kaggle. The 90-minute idle-disconnect on Colab free tier can kill a multi-hour training run. Kaggle's 30 h/week free is reliable and the T4 x2 option gives you 32 GB combined VRAM if you enable multi-GPU (though Unsloth single-GPU is simpler and sufficient for 1.7B).

### 7.2 Kaggle install and HF_TOKEN setup

```python
# Cell 1 — install (Kaggle flavour, CUDA 12.8 wheels)
%%capture
import subprocess
subprocess.run([
    "pip", "install", "torch", "torchvision", "torchaudio", "xformers",
    "--index-url", "https://download.pytorch.org/whl/cu128"
], check=True)
subprocess.run(["pip", "install", "unsloth"], check=True)
subprocess.run(["pip", "install", "--no-deps", "--upgrade", "torchao>=0.16.0"], check=True)
subprocess.run(["pip", "install", "transformers==4.56.2"], check=True)
subprocess.run(["pip", "install", "--no-deps", "trl==0.22.2"], check=True)
```

```python
# Cell 2 — authenticate to HuggingFace using Kaggle Secrets
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

secrets = UserSecretsClient()
hf_token = secrets.get_secret("HF_TOKEN")   # Key name you set in the Secrets panel
login(hf_token)
```

> To add the secret: in your Kaggle notebook → Settings → Secrets → Add New Secret → key: `HF_TOKEN`, value: your HuggingFace read token.

> **Internet access:** Kaggle notebooks have internet **disabled by default**. Toggle it on: Settings (right sidebar) → Internet → turn on. This is required to download model weights from HuggingFace.

### 7.3 Colab install (for quick experiments)

```python
%%capture
import os, re, torch

v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
xformers = 'xformers==' + {
    '2.10': '0.0.34', '2.9': '0.0.33.post1', '2.8': '0.0.32.post2'
}.get(v, "0.0.34")
!pip install sentencepiece protobuf "datasets==4.3.0" \
    "huggingface_hub>=0.34.0" hf_transfer
!pip install --no-deps unsloth_zoo bitsandbytes accelerate \
    {xformers} peft trl triton unsloth
!pip install --no-deps --upgrade "torchao>=0.16.0"
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
```

HuggingFace auth on Colab:
```python
from google.colab import userdata
from huggingface_hub import login
login(userdata.get('HF_TOKEN'))   # Store in Colab Secrets (key icon in left sidebar)
```

### 7.4 Known gotchas on free tiers

| Gotcha | Details |
|---|---|
| Version pins break silently | `transformers` 4.x and `trl` 0.x have breaking changes every few weeks. Always copy the pins from the current official notebook source. |
| Colab idle disconnect | The runtime dies after ~90 minutes with no browser interaction. For a SFT run on 5000 examples this is likely fine (run completes in ~1–2 h for 1.7B), but save checkpoints (`save_steps=50` in SFTConfig). |
| Kaggle T4 x2 multi-GPU | Unsloth multi-GPU on Kaggle T4 x2 has a known bug (Unsloth issue #5178) where it only detects one GPU. Use single T4 to stay safe. |
| FP8 not on T4 | T4 does not support FP8 precision (requires at least L4). Do not set `fp8=True` in training args — use bf16/fp16 as shown above. |
| CUDA driver vs torch wheels | The Kaggle install block uses `--index-url .../cu128` (CUDA 12.8 wheels). If Kaggle's driver is older, use `cu124` or `cu121` instead. Check with `!nvidia-smi` before installing. |
| Model download quota | Qwen3-1.7B 16-bit is ~3.4 GB; 4-bit is ~0.9 GB. Kaggle's per-notebook output quota is 20 GB. Save adapters (100 MB) to output, push merged model to HF Hub before the session ends. |
| `enable_thinking` flag | Qwen3 models have a "thinking" mode activated by `enable_thinking=True` in `apply_chat_template`. For a triage assistant you want `enable_thinking=False` to get direct answers, not chain-of-thought tokens. |

---

## 8. Official Notebooks — Exact URLs

All notebooks live at `github.com/unslothai/notebooks/tree/main/nb/`. The Colab links open the notebook directly in Colab; the Kaggle links clone it into a new Kaggle notebook editor.

### Qwen3 SFT notebooks

| Notebook | Colab link | Kaggle link |
|---|---|---|
| Qwen3 (14B) Reasoning + Conversational **(recommended starting point)** | [Open in Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb) | [Open in Kaggle](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Qwen3_(14B).ipynb) |
| Qwen3 (4B) Instruct | [Open in Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-Instruct.ipynb) | — |
| Qwen3 (4B) Thinking | [Open in Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-Thinking.ipynb) | — |
| Qwen3 (4B) GRPO | [Open in Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb) | — |

> There is no Qwen3-1.7B-specific notebook. Use the 14B or 4B notebook and change `model_name = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"`. All other code is identical.

### DPO notebook

| Notebook | Colab link | Kaggle link |
|---|---|---|
| DPO Zephyr (7B) **(the canonical DPO example)** | [Open in Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb) | [Open in Kaggle](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Funslothai/notebooks/blob/main/nb/Kaggle-Zephyr_(7B)-DPO.ipynb) |

> The DPO notebook uses Zephyr (a Mistral fine-tune), not Qwen3. The code pattern is identical — swap the model name, apply the Qwen3 chat template instead of Zephyr's, and use the same `PatchDPOTrainer()` call.

### Central notebook directory

[https://unsloth.ai/docs/get-started/unsloth-notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks)

---

## 9. End-to-End Summary for This Project

```
1. Kaggle notebook, T4 GPU, internet ON
2. pip install (Kaggle block above, pins: transformers==4.56.2, trl==0.22.2)
3. HF_TOKEN from Kaggle Secrets → huggingface_hub.login()

--- SFT phase ---
4. FastLanguageModel.from_pretrained("unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
       max_seq_length=2048, load_in_4bit=True)
5. get_peft_model(r=16, lora_alpha=16, target_modules=[...7 projections...],
       use_gradient_checkpointing="unsloth")
6. Format 5000 examples with tokenizer.apply_chat_template(enable_thinking=False)
7. SFTTrainer.train() → save adapter: model.save_pretrained("sft_adapter/")

--- DPO phase ---
8. Load SFT adapter back: FastLanguageModel.from_pretrained() + get_peft_model()
   (or continue in same session)
9. PatchDPOTrainer()  ← before any TRL import
10. DPOTrainer(ref_model=None, beta=0.1, ...).train()

--- Export ---
11. model.save_pretrained_merged("qwen3-chsa-merged", tokenizer,
        save_method="merged_16bit")
12. Push to HF Hub OR copy to RunPod Docker image
13. vllm serve ./qwen3-chsa-merged --host 0.0.0.0 --port 8000
```

---

## Sources

- [Unsloth official documentation](https://unsloth.ai/docs)
- [Unsloth Notebooks directory](https://unsloth.ai/docs/get-started/unsloth-notebooks)
- [Qwen3 — How to Run & Fine-tune (Unsloth docs)](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune)
- [Run & Fine-tune Qwen3 (Unsloth blog)](https://unsloth.ai/blog/qwen3)
- [unsloth/Qwen3-1.7B-unsloth-bnb-4bit (HuggingFace)](https://huggingface.co/unsloth/Qwen3-1.7B-unsloth-bnb-4bit)
- [Unsloth GitHub repository](https://github.com/unslothai/unsloth)
- [vLLM Deployment Guide (Unsloth docs)](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide)
- [Preference Optimization (DPO/ORPO/KTO) — Unsloth docs](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/reinforcement-learning-dpo-orpo-and-kto)
- [How to fine-tune and evaluate Qwen3 with Unsloth (W&B report)](https://wandb.ai/byyoung3/Generative-AI/reports/How-to-fine-tune-and-evaluate-Qwen3-with-Unsloth---VmlldzoxMjU3OTI0Ng)
- [Unsloth and Training Hub: Lightning-fast LoRA (Red Hat Developer, April 2026)](https://developers.redhat.com/articles/2026/04/01/unsloth-and-training-hub-lightning-fast-lora-and-qlora-fine-tuning)
- [Unsloth vs Axolotl vs LLaMA-Factory — The AI Engineer Substack](https://theaiengineer.substack.com/p/unsloth-vs-axolotl-vs-llama-factory)
- [Unsloth vs Standard Training — Pelin Balci, Medium](https://medium.com/@balci.pelin/unsloth-vs-standard-training-92d4c35b8ad8)
- [Qwen3 Fine-tuning — Qwen official ReadTheDocs](https://qwen.readthedocs.io/en/latest/training/unsloth.html)
- [Authenticating to HF in Kaggle Notebooks](https://www.kaggle.com/code/mrisdal/authenticating-to-hf-in-kaggle-notebooks)
- [Google Colab Free Tier T4 Guide 2026](https://aicreditmart.com/ai-credits-providers/google-colab-free-tier-t4-gpu-access-guide-2026/)


---

## Open questions to confirm during implementation

- No Unsloth notebook specifically targets Qwen3-1.7B; the smallest published Qwen3 notebook uses 14B. Verify that the 14B notebook's LoRA rank (r=32) and batch config are sane for 1.7B, or reduce r to 16 to keep training short.
- The DPO notebook (Zephyr 7B) uses ref_model=None, which implicitly uses the frozen LoRA base as the reference. Confirm this is correct behaviour for your version of TRL (it is the standard 'PEFT DPO' pattern, but worth verifying with the pinned trl==0.22.2).
- Unsloth's Kaggle-native install script uses CUDA 12.8 wheels; verify the Kaggle T4 runtime CUDA driver version before copying blindly — if it is older than 12.x, you may need the cu121 or cu124 index URL instead.
- The merged_16bit export for Qwen3-1.7B produces a ~3.4 GB SafeTensors file. Confirm RunPod serverless can cold-load this in under the gateway timeout, or plan to bake the model into the Docker image.
- Unsloth's licence is dual Apache 2.0 (core) / AGPL-3.0 (Studio UI). The pip package used here is the Apache-2.0 core — confirm this is acceptable for the OpenClassrooms deliverable before submitting.
