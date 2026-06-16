> **TL;DR — key takeaways**
>
> - SFTTrainer accepts a peft_config=LoraConfig(...) argument directly, making LoRA setup a one-liner on top of any standard training script — no manual model wrapping required.
> - For a 1.7B LoRA fine-tune on a small dataset, the TRL docs recommend learning_rate around 1e-4 (10x the full fine-tuning default), 2-3 epochs, per_device_batch_size=2 with gradient_accumulation=8-16, and max_length=512-1024 to stay inside T4 VRAM.
> - LoRA rank r=8 or r=16 is sufficient for a small model like Qwen3-1.7B on a focused domain task; larger r increases capacity but also VRAM and overfitting risk on small datasets.
> - DPOTrainer requires a dataset with three columns — prompt, chosen, and rejected — and a beta hyperparameter (default 0.1) that controls how far the aligned model is allowed to drift from the reference model.
> - GRPO differs from DPO in that it is an online RL method that requires the model to actively generate completions during training and score them with a reward function; it excels when you have a programmatic verifiable reward (e.g., correct medical triage category) rather than preference pairs.
> - Use plain TRL+PEFT when you need maximum transparency, debuggability, and compatibility with Kaggle/Colab without custom-kernel dependencies; Unsloth is a drop-in speed multiplier (2-4x) that is worth adding once the pipeline is proven working.


# TRL + PEFT: SFT and DPO for the CHSA Medical Triage Assistant

> **Scope.** This document covers the standard Hugging Face path for supervised fine-tuning (SFT) and preference alignment on the Qwen3-1.7B-Base model. It explains every key concept and hyperparameter in plain language, gives concrete starting values for a 1.7B model on a small (~5 000 example) dataset, and ends with a practical comparison to Unsloth. Sources are drawn from the official TRL docs (v1.6, June 2026) and PEFT docs.

---

## 1. The Library Stack

| Layer | Library | Role |
|---|---|---|
| Training loop | [TRL](https://huggingface.co/docs/trl) (Transformer Reinforcement Learning) | `SFTTrainer`, `DPOTrainer`, `GRPOTrainer` |
| Efficient adapters | [PEFT](https://huggingface.co/docs/peft) (Parameter-Efficient Fine-Tuning) | `LoraConfig`, `get_peft_model` |
| Model + tokenizer | Transformers | `AutoModelForCausalLM`, `AutoTokenizer` |
| Quantization (optional) | bitsandbytes | 4-bit / 8-bit base model loading (QLoRA) |

Install everything needed for this project:

```bash
pip install trl[peft] bitsandbytes datasets accelerate
```

---

## 2. SFTTrainer and How LoRA Plugs In

### 2.1 What SFT does

Supervised Fine-Tuning (SFT) teaches the base model to produce useful responses by showing it many (instruction, response) pairs and minimising the cross-entropy loss on the response tokens. Think of it as the model learning "when a user says X, the right kind of answer looks like Y." The loss is computed **only on the response (completion) tokens**, not on the user prompt — this is the default behaviour when you use a `prompt`/`completion` dataset format.

### 2.2 Dataset formats accepted by SFTTrainer

```python
# Simplest: plain text
{"text": "The patient reports chest pain..."}

# Recommended for instruction-following
{"prompt": [{"role": "user", "content": "Patient reports chest pain..."}],
 "completion": [{"role": "assistant", "content": "Triage level: URGENT..."}]}

# Or with explicit role keys for multi-turn
{"messages": [{"role": "user",   "content": "..."},
              {"role": "assistant", "content": "..."}]}
```

When you pass a conversational dataset, TRL automatically applies the model's chat template, so you do not have to format the `<|im_start|>` / `<|im_end|>` tokens by hand.

### 2.3 LoRA: what it is and why you need it

A 1.7 B parameter model has ~1.7 billion weights. Full fine-tuning would update all of them — more GPU memory than a free Kaggle T4 can hold, and overkill for a focused domain adaptation task.

**LoRA (Low-Rank Adaptation)** freezes the original weights and inserts small trainable matrices alongside targeted linear layers. Concretely, for a weight matrix **W** of shape *(d_out, d_in)*, LoRA adds two matrices **A** *(r, d_in)* and **B** *(d_out, r)* where *r ≪ d_in*. Only A and B are updated during training. The effective update is `ΔW = B × A`, scaled by `alpha / r`. This typically reduces the number of trainable parameters by 100–1000x while retaining 90–95 % of the quality of a full fine-tune for domain adaptation tasks.

### 2.4 Plugging LoRA into SFTTrainer

The integration is one argument: `peft_config=LoraConfig(...)`. TRL calls `get_peft_model()` internally, so you never have to wrap the model yourself.

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("your_hf_username/chsa-sft-dataset", split="train")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

training_args = SFTConfig(
    output_dir="qwen3-1.7b-chsa-sft",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    max_length=1024,
    assistant_only_loss=True,   # compute loss on assistant tokens only
    gradient_checkpointing=True,  # enabled by default in SFTConfig
    bf16=True,                    # enabled by default in SFTConfig
    logging_steps=10,
)

trainer = SFTTrainer(
    model="Qwen/Qwen3-1.7B-Base",
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model("qwen3-1.7b-chsa-sft-lora")
```

After training, `trainer.save_model()` saves only the small LoRA adapter weights (a few MB), not the full 1.7 B parameter model. To run inference, you load the base model and then load the adapter on top using `PeftModel.from_pretrained()`, or merge them permanently with `model.merge_and_unload()`.

---

## 3. Key Hyperparameters Explained

### 3.1 Learning Rate

**What it is.** The step size used to update the weights at each optimiser step. Too high: the loss oscillates and the model "forgets" things. Too low: training is slow and may get stuck.

**Baseline for full fine-tuning.** SFTConfig defaults to `2e-5`.

**With LoRA, multiply by ~10.** Because only a tiny fraction of parameters are trainable, each update has less total effect on the model's behaviour, so you need a larger step to see meaningful change. TRL's official recommendation is `≈1e-4` for SFT with LoRA. Use a cosine or linear learning-rate scheduler with a short warmup (5–10 % of steps).

**For a 1.7B model on ~5 000 examples:** `learning_rate=1e-4` is a solid starting point.

### 3.2 Number of Epochs (`num_train_epochs`)

**What it is.** How many times the trainer passes through the full training dataset. One epoch = all ~5 000 examples seen once.

**How to choose.** With a small dataset you risk **overfitting** (the model memorises training examples instead of generalising). For a focused domain with ~5 000 examples, 2–3 epochs is a reasonable range. Monitor the validation loss: if it starts rising while the training loss keeps falling, you are overfitting — stop early or reduce epochs.

**Starting value:** `num_train_epochs=3`.

### 3.3 Per-Device Train Batch Size (`per_device_train_batch_size`)

**What it is.** The number of training examples processed simultaneously on one GPU before a weight update is computed. A larger batch uses more VRAM.

**Constraint: T4 has 16 GB VRAM.** A Qwen3-1.7B model in bfloat16 + LoRA adapters + activations for one example with 1 024 tokens costs roughly 4–6 GB. Starting with `per_device_train_batch_size=2` is safe; try 4 if VRAM allows.

**Starting value:** `per_device_train_batch_size=2`.

### 3.4 Gradient Accumulation Steps (`gradient_accumulation_steps`)

**What it is.** Instead of updating weights after every mini-batch, the trainer accumulates gradients over N batches before performing a single update. The **effective batch size** = `per_device_batch_size × gradient_accumulation_steps`. A larger effective batch leads to more stable gradient estimates and is often required to match the training dynamics that researchers used when developing best-practice hyperparameters.

**Why you need it.** With `batch_size=2` and `gradient_accumulation=8`, your effective batch size is 16, which is typical for LoRA SFT. You keep VRAM usage low while still getting the benefits of a larger batch.

**Starting value:** `gradient_accumulation_steps=8` → effective batch = 16.

### 3.5 Max Sequence Length (`max_length`)

**What it is.** Sequences longer than this value are truncated. This directly controls VRAM: doubling the sequence length roughly quadruples the memory used for attention.

**How to choose.** Look at the 95th percentile length of your training examples (in tokens). For medical Q&A pairs that are a few sentences each, most will fit inside 512 tokens. Set `max_length` to cover the 95th percentile without being unnecessarily large.

**Starting value:** `max_length=1024` (safe default for short medical Q&A; drop to 512 if VRAM is tight).

---

## 4. LoRA-Specific Hyperparameters

### 4.1 `r` — The Rank

**What it is.** The inner dimension of the two LoRA matrices. A rank of 8 means matrices A and B have 8 columns/rows respectively. Higher rank = more trainable parameters = more capacity, but also more memory and higher risk of overfitting on a small dataset.

**How to think about it.** Rank controls "how many independent directions of change" the adapter can learn. For general-purpose alignment of a large model on a diverse dataset, r=64 or r=128 is common. For focused domain adaptation on a small dataset with a small model, r=8 or r=16 is typically enough and is less likely to overfit.

**Starting value for this project:** `r=16`.

### 4.2 `lora_alpha` — The Scaling Factor

**What it is.** The update applied to the frozen weights is scaled by `lora_alpha / r` before being added. This acts as a kind of "volume knob" for how much the adapter's output influences the model.

**The rule of thumb.** Set `lora_alpha = 2 × r`. With `r=16`, use `lora_alpha=32`. Some practitioners use `lora_alpha = r` (ratio = 1.0) for more conservative updates; the TRL docs show both patterns. The key is that the ratio matters more than the absolute value.

**Starting value:** `lora_alpha=32` (with `r=16`).

### 4.3 `lora_dropout`

**What it is.** Randomly zeros a fraction of the LoRA adapter's activations during training. Standard regularisation to prevent overfitting — same concept as dropout in any neural network.

**How to choose.** With a small dataset, a small amount of dropout (0.05) provides modest regularisation without harming learning. Larger values (0.1) are used when overfitting is observed.

**Starting value:** `lora_dropout=0.05`.

### 4.4 `target_modules`

**What it is.** The list of layer names in the model that will receive LoRA adapters. Everything not listed remains fully frozen.

**How to choose.** Transformer models contain attention layers (typically `q_proj`, `k_proj`, `v_proj`, `o_proj`) and feed-forward layers (`gate_proj`, `up_proj`, `down_proj` in Qwen / Llama architectures). Targeting only attention (`q_proj`, `v_proj`) is the most memory-efficient option and often sufficient. Targeting all linear layers (`"all-linear"` shorthand in PEFT) gives more capacity and is the current community default for high-quality fine-tunes.

```python
# Minimal — memory-efficient, often sufficient for domain adaptation
target_modules=["q_proj", "v_proj"]

# All attention projections
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

# All linear layers — best quality, slightly more VRAM
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

# PEFT shorthand for all linear layers (excludes lm_head)
target_modules="all-linear"
```

**Starting value for this project:** all attention + FFN projections, or `target_modules="all-linear"` for simplicity. Verify that the layer names exist in Qwen3-1.7B by inspecting `model.named_modules()`.

---

## 5. DPOTrainer

### 5.1 What DPO Does

Direct Preference Optimization (DPO) is the preference alignment step that runs **after** SFT. Where SFT teaches the model to produce responses in the right style, DPO refines which of two competing responses it should prefer. Instead of training a separate reward model and then running PPO (which is complex), DPO reformulates the preference problem as a direct classification loss — making it stable and much simpler to run.

The DPO loss is:

```
L_DPO = -log σ( β × (log_ratio_chosen - log_ratio_rejected) )
```

where `log_ratio` = `log P_policy(response | prompt) - log P_reference(response | prompt)`. In plain language: the model is trained to assign higher probability to the preferred response **relative to a frozen reference model**, and penalised for the opposite.

### 5.2 Required Dataset Format

DPO requires a **preference dataset** with exactly three fields:

```python
# Conversational format (recommended — TRL applies chat template automatically)
{
  "prompt": [{"role": "user", "content": "Patient presents with..."}],
  "chosen": [{"role": "assistant", "content": "Triage level: URGENT..."}],
  "rejected": [{"role": "assistant", "content": "Triage level: NON-URGENT..."}]
}

# Plain text format also works
{
  "prompt": "Patient presents with...",
  "chosen": "Triage level: URGENT...",
  "rejected": "Triage level: NON-URGENT..."
}
```

The `prompt` is the shared context. `chosen` is the preferred response. `rejected` is the dispreferred response. The DPOTrainer also accepts an **implicit prompt** format (where the prompt is embedded inside `chosen` and `rejected`), but the explicit format above is recommended.

### 5.3 The Beta Parameter

**What it is.** `beta` (β) controls how tightly the trained model is constrained to stay close to the reference model (the SFT checkpoint). Mathematically, it is the coefficient on a KL-divergence penalty term.

- **High beta (e.g., 0.5):** Strong constraint — the model changes its preferences but barely moves away from the SFT distribution. Safer but slower convergence.
- **Low beta (e.g., 0.01):** Weak constraint — the model can deviate further from the reference to chase the preference signal. Riskier; can cause the model to degrade on general tasks.

**Typical values.** DPOConfig defaults to `beta=0.1`. Community practice for small models on small preference datasets: `0.1` is a safe default. If the `rewards/margins` metric does not grow during training, try lowering to `0.05`. If you see the model's general coherence degrade, try raising to `0.2`.

### 5.4 Reference Model With LoRA

When `peft_config` is passed to DPOTrainer, you **do not** need to provide a separate `ref_model`. TRL automatically uses the frozen base model (before the LoRA updates) as the reference policy. This saves significant VRAM.

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

preference_dataset = load_dataset("your_hf_username/chsa-dpo-dataset", split="train")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

training_args = DPOConfig(
    output_dir="qwen3-1.7b-chsa-dpo",
    num_train_epochs=1,                # DPO datasets are smaller; 1-2 epochs is typical
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,                # ~10x DPO full fine-tune default of 5e-7
    beta=0.1,
    max_length=1024,
    gradient_checkpointing=True,
    bf16=True,
)

trainer = DPOTrainer(
    model="qwen3-1.7b-chsa-sft-lora",  # load your SFT-trained checkpoint
    args=training_args,
    train_dataset=preference_dataset,
    peft_config=peft_config,
)

trainer.train()
```

**Key metrics to watch during DPO training:**
- `rewards/margins`: average gap between chosen and rejected rewards — should increase.
- `rewards/accuracies`: fraction of examples where chosen > rejected — should approach 0.7–0.85 for a clean dataset.
- `loss`: should decrease, but slowly. A very fast drop may indicate overfitting on a tiny preference set.

---

## 6. GRPO: What It Is and When to Pick It

**Group Relative Policy Optimization (GRPO)** is an online reinforcement learning algorithm originally developed for mathematical reasoning (DeepSeekMath, 2024). The key distinction from DPO is that GRPO **generates multiple completions live during training** and scores them with a reward function, rather than learning from a pre-collected preference dataset.

At each step, GRPO samples *G* completions (e.g., 8) for the same prompt, scores each with a reward function (e.g., "is the triage category correct?"), normalises the scores into relative advantages, and updates the policy to increase the probability of higher-reward responses. Unlike PPO, GRPO has no value network — it uses the group's mean reward as the baseline, which is why it is cheaper than PPO but still more expensive than DPO.

**Pick GRPO over DPO when:**
- You have a **programmatic, verifiable reward** — for example, exact-match check against a known correct triage code, JSON schema validation, or a unit test.
- You want the model to discover **reasoning strategies** (like chain-of-thought) rather than just copying preferred completions from a dataset.
- You have enough compute for online generation (typically multi-GPU; slow on a single T4).

**Stick with DPO for this project's POC because:**
- DPO is offline — it trains on a fixed dataset, which is much faster on a single free GPU.
- Building a preference dataset (chosen/rejected pairs) is straightforward from the existing SFT data.
- GRPO on a T4 requires generating 8 completions per prompt in the inner loop, which makes each training step 8–10x slower than SFT.
- GRPO shines for tasks like math where ground-truth is easily verifiable; medical triage preference is more nuanced and benefits from human-annotated preference pairs.

---

## 7. TRL + PEFT vs Unsloth: When to Use Each

Both paths produce the same trained model. The choice is about **speed vs simplicity vs portability**.

### Plain TRL + PEFT

**Pros:**
- Official Hugging Face libraries — maximum documentation, community support, and long-term maintenance.
- Works out of the box on any GPU, including Kaggle T4 and Colab, without custom CUDA kernel compilation.
- Fully transparent: every step is standard PyTorch + Transformers; easy to debug, add custom callbacks, and integrate with Weights & Biases.
- Required if you want to chain SFT → DPO in a single script using TRL's built-in trainers.
- No dependency risk: PEFT and TRL are pure Python on top of standard transformers.

**Cons:**
- Slower than Unsloth (roughly 2–4x on training throughput).
- Higher VRAM usage for the same batch size, because attention kernels are not optimised.

### Unsloth

**Pros:**
- 2–4x faster training and 40–70 % less VRAM via hand-written Triton kernels for attention, cross-entropy, RoPE, and LoRA.
- Compatible with the TRL API — Unsloth wraps `FastLanguageModel.from_pretrained()` and the result drops directly into `SFTTrainer` and `DPOTrainer`.
- Allows larger batch sizes or longer sequences on the same GPU.

**Cons:**
- Custom Triton kernels may fail to compile on older CUDA versions or unusual GPU types — adds a non-trivial dependency.
- Slightly less transparent: when something goes wrong, the error may surface inside a custom kernel rather than standard PyTorch, making debugging harder.
- Unsloth's free tier supports a subset of models; Qwen3 support must be verified for the specific version.
- Breaks occasionally after Transformers updates; you are dependent on the Unsloth team to patch compatibility.

### Recommendation for This Project

**Start with plain TRL + PEFT.** The pipeline will work reliably on Kaggle T4, is fully debuggable, and is easier to explain in the report. Once the end-to-end pipeline is confirmed working, consider switching to Unsloth if training time is a bottleneck — it is a near-drop-in replacement. The TRL docs explicitly document an [Unsloth Integration](https://huggingface.co/docs/trl/main/unsloth_integration) for this exact reason.

---

## 8. Complete Starter Configuration (Summary Table)

| Hyperparameter | SFT Value | DPO Value | Rationale |
|---|---|---|---|
| `learning_rate` | `1e-4` | `5e-6` | 10x LoRA multiplier over TRL defaults |
| `num_train_epochs` | `3` | `1` | Small dataset; avoid overfitting |
| `per_device_train_batch_size` | `2` | `2` | T4 VRAM limit |
| `gradient_accumulation_steps` | `8` | `8` | Effective batch = 16 |
| `max_length` | `1024` | `1024` | Covers 95th %ile of medical Q&A |
| `bf16` | `True` | `True` | Default in SFTConfig / DPOConfig |
| `gradient_checkpointing` | `True` | `True` | Default in both; saves ~30 % VRAM |
| `r` | `16` | `16` | Sufficient for 1.7B domain adapt |
| `lora_alpha` | `32` | `32` | 2 × r rule of thumb |
| `lora_dropout` | `0.05` | `0.05` | Light regularisation |
| `target_modules` | `"all-linear"` | `"all-linear"` | Full coverage, PEFT shorthand |
| `beta` (DPO only) | — | `0.1` | TRL default; conservative start |

---

## 9. Sources

- [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer) — dataset formats, peft_config integration, assistant_only_loss, SFTConfig defaults.
- [TRL DPOTrainer documentation](https://huggingface.co/docs/trl/main/dpo_trainer) — dataset format, beta parameter, loss types, DPOConfig defaults.
- [TRL PEFT Integration documentation](https://huggingface.co/docs/trl/main/en/peft_integration) — learning rate table for all trainers, QLoRA setup, target_modules patterns.
- [TRL GRPO Trainer documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer) — GRPO algorithm, online training, reward functions.
- [PEFT LoraConfig API reference](https://huggingface.co/docs/peft/main/en/package_reference/lora) — full parameter list for LoraConfig.
- [Hugging Face LLM Course — Implementing GRPO in TRL](https://huggingface.co/learn/llm-course/en/chapter12/4) — GRPO tutorial.


---

## Open questions to confirm during implementation

- Qwen3-1.7B-Base has its own chat template baked into the tokenizer — confirm whether eos_token needs to be overridden to '<|im_end|>' in SFTConfig, as the TRL docs warn is required for Qwen2.5 models.
- Which specific linear layers does Qwen3-1.7B expose (q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj)? Run model.named_modules() to confirm before setting target_modules, or use target_modules='all-linear'.
- The DPO reference model is automatically inferred from the initial weights when peft_config is used — verify this holds with Qwen3 by checking that rewards/margins is positive and climbing during the first 100 steps.
- Does the free Kaggle T4 (16 GB) have enough VRAM for DPO with LoRA on Qwen3-1.7B at batch_size=2 + gradient_accumulation=8? DPO internally holds both the policy and reference log-probs simultaneously, which roughly doubles peak activation memory vs SFT.
- TRL's GRPO trainer currently requires generating completions online, which is very slow on a single T4 — is a smaller synthetic reward signal (exact-match triage category) enough to justify the extra GPU time, or should DPO with preference pairs remain the choice for this POC?
