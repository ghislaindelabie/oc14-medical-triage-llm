#!/usr/bin/env python3
"""Generate the Kaggle training notebooks (reproducible — edit cells here, not the .ipynb).

Emits notebooks/oc14-sft-lora/oc14-sft-lora.ipynb (+ kernel-metadata.json) ready to push:
    uv run --with nbformat python notebooks/build_kaggle_notebooks.py
    ( set -a; . ~/.env; set +a; uv run --with kaggle kaggle kernels push -p notebooks/oc14-sft-lora )

Grounded in the verified facts (docs/research 00 §0b, 01, 02): start from Qwen3-1.7B-Base
(primary), LoRA r=16, read tokenizer.eos_token (do NOT hardcode <|im_end|>), train on
assistant responses only, save the ADAPTER (merge happens after DPO, not here). W&B + HF
are read from Kaggle Secrets with graceful fallback. First run is a 5-step SMOKE.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SFT_CELLS = [
    ("md", """# OC14 — SFT (LoRA) · Qwen3-1.7B-Base · medical-triage assistant

Supervised fine-tuning with Unsloth + LoRA on Kaggle (T4). Reads the private dataset
`oc14-triage-data`. **First run: `SMOKE = True`** → 5 steps (~minutes) to validate the
pipeline and library versions. Then set `SMOKE = False` and re-run for the full ~2 epochs.

Optional Kaggle **Secrets** (Add-ons → Secrets): `WANDB_API_KEY` (loss curves),
`HF_TOKEN` (push the adapter). Without them the notebook still runs and saves the adapter
to the notebook output."""),
    ("code", "# Official Unsloth Kaggle install (verbatim, June 2026): install torch from the cu128\n"
     "# index FIRST so its wheels carry the T4/sm_75 CUDA kernels. Plain `pip install unsloth`\n"
     "# let pip resolve a torch build lacking them -> 'CUDA: no kernel image' at model load.\n"
     "!pip install pip3-autoremove\n"
     "!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128\n"
     "!pip install unsloth\n"
     "!pip install --no-deps --upgrade \"torchao>=0.16.0\"\n"
     "!pip install transformers==4.56.2\n"
     "!pip install --no-deps trl==0.22.2"),
    ("code", "import subprocess, sys\n"
     "with open('/kaggle/working/requirements-train.lock.txt', 'w') as fh:\n"
     "    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=fh)\n"
     "print('lockfile written ->', '/kaggle/working/requirements-train.lock.txt')"),
    ("code", "import os\n"
     "try:\n"
     "    from kaggle_secrets import UserSecretsClient\n"
     "    _us = UserSecretsClient()\n"
     "    for _k in ('WANDB_API_KEY', 'HF_TOKEN'):\n"
     "        try:\n"
     "            os.environ[_k] = _us.get_secret(_k)\n"
     "        except Exception:\n"
     "            pass\n"
     "except Exception:\n"
     "    pass\n"
     "REPORT_TO = 'wandb' if os.environ.get('WANDB_API_KEY') else 'none'\n"
     "HF_TOKEN = os.environ.get('HF_TOKEN')\n"
     "print('W&B:', REPORT_TO, '| HF push:', bool(HF_TOKEN))"),
    ("code", "# ---- config ----\n"
     "SMOKE = False              # full run (~2 epochs). Set True for a ~5-step validation (minutes).\n"
     "MODEL = 'Qwen/Qwen3-1.7B-Base'\n"
     "MAX_SEQ_LEN = 2048         # covers ~99.6% of SFT rows (p99 ~1296 tok); rare longer rows truncated\n"
     "SEED = 3407\n"
     "DATA_DIR = '/kaggle/input/oc14-triage-data'\n"
     "OUT = '/kaggle/working/sft_adapter'\n"
     "HF_REPO = 'ghislaindelabie/oc14-qwen3-1.7b-base-sft-lora'"),
    ("code", "from unsloth import FastLanguageModel\n"
     "model, tokenizer = FastLanguageModel.from_pretrained(\n"
     "    model_name=MODEL, max_seq_length=MAX_SEQ_LEN,\n"
     "    load_in_4bit=True, load_in_8bit=False, full_finetuning=False)\n"
     "model = FastLanguageModel.get_peft_model(\n"
     "    model, r=16, lora_alpha=16, lora_dropout=0, bias='none',\n"
     "    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],\n"
     "    use_gradient_checkpointing='unsloth', random_state=SEED,\n"
     "    use_rslora=False, loftq_config=None)\n"
     "print('eos_token:', tokenizer.eos_token, '| id:', tokenizer.eos_token_id)  # expect <|endoftext|>"),
    ("code", "import glob\n"
     "# Qwen3-1.7B-Base ships NO chat_template -> set a plain ChatML one (Qwen-style, no <think>).\n"
     "# This propagates to the saved tokenizer, so vLLM serving uses the same format later.\n"
     "tokenizer.chat_template = (\n"
     "    \"{% for message in messages %}\"\n"
     "    \"{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}\"\n"
     "    \"{% endfor %}\"\n"
     "    \"{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}\")\n"
     "print('inputs:', os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else 'NONE')\n"
     "_hits = glob.glob('/kaggle/input/**/sft_train.jsonl', recursive=True)\n"
     "assert _hits, 'sft_train.jsonl not found under /kaggle/input — is the dataset attached?'\n"
     "DATA_DIR = os.path.dirname(_hits[0])\n"
     "print('DATA_DIR =', DATA_DIR)\n"
     "from datasets import load_dataset\n"
     "ds = load_dataset('json', data_files={\n"
     "    'train': f'{DATA_DIR}/sft_train.jsonl', 'val': f'{DATA_DIR}/sft_val.jsonl'})\n"
     "def to_text(ex):\n"
     "    return {'text': tokenizer.apply_chat_template(ex['messages'], tokenize=False,\n"
     "                                                  add_generation_prompt=False)}\n"
     "ds = ds.map(to_text, remove_columns=ds['train'].column_names)\n"
     "print(ds)\n"
     "print(ds['train'][0]['text'][:500])"),
    ("code", "from trl import SFTConfig, SFTTrainer\n"
     "args = SFTConfig(\n"
     "    dataset_text_field='text',\n"
     "    per_device_train_batch_size=4, gradient_accumulation_steps=8,\n"
     "    warmup_ratio=0.05, num_train_epochs=2, max_steps=(5 if SMOKE else -1),\n"
     "    learning_rate=2e-4, logging_steps=5, save_steps=50, save_total_limit=2,\n"
     "    optim='adamw_8bit', weight_decay=0.01, lr_scheduler_type='linear',\n"
     "    seed=SEED, output_dir='/kaggle/working/sft_out', report_to=REPORT_TO,\n"
     "    run_name='oc14-sft-qwen3-base', padding_free=False)\n"
     "trainer = SFTTrainer(model=model, tokenizer=tokenizer,\n"
     "                     train_dataset=ds['train'], eval_dataset=ds['val'], args=args)\n"
     "# Loss on assistant turns only (mask system/user) — Qwen3 ChatML markers.\n"
     "from unsloth.chat_templates import train_on_responses_only\n"
     "trainer = train_on_responses_only(\n"
     "    trainer, instruction_part='<|im_start|>user\\n', response_part='<|im_start|>assistant\\n')"),
    ("code", "import time\n"
     "_t = time.time()\n"
     "stats = trainer.train()\n"
     "print('train seconds:', round(time.time() - _t, 1))\n"
     "print(getattr(stats, 'metrics', stats))"),
    ("code", "model.save_pretrained(OUT)\n"
     "tokenizer.save_pretrained(OUT)\n"
     "print('saved adapter ->', OUT)\n"
     "if HF_TOKEN and not SMOKE:\n"
     "    model.push_to_hub(HF_REPO, token=HF_TOKEN)\n"
     "    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)\n"
     "    print('pushed adapter ->', HF_REPO)"),
    ("code", "# Eyeball a couple of triage answers (rough after a SMOKE run — just a sanity check).\n"
     "FastLanguageModel.for_inference(model)\n"
     "SYS = 'Tu es un assistant de triage médical du CHSA. Donne le niveau d\\'urgence, une "
     "justification et une recommandation.'\n"
     "for q in ['Un patient de 60 ans a une douleur thoracique aiguë avec sueurs depuis 20 min.',\n"
     "          'A 30-year-old has mild seasonal allergies and itchy eyes. What do you advise?']:\n"
     "    msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': q}]\n"
     "    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt'\n"
     "                                        ).to(model.device)\n"
     "    out = model.generate(input_ids=ids, max_new_tokens=200, temperature=0.7, do_sample=True)\n"
     "    print('Q:', q)\n"
     "    print(tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True))\n"
     "    print('-' * 70)"),
]


def build(cells, path: Path):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
                for kind, src in cells]
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))
    print(f"wrote {path}  ({len(cells)} cells)")


# --- Quick eval notebook: correct inference (stop on <|im_end|>, trained system prompt) ---
EVAL_CELLS = [
    ("md", """# OC14 — Quick eval of the SFT (Base) model

Loads the LoRA adapter from the SFT kernel's output and generates with the **correct**
inference config: stop on `<|im_end|>` (the small model otherwise runs past the answer into
repetition), and the **exact trained system prompt** (read back from the dataset). Scores the
held-out hand-labelled triage vignettes. Small set (6) — a sanity check, not the final eval."""),
    SFT_CELLS[1],  # identical cu128 install
    SFT_CELLS[2],  # pip-freeze lockfile
    ("code", "import glob, os, json\n"
     "# Prefer the SFT+DPO adapter if attached; else the SFT adapter. (Compare both runs.)\n"
     "ad = (glob.glob('/kaggle/input/**/dpo_adapter/adapter_config.json', recursive=True)\n"
     "      or glob.glob('/kaggle/input/**/sft_adapter/adapter_config.json', recursive=True))\n"
     "assert ad, 'adapter not found — attach the SFT and/or DPO kernel as a kernel source'\n"
     "ADAPTER_DIR = os.path.dirname(ad[0]); print('ADAPTER_DIR =', ADAPTER_DIR)\n"
     "dd = glob.glob('/kaggle/input/**/sft_train.jsonl', recursive=True)\n"
     "DATA_DIR = os.path.dirname(dd[0]) if dd else None\n"
     "# Read the EXACT trained system prompts back from the data (guarantees eval matches training).\n"
     "SYS = {}\n"
     "if DATA_DIR:\n"
     "    for ln in open(f'{DATA_DIR}/sft_train.jsonl', encoding='utf-8').read().split('\\n'):\n"
     "        if not ln.strip():\n"
     "            continue\n"
     "        r = json.loads(ln)\n"
     "        SYS.setdefault(r['lang'], r['messages'][0]['content'])\n"
     "        if len(SYS) == 2:\n"
     "            break\n"
     "print('system prompts loaded for langs:', list(SYS))"),
    ("code", "from unsloth import FastLanguageModel\n"
     "model, tokenizer = FastLanguageModel.from_pretrained(\n"
     "    model_name=ADAPTER_DIR, max_seq_length=2048, load_in_4bit=True)\n"
     "FastLanguageModel.for_inference(model)\n"
     "IM_END = tokenizer.convert_tokens_to_ids('<|im_end|>')\n"
     "STOP_IDS = [IM_END, tokenizer.eos_token_id]\n"
     "print('eos:', tokenizer.eos_token, '| im_end id:', IM_END)\n"
     "def gen(system, user):\n"
     "    msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]\n"
     "    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt'\n"
     "                                        ).to(model.device)\n"
     "    out = model.generate(input_ids=ids, max_new_tokens=256, do_sample=True,\n"
     "                         temperature=0.3, top_p=0.9, repetition_penalty=1.1,\n"
     "                         eos_token_id=STOP_IDS)\n"
     "    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()"),
    ("code", "LEVELS = ('urgence maximale', 'urgence modérée', 'urgence différée')\n"
     "def extract_urgency(t):\n"
     "    low = t.lower(); hits = [(low.index(l), l) for l in LEVELS if l in low]\n"
     "    return min(hits)[1] if hits else None\n"
     "def has_disclaimer(t):\n"
     "    low = t.lower(); return ('ne remplace pas' in low) or ('does not replace' in low)\n"
     "# Held-out hand-labelled vignettes (from vignettes.py, split='eval').\n"
     "VIGN = [\n"
     "  ('fr', 'Homme de 60 ans, malaise avec perte de connaissance brève et pâleur intense.', 'urgence maximale'),\n"
     "  ('fr', 'Adolescent, mal de gorge et fièvre légère depuis hier, sans difficulté à avaler.', 'urgence modérée'),\n"
     "  ('fr', 'Femme de 35 ans, demande de certificat médical pour le sport.', 'urgence différée'),\n"
     "  ('en', '50-year-old, sudden worst-ever headache with vomiting and neck stiffness.', 'urgence maximale'),\n"
     "  ('en', '27-year-old with mild ankle pain after jogging, able to walk.', 'urgence modérée'),\n"
     "  ('en', '40-year-old asking how to renew a stable long-term prescription.', 'urgence différée'),\n"
     "]\n"
     "res = []\n"
     "for lang, user, gold in VIGN:\n"
     "    txt = gen(SYS.get(lang, 'You are a medical triage assistant.'), user)\n"
     "    pred = extract_urgency(txt)\n"
     "    res.append((lang, gold, pred, has_disclaimer(txt), txt))\n"
     "    print(f'[{lang}] gold={gold} | pred={pred}')\n"
     "    print('  ' + txt.replace(chr(10), ' ')[:300]); print('-' * 70)\n"
     "n = len(res)\n"
     "acc = sum(g == p for _, g, p, _, _ in res) / n\n"
     "fmt = sum(p is not None for _, _, p, _, _ in res) / n\n"
     "disc = sum(d for *_, d, _ in res) / n\n"
     "print(f'\\n=== n={n} | urgency_accuracy={acc:.2f} | format_rate={fmt:.2f} | disclaimer_rate={disc:.2f}')"),
]


# --- DPO notebook: align the SFT model to safer answers, then merge once ---
DPO_CELLS = [
    ("md", """# OC14 — DPO on the SFT (Base) model

Direct Preference Optimization on top of the SFT LoRA adapter. **Ordering invariant:** DPO runs on the
SFT model **with the adapter still attached** (`ref_model=None` recovers the reference by disabling the
adapter); the adapter is **merged into the base weights exactly once, after DPO** (the full run).
The SFT adapter is read from the SFT kernel's output (kernel source). **First run: `SMOKE = True`**
(~8 steps). Reads `WANDB_API_KEY`/`HF_TOKEN` from Kaggle Secrets if present."""),
    SFT_CELLS[1],  # cu128 install
    SFT_CELLS[2],  # pip-freeze lockfile
    SFT_CELLS[3],  # secrets -> REPORT_TO, HF_TOKEN
    ("code", "import glob\n"
     "SMOKE = False              # full run (1 epoch + merge once). Set True for an ~8-step validation.\n"
     "SEED = 3407\n"
     "_ad = glob.glob('/kaggle/input/**/sft_adapter/adapter_config.json', recursive=True)\n"
     "assert _ad, 'SFT adapter not found — attach the SFT kernel as a kernel source'\n"
     "SFT_ADAPTER_DIR = os.path.dirname(_ad[0]); print('SFT_ADAPTER_DIR =', SFT_ADAPTER_DIR)\n"
     "_dd = glob.glob('/kaggle/input/**/dpo_train.jsonl', recursive=True)\n"
     "assert _dd, 'dpo_train.jsonl not found'\n"
     "DATA_DIR = os.path.dirname(_dd[0]); print('DATA_DIR =', DATA_DIR)\n"
     "OUT_MERGED = '/kaggle/working/dpo_merged_16bit'\n"
     "HF_REPO = 'ghislaindelabie/oc14-qwen3-1.7b-base-sft-dpo'"),
    ("code", "from unsloth import PatchDPOTrainer\n"
     "PatchDPOTrainer()  # must precede DPOTrainer creation\n"
     "from unsloth import FastLanguageModel\n"
     "# Continue from the SFT adapter (do NOT add fresh LoRA — that would discard SFT).\n"
     "model, tokenizer = FastLanguageModel.from_pretrained(\n"
     "    model_name=SFT_ADAPTER_DIR, max_seq_length=2048, load_in_4bit=True)\n"
     "print('eos:', tokenizer.eos_token, '| chat_template set:', tokenizer.chat_template is not None)"),
    ("code", "from datasets import load_dataset\n"
     "ds = load_dataset('json', data_files={\n"
     "    'train': f'{DATA_DIR}/dpo_train.jsonl', 'val': f'{DATA_DIR}/dpo_val.jsonl'})\n"
     "keep = {'prompt', 'chosen', 'rejected'}\n"
     "ds = ds.remove_columns([c for c in ds['train'].column_names if c not in keep])\n"
     "print(ds)"),
    ("code", "from trl import DPOConfig, DPOTrainer\n"
     "# trl 0.22.2: beta + max_length + max_prompt_length live in DPOConfig; use processing_class=.\n"
     "cfg = DPOConfig(\n"
     "    beta=0.1, max_length=2048, max_prompt_length=1024,\n"
     "    per_device_train_batch_size=2, gradient_accumulation_steps=4,\n"
     "    warmup_ratio=0.1, num_train_epochs=1, max_steps=(8 if SMOKE else -1),\n"
     "    learning_rate=5e-6, logging_steps=5, save_steps=50, save_total_limit=2,\n"
     "    optim='adamw_8bit', weight_decay=0.0, lr_scheduler_type='linear',\n"
     "    seed=SEED, output_dir='/kaggle/working/dpo_out', report_to=REPORT_TO,\n"
     "    run_name='oc14-dpo-qwen3-base')\n"
     "trainer = DPOTrainer(model=model, ref_model=None, args=cfg,\n"
     "                     train_dataset=ds['train'], eval_dataset=ds['val'],\n"
     "                     processing_class=tokenizer)\n"
     "import time\n"
     "_t = time.time(); stats = trainer.train(); print('dpo seconds:', round(time.time() - _t, 1))\n"
     "print(getattr(stats, 'metrics', stats))"),
    ("code", "model.save_pretrained('/kaggle/working/dpo_adapter')\n"
     "tokenizer.save_pretrained('/kaggle/working/dpo_adapter')\n"
     "print('saved DPO adapter')\n"
     "if not SMOKE:\n"
     "    # Merge ONCE, after DPO -> ordinary 16-bit weights for vLLM. Assert it actually wrote files.\n"
     "    model.save_pretrained_merged(OUT_MERGED, tokenizer, save_method='merged_16bit')\n"
     "    files = os.listdir(OUT_MERGED); print('merged files:', files)\n"
     "    assert any(f.endswith('.safetensors') for f in files), 'merge wrote no weights!'\n"
     "    if HF_TOKEN:\n"
     "        model.push_to_hub_merged(HF_REPO, tokenizer, save_method='merged_16bit', token=HF_TOKEN)\n"
     "        print('pushed merged ->', HF_REPO)"),
    ("code", "FastLanguageModel.for_inference(model)\n"
     "IM_END = tokenizer.convert_tokens_to_ids('<|im_end|>')\n"
     "sys_msg = ds['train'][0]['prompt'][0]['content']  # the trained system prompt\n"
     "msgs = [{'role': 'system', 'content': sys_msg},\n"
     "        {'role': 'user', 'content': 'Un patient de 60 ans a une douleur thoracique aiguë avec sueurs depuis 20 min.'}]\n"
     "ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt').to(model.device)\n"
     "out = model.generate(input_ids=ids, max_new_tokens=200, do_sample=True, temperature=0.3,\n"
     "                     top_p=0.9, repetition_penalty=1.1, eos_token_id=[IM_END, tokenizer.eos_token_id])\n"
     "print(tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True))"),
]


if __name__ == "__main__":
    build(SFT_CELLS, HERE / "oc14-sft-lora" / "oc14-sft-lora.ipynb")
    build(EVAL_CELLS, HERE / "oc14-sft-eval" / "oc14-sft-eval.ipynb")
    build(DPO_CELLS, HERE / "oc14-dpo" / "oc14-dpo.ipynb")
