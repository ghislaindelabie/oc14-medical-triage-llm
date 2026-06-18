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
    ("code", "!pip install -q unsloth\n"
     "import subprocess, sys\n"
     "with open('/kaggle/working/requirements-train.lock.txt', 'w') as fh:\n"
     "    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=fh)"),
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
     "SMOKE = True               # first run: 5 steps to validate. Set False for the full run.\n"
     "MODEL = 'Qwen/Qwen3-1.7B-Base'\n"
     "MAX_SEQ_LEN = 1024         # covers ~99% of SFT rows (p99 ~1296 tok)\n"
     "SEED = 3407\n"
     "DATA_DIR = '/kaggle/input/oc14-triage-data'\n"
     "OUT = '/kaggle/working/sft_adapter'\n"
     "HF_REPO = 'ghislaindelabie/oc14-qwen3-1.7b-base-sft-lora'"),
    ("code", "from unsloth import FastLanguageModel\n"
     "model, tokenizer = FastLanguageModel.from_pretrained(\n"
     "    model_name=MODEL, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True, dtype=None)\n"
     "model = FastLanguageModel.get_peft_model(\n"
     "    model, r=16, lora_alpha=16, lora_dropout=0, bias='none',\n"
     "    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],\n"
     "    use_gradient_checkpointing='unsloth', random_state=SEED)\n"
     "print('eos_token:', tokenizer.eos_token, '| id:', tokenizer.eos_token_id)  # expect <|endoftext|>"),
    ("code", "from datasets import load_dataset\n"
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
     "    dataset_text_field='text', max_seq_length=MAX_SEQ_LEN,\n"
     "    per_device_train_batch_size=8, gradient_accumulation_steps=4,\n"
     "    warmup_ratio=0.05, num_train_epochs=2, max_steps=(5 if SMOKE else -1),\n"
     "    learning_rate=2e-4, logging_steps=5, save_steps=50, save_total_limit=2,\n"
     "    optim='adamw_8bit', weight_decay=0.01, lr_scheduler_type='linear',\n"
     "    seed=SEED, output_dir='/kaggle/working/sft_out', report_to=REPORT_TO,\n"
     "    run_name='oc14-sft-qwen3-base')\n"
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


if __name__ == "__main__":
    build(SFT_CELLS, HERE / "oc14-sft-lora" / "oc14-sft-lora.ipynb")
