set -uo pipefail
RID="ghislaindelabie/oc14-runpod-results"
report() { python - "$1" "$2" <<'PY'
import os,sys,json
from huggingface_hub import HfApi
name,payload=sys.argv[1],sys.argv[2]
open("/tmp/"+name,"w").write(payload)
HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj="/tmp/"+name,
  path_in_repo="sweep/"+name, repo_id="ghislaindelabie/oc14-runpod-results", repo_type="dataset")
print("reported",name)
PY
}
trap 'report status_error.json "{\"stage\":\"trap\",\"line\":\"$LINENO\"}"' ERR
echo "=== OC14 SWEEP POD START $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pip install -q unsloth 2>&1 | tail -2
pip install -q --no-deps "trl==0.22.2" 2>&1 | tail -1
pip install -q "transformers==4.56.2" "huggingface_hub" "wandb" 2>&1 | tail -1
report status_env.json '{"stage":"env_ready"}'

python - <<'PY'
import os, json, time
t0=time.time()
from huggingface_hub import HfApi, hf_hub_download
RID="ghislaindelabie/oc14-runpod-results"
api=HfApi(token=os.environ["HF_TOKEN"])
def rep(name,obj):
    open("/tmp/"+name,"w").write(json.dumps(obj,ensure_ascii=False))
    api.upload_file(path_or_fileobj="/tmp/"+name,path_in_repo="sweep/"+name,repo_id=RID,repo_type="dataset")
    print("REPORT",name,obj if len(str(obj))<300 else "...")
for f in ("sft_train.jsonl","sft_val.jsonl"):
    hf_hub_download(RID,f"inputs/{f}",repo_type="dataset",local_dir="/data")
DATA="/data/inputs"

import wandb
WANDB=bool(os.environ.get("WANDB_API_KEY"))
if WANDB:
    wandb.login(key=os.environ["WANDB_API_KEY"])
PROJECT="oc14-sft-sweep"; ENTITY="ghislaindelabie"

# reduced grid over the interesting knobs (a few configs), each a SHORT proxy run
GRID=[
  {"learning_rate":1e-4,"lora_r":8, "warmup_ratio":0.03},
  {"learning_rate":2e-4,"lora_r":16,"warmup_ratio":0.03},
  {"learning_rate":3e-4,"lora_r":16,"warmup_ratio":0.10},
  {"learning_rate":2e-4,"lora_r":32,"warmup_ratio":0.10},
  {"learning_rate":3e-4,"lora_r":8, "warmup_ratio":0.03},
]
MODEL="Qwen/Qwen3-1.7B-Base"; MAX_SEQ=2048; SEED=3407
MAX_STEPS=60; SUBSET=4000
CHAT=("{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + "
      "message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
      "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}")

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

summary=[]
for i,hp in enumerate(GRID):
    tag=f"lr{hp['learning_rate']:.0e}_r{hp['lora_r']}_wu{hp['warmup_ratio']}"
    print(f"=== config {i+1}/{len(GRID)} {tag} ===")
    run=wandb.init(project=PROJECT,entity=ENTITY,name=tag,config=hp,reinit=True) if WANDB else None
    model,tok=FastLanguageModel.from_pretrained(model_name=MODEL,max_seq_length=MAX_SEQ,
        load_in_4bit=True,full_finetuning=False)
    model=FastLanguageModel.get_peft_model(model,r=hp["lora_r"],lora_alpha=hp["lora_r"],
        lora_dropout=0,bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing="unsloth",random_state=SEED)
    tok.chat_template=CHAT
    ds=load_dataset("json",data_files={"train":f"{DATA}/sft_train.jsonl","val":f"{DATA}/sft_val.jsonl"})
    ds["train"]=ds["train"].select(range(min(SUBSET,len(ds["train"]))))
    ds["val"]=ds["val"].select(range(min(400,len(ds["val"]))))
    def to_text(ex): return {"text":tok.apply_chat_template(ex["messages"],tokenize=False,add_generation_prompt=False)}
    ds=ds.map(to_text,remove_columns=ds["train"].column_names)
    args=SFTConfig(dataset_text_field="text",per_device_train_batch_size=4,
        gradient_accumulation_steps=8,warmup_ratio=hp["warmup_ratio"],max_steps=MAX_STEPS,
        learning_rate=hp["learning_rate"],logging_steps=5,save_strategy="no",
        optim="adamw_8bit",weight_decay=0.01,lr_scheduler_type="linear",seed=SEED,
        output_dir=f"/out/{tag}",report_to=("wandb" if WANDB else "none"),run_name=tag,
        eval_strategy="steps",eval_steps=15,per_device_eval_batch_size=4,padding_free=False)
    trainer=SFTTrainer(model=model,tokenizer=tok,train_dataset=ds["train"],
        eval_dataset=ds["val"],args=args)
    trainer=train_on_responses_only(trainer,instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n")
    st=trainer.train()
    ev=trainer.evaluate()
    entry={"config":hp,"tag":tag,"eval_loss":round(float(ev.get("eval_loss",float("nan"))),4),
           "train_loss":round(float(getattr(st,"metrics",{}).get("train_loss",float("nan"))),4)}
    summary.append(entry)
    if run:
        wandb.log({"final_eval_loss":entry["eval_loss"]}); run.finish()
    rep("progress.json",{"done":i+1,"total":len(GRID),"so_far":summary})
    del model,trainer
    import torch,gc; gc.collect(); torch.cuda.empty_cache()

summary.sort(key=lambda e:e["eval_loss"])
rep("results.json",{"stage":"sweep_done","project":PROJECT,"entity":ENTITY,
    "n_configs":len(summary),"ranked":summary,"best":summary[0] if summary else None,
    "total_secs":round(time.time()-t0,1)})
print("SWEEP RESULT",json.dumps(summary,ensure_ascii=False))
PY
echo "=== SWEEP DONE $(date -u) ==="
report status_done.json '{"stage":"done"}'
sleep 30
