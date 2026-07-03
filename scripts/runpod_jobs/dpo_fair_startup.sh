set -uo pipefail
report() { python - "$1" "$2" <<'PY'
import os,sys,json
from huggingface_hub import HfApi
name,payload=sys.argv[1],sys.argv[2]
open("/tmp/"+name,"w").write(payload)
HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj="/tmp/"+name,
  path_in_repo="dpofair/"+name, repo_id="ghislaindelabie/oc14-runpod-results", repo_type="dataset")
print("reported",name)
PY
}
echo "=== OC14 DPO-FAIR (retrain+push+eval BOTH on n=300 canonical harness) $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pip install -q unsloth 2>&1 | tail -2
pip install -q --no-deps "trl==0.22.2" 2>&1 | tail -1
pip install -q "transformers==4.56.2" "huggingface_hub" "wandb" 2>&1 | tail -1
report status_env.json '{"stage":"env_ready"}'

python - <<'PY'
import os, json, time, traceback
t0=time.time()
from huggingface_hub import HfApi, hf_hub_download
RID="ghislaindelabie/oc14-runpod-results"
api=HfApi(token=os.environ["HF_TOKEN"])
def rep(name,obj):
    open("/tmp/"+name,"w").write(json.dumps(obj,ensure_ascii=False))
    api.upload_file(path_or_fileobj="/tmp/"+name,path_in_repo="dpofair/"+name,repo_id=RID,repo_type="dataset")
    print("REPORT",name,str(obj)[:300])

SFT="ghislaindelabie/oc14-qwen3-1.7b-triage-sft"   # merged SFT v9 (served model)
DPO_HF="ghislaindelabie/oc14-qwen3-1.7b-triage-dpo-rpo"  # where we PUSH the DPO adapter (persist)
LEVELS=("urgence maximale","urgence modérée","urgence différée")

try:
    for f in ("dpo_train.jsonl","dpo_val.jsonl","triage_eval_gold.jsonl"):
        hf_hub_download(RID,f"inputs/{f}",repo_type="dataset",local_dir="/data")
    DATA="/data/inputs"

    from unsloth import PatchDPOTrainer
    PatchDPOTrainer()
    from unsloth import FastLanguageModel
    import torch

    # ---------- TRAIN DPO (deterministic seed 3407, balanced pairs, rpo_alpha=1.0) ----------
    model,tok=FastLanguageModel.from_pretrained(model_name=SFT,max_seq_length=1024,load_in_4bit=True)
    model=FastLanguageModel.get_peft_model(model,r=16,lora_alpha=16,lora_dropout=0,bias="none",
      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
      use_gradient_checkpointing="unsloth",random_state=3407)
    rep("status_model.json",{"stage":"model_loaded","secs":round(time.time()-t0,1),
        "eos":tok.eos_token,"chat_template":tok.chat_template is not None})

    from datasets import load_dataset
    ds=load_dataset("json",data_files={"train":f"{DATA}/dpo_train.jsonl","val":f"{DATA}/dpo_val.jsonl"})
    keep={"prompt","chosen","rejected"}
    ds=ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    from trl import DPOConfig, DPOTrainer
    cfg=DPOConfig(beta=0.1,max_length=1024,max_prompt_length=768,
      per_device_train_batch_size=1,gradient_accumulation_steps=8,
      warmup_ratio=0.1,num_train_epochs=3,learning_rate=5e-6,logging_steps=5,
      save_strategy="no",optim="adamw_8bit",weight_decay=0.0,lr_scheduler_type="linear",
      eval_strategy="no",rpo_alpha=1.0,seed=3407,output_dir="/out",
      report_to=("wandb" if os.environ.get("WANDB_API_KEY") else "none"),run_name="oc14-dpo-rpo-fair")
    if os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_PROJECT","oc14-triage-eval")
    tr=DPOTrainer(model=model,ref_model=None,args=cfg,train_dataset=ds["train"],
                  eval_dataset=ds["val"],processing_class=tok)
    st=tr.train()
    rep("status_trained.json",{"stage":"trained","train_secs":round(time.time()-t0,1),
        "metrics":{k:(round(v,4) if isinstance(v,float) else v) for k,v in getattr(st,"metrics",{}).items()}})

    # PUSH the DPO adapter to HF (private) so it PERSISTS + is reproducible
    try:
        model.save_pretrained("/out/dpo_adapter"); tok.save_pretrained("/out/dpo_adapter")
        api.create_repo(DPO_HF, private=True, exist_ok=True)
        model.push_to_hub(DPO_HF, token=os.environ["HF_TOKEN"], private=True)
        tok.push_to_hub(DPO_HF, token=os.environ["HF_TOKEN"])
        rep("status_pushed.json",{"stage":"pushed_adapter","repo":DPO_HF})
    except Exception as e:
        rep("status_push_err.json",{"stage":"push_err","err":str(e)[:300]})

    # ---------- CANONICAL BATCHED EVAL (n=300, max_new_tokens=128, greedy, rep_pen=1.1) ----------
    import re
    from collections import Counter
    VERD=re.compile(r"(?:niveau d.urgence|urgency level)\s*:?\s*(.{0,40})",re.I)
    def extract(t):
        low=(t or "").lower(); m=VERD.search(low)
        if m:
            seg=m.group(1); hits=[(seg.index(l),l) for l in LEVELS if l in seg]
            if hits: return min(hits)[1]
        hits=[(low.index(l),l) for l in LEVELS if l in low]
        return min(hits)[1] if hits else None
    def triage_report(pairs):
        pairs=[(p,g) for p,g in pairs if g in LEVELS]; n=len(pairs)
        rec={};prec={};f1={}
        for lv in LEVELS:
            ng=sum(g==lv for p,g in pairs); npr=sum(p==lv for p,g in pairs)
            tp=sum(p==lv for p,g in pairs if g==lv)
            rec[lv]=round(tp/ng,3) if ng else None; prec[lv]=round(tp/npr,3) if npr else None
            r_,p_=rec[lv],prec[lv]
            f1[lv]=round(2*p_*r_/(p_+r_),3) if (p_ and r_) else (0.0 if (p_==0 or r_==0) else None)
        present=[lv for lv in LEVELS if any(g==lv for p,g in pairs)]
        macro=lambda d: round(sum(d[lv] or 0 for lv in present)/len(present),3) if present else None
        conf=Counter((g,p) for p,g in pairs)
        return {"n":n,"accuracy":round(sum(p==g for p,g in pairs)/n,3),
                "recall_per_level":rec,"precision_per_level":prec,"f1_per_level":f1,
                "macro_recall":macro(rec),"macro_precision":macro(prec),"macro_f1":macro(f1),
                "recall_urgence_maximale":rec["urgence maximale"],
                "confusion_gold_pred":{f'{g}->{p or "(none)"}':c for (g,p),c in
                    sorted(conf.items(),key=lambda kv:(kv[0][0],str(kv[0][1])))}}
    # exact trained FR system prompt (read back from the training data)
    SYSFR=None
    for ln in open(f"{DATA}/dpo_train.jsonl",encoding="utf-8"):
        if ln.strip():
            r=json.loads(ln); SYSFR=r["prompt"][0]["content"]; break
    gold=[json.loads(l) for l in open(f"{DATA}/triage_eval_gold.jsonl") if l.strip()]
    print("eval gold:",len(gold))

    IM_END=tok.convert_tokens_to_ids("<|im_end|>"); STOP=[IM_END,tok.eos_token_id]
    tok.padding_side="left"
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    def gen_batch(users):
        texts=[tok.apply_chat_template([{"role":"system","content":SYSFR},{"role":"user","content":u}],
                 add_generation_prompt=True,enable_thinking=False,tokenize=False) for u in users]
        enc=tok(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(model.device)
        out=model.generate(**enc,max_new_tokens=128,do_sample=False,repetition_penalty=1.1,
                           eos_token_id=STOP,pad_token_id=tok.pad_token_id)
        plen=enc["input_ids"].shape[1]
        return [tok.decode(o[plen:],skip_special_tokens=True).strip() for o in out]

    def eval_full(tag):
        FastLanguageModel.for_inference(model); torch.manual_seed(3407)
        pairs=[]; B=16
        for i in range(0,len(gold),B):
            chunk=gold[i:i+B]; outs=gen_batch([r["user"] for r in chunk])
            for r,txt in zip(chunk,outs): pairs.append((extract(txt),r["gold_urgency"]))
            print(tag,min(i+B,len(gold)),"/",len(gold))
        return triage_report(pairs)

    # DPO eval (adapter currently attached)
    dpo_rep=eval_full("DPO")
    rep("eval_dpo.json",{"model":"sft-v9+dpo-rpo","harness":"canonical-batched","n":dpo_rep["n"],
        "max_new_tokens":128,"greedy":True,"rep_penalty":1.1,**dpo_rep,"secs":round(time.time()-t0,1)})
    print("DPO REPORT",json.dumps(dpo_rep,ensure_ascii=False))

    # v9 eval: disable the adapter -> pure merged SFT v9 (SAME harness, SAME 300, SAME params)
    with model.disable_adapter():
        v9_rep=eval_full("V9")
    rep("eval_v9.json",{"model":"sft-v9-merged","harness":"canonical-batched","n":v9_rep["n"],
        "max_new_tokens":128,"greedy":True,"rep_penalty":1.1,**v9_rep,"secs":round(time.time()-t0,1)})
    print("V9 REPORT",json.dumps(v9_rep,ensure_ascii=False))

    fair={"stage":"fair_done","harness":"canonical-batched n=300 greedy max_new_tokens=128 rep_pen=1.1",
          "v9_macro_f1":v9_rep["macro_f1"],"dpo_macro_f1":dpo_rep["macro_f1"],
          "delta":round((dpo_rep["macro_f1"] or 0)-(v9_rep["macro_f1"] or 0),4),
          "v9_recall":v9_rep["recall_per_level"],"dpo_recall":dpo_rep["recall_per_level"],
          "v9_accuracy":v9_rep["accuracy"],"dpo_accuracy":dpo_rep["accuracy"],
          "dpo_adapter_hf":DPO_HF,"total_secs":round(time.time()-t0,1)}
    rep("results.json",fair)
    print("FAIR RESULT",json.dumps(fair,ensure_ascii=False))
except Exception as e:
    rep("status_error.json",{"stage":"exception","err":str(e)[:300],"trace":traceback.format_exc()[-1600:]})
    raise
PY
echo "=== DPO-FAIR DONE $(date -u) ==="
report status_done.json '{"stage":"done"}'
sleep 20
