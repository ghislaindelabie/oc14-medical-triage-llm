set -uo pipefail
report() { python - "$1" "$2" <<'PY'
import os,sys,json
from huggingface_hub import HfApi
name,payload=sys.argv[1],sys.argv[2]
open("/tmp/"+name,"w").write(payload)
HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj="/tmp/"+name,
  path_in_repo="dpo/"+name, repo_id="ghislaindelabie/oc14-runpod-results", repo_type="dataset")
print("reported",name)
PY
}
echo "=== OC14 DPO POD START $(date -u) ==="
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
    api.upload_file(path_or_fileobj="/tmp/"+name,path_in_repo="dpo/"+name,repo_id=RID,repo_type="dataset")
    print("REPORT",name,str(obj)[:300])

try:
    for f in ("dpo_train.jsonl","dpo_val.jsonl","triage_eval_gold.jsonl"):
        hf_hub_download(RID,f"inputs/{f}",repo_type="dataset",local_dir="/data")
    DATA="/data/inputs"

    from unsloth import PatchDPOTrainer
    PatchDPOTrainer()
    from unsloth import FastLanguageModel
    SFT="ghislaindelabie/oc14-qwen3-1.7b-triage-sft"   # merged SFT v9 (served model)
    model,tok=FastLanguageModel.from_pretrained(model_name=SFT,max_seq_length=1024,load_in_4bit=True)
    # Merged SFT v9 carries no adapter; add ONE fresh LoRA, train DPO, merge ONCE after.
    # (Ordering invariant: SFT is the frozen base; exactly one adapter, merged post-DPO.)
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
      eval_strategy="no",per_device_eval_batch_size=1,
      rpo_alpha=1.0,   # ANTI-COLLAPSE: NLL/SFT term on chosen -> stops likelihood displacement
      seed=3407,output_dir="/out",
      report_to=("wandb" if os.environ.get("WANDB_API_KEY") else "none"),
      run_name="oc14-dpo-rpo-runpod")
    if os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_PROJECT","oc14-triage-eval")
    tr=DPOTrainer(model=model,ref_model=None,args=cfg,train_dataset=ds["train"],
                  eval_dataset=ds["val"],processing_class=tok)
    st=tr.train()
    rep("status_trained.json",{"stage":"trained","train_secs":round(time.time()-t0,1),
        "metrics":{k:(round(v,4) if isinstance(v,float) else v) for k,v in getattr(st,"metrics",{}).items()}})

    # ---- eval on the 300-case gold (greedy), macro-F1 like the repo metric ----
    FastLanguageModel.for_inference(model)
    IM_END=tok.convert_tokens_to_ids("<|im_end|>")
    LEVELS=("urgence maximale","urgence modérée","urgence différée")
    SYS=("Tu es un assistant de triage médical pour le service des urgences du Centre Hospitalier "
     "Saint-Aurélien (CHSA). Tu assistes le personnel soignant ; tu ne remplaces jamais un "
     "professionnel de santé.\nPour chaque situation décrite, réponds dans la langue de la question "
     "et structure ta réponse ainsi :\n1. Niveau d'urgence : urgence maximale / urgence modérée / "
     "urgence différée.\n2. Justification clinique : explique brièvement les éléments qui motivent ce "
     "niveau.\n3. Recommandation : l'action à entreprendre.\nRègles de sécurité : signale immédiatement "
     "tout signe d'alerte (douleur thoracique, détresse respiratoire, signes neurologiques aigus, etc.) "
     "comme urgence maximale ; ne pose jamais de diagnostic définitif et ne prescris aucun médicament ; "
     "en cas de doute, oriente vers une évaluation médicale. Termine par un bref avertissement rappelant "
     "que cet avis ne remplace pas une consultation médicale.")
    import re, torch
    VERD=re.compile(r"(?:niveau d['’]urgence|urgency level)\s*:?\s*(.{0,40})",re.I)
    def extract(text):
        low=(text or "").lower(); m=VERD.search(low)
        if m:
            seg=m.group(1); hits=[(seg.index(l),l) for l in LEVELS if l in seg]
            if hits: return min(hits)[1]
        found=[(low.index(l),l) for l in LEVELS if l in low]
        return min(found)[1] if found else None
    rows=[json.loads(l) for l in open(f"{DATA}/triage_eval_gold.jsonl") if l.strip()]
    preds=[]
    for i,r in enumerate(rows):
        msgs=[{"role":"system","content":SYS},{"role":"user","content":r["user"]}]
        ids=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt").to(model.device)
        with torch.no_grad():
            out=model.generate(input_ids=ids,max_new_tokens=128,do_sample=False,
                                eos_token_id=[IM_END,tok.eos_token_id])
        txt=tok.decode(out[0][ids.shape[1]:],skip_special_tokens=True)
        preds.append((extract(txt), r["gold_urgency"]))
        if i%60==0: print("eval",i,"/",len(rows))
    pairs=[(p,g) for p,g in preds if g in LEVELS]
    def f1s(pairs):
        rec={};prec={};f1={}
        for lv in LEVELS:
            g=[1 for p,gg in pairs if gg==lv]; pr=[1 for p,gg in pairs if p==lv]
            tp=sum(1 for p,gg in pairs if gg==lv and p==lv)
            rec[lv]=tp/len(g) if g else None; prec[lv]=tp/len(pr) if pr else None
            f1[lv]=(2*prec[lv]*rec[lv]/(prec[lv]+rec[lv]) if (prec[lv] and rec[lv]) else 0.0)
        present=[lv for lv in LEVELS if any(gg==lv for _,gg in pairs)]
        macro=round(sum((f1[lv] or 0) for lv in present)/len(present),4)
        acc=round(sum(1 for p,gg in pairs if p==gg)/len(pairs),4)
        return macro,acc,{k:(round(v,3) if v is not None else None) for k,v in rec.items()},\
               {k:(round(v,3) if v is not None else None) for k,v in f1.items()}
    macro,acc,rec,f1=f1s(pairs)
    result={"stage":"eval_done","macro_f1":macro,"accuracy":acc,"n":len(pairs),
            "recall_per_level":rec,"f1_per_level":f1,"baseline_v9":0.822,
            "delta_vs_v9":round(macro-0.822,4),"rpo_alpha":1.0,
            "total_secs":round(time.time()-t0,1)}
    rep("results.json",result)
    print("RESULT",json.dumps(result,ensure_ascii=False))

    try:
        model.push_to_hub_merged("ghislaindelabie/oc14-qwen3-1.7b-triage-dpo-rpo",tok,
                                 save_method="merged_16bit",token=os.environ["HF_TOKEN"])
        rep("status_pushed.json",{"stage":"pushed","repo":"oc14-qwen3-1.7b-triage-dpo-rpo"})
    except Exception as e:
        rep("status_push_err.json",{"stage":"push_err","err":str(e)[:200]})
except Exception as e:
    rep("status_error.json",{"stage":"exception","err":str(e)[:300],
        "trace":traceback.format_exc()[-1500:]})
    raise
PY
echo "=== DPO DONE $(date -u) ==="
report status_done.json '{"stage":"done"}'
sleep 20
