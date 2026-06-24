"""One-off ablation: quantify the over-triage rule's contribution to the maximale skew.

Relabels a sample of already-labelled triage cases with the over-triage tie-breaker REMOVED
(swapped for a neutral "pick the most likely level"), holding corpus + models + the rest of the
rubric constant, then compares the maximale rate vs the original (over-triage-ON) consensus on the
SAME cases. The shift is attributable to the rule alone (corpus skew is held constant).

    uv run --with openai --with anthropic python scripts/ablation_overtriage.py 100
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from oc14_triage.config import PROCESSED, ROOT
from oc14_triage.labeling.aggregate import Label, consensus, parse_label
from oc14_triage.labeling.clients import available_clients
from oc14_triage.labeling.rubric import SYSTEM_PROMPT, URGENCY_LEVELS, build_user_prompt

load_dotenv(ROOT / ".env")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100

# Build the ablated rubric: drop the explicit over-triage tie-breaker + "dans le doute, sur-trier".
_T1 = ("choisis le PLUS urgent (sur-triage). Un sous-triage est\n"
       "  cliniquement plus dangereux qu'un sur-triage.")
_R1 = "choisis le niveau le plus probable au vu des éléments décrits."
assert SYSTEM_PROMPT.count(_T1) == 1, "over-triage rule text not found verbatim — update _T1"
assert "dans le doute, sur-trier" in SYSTEM_PROMPT
ABLATED = SYSTEM_PROMPT.replace(_T1, _R1).replace("dans le doute, sur-trier", "selon les éléments décrits")
assert "sur-tri" not in ABLATED.lower(), "residual over-triage cue remains"

_lines = (PROCESSED / "triage_labeled.jsonl").read_text(encoding="utf-8").split("\n")
rows = [json.loads(x) for x in _lines if x.strip()]
sample = [r for r in rows if r.get("urgency")][:N]
clients = available_clients()
print(f"ablation on {len(sample)} triage cases | models {[c.name for c in clients]}")


def relabel(r):
    labels: list[Label] = []
    for cl in clients:
        try:
            labels.append(parse_label(cl.name, cl.complete(ABLATED, build_user_prompt(r["text"]))))
        except Exception as e:  # noqa: BLE001
            labels.append(Label(cl.name, False, None, None, False, error=str(e)[:80]))
    return r["case_id"], consensus(r["case_id"], labels)


new = {}
with ThreadPoolExecutor(max_workers=12) as ex:
    for fut in as_completed([ex.submit(relabel, r) for r in sample]):
        cid, con = fut.result()
        new[cid] = con

orig = Counter(r["urgency"] for r in sample)
abl = Counter(new[r["case_id"]].urgency for r in sample if new[r["case_id"]].urgency)
trans = Counter()
for r in sample:
    o, a = r["urgency"], new[r["case_id"]].urgency
    if a and a != o:
        trans[(o, a)] += 1


def pct(d, k):
    t = sum(d.values())
    return round(100 * d.get(k, 0) / t, 1) if t else 0.0


print("\n=== urgency distribution: over-triage ON (original) vs OFF (ablated) ===")
for lv in URGENCY_LEVELS:
    print(f"  {lv:18s}: {pct(orig, lv):5.1f}%  ->  {pct(abl, lv):5.1f}%   (n {orig.get(lv, 0)} -> {abl.get(lv, 0)})")
demotions = sum(n for (o, a), n in trans.items() if o == "urgence maximale")
print(f"\nmaximale shift: {pct(orig, 'urgence maximale')}% -> {pct(abl, 'urgence maximale')}% "
      f"({demotions}/{orig.get('urgence maximale', 0)} maximale cases de-escalated when the rule is removed)")
print("transitions (orig -> ablated):", {f"{o}→{a}": n for (o, a), n in trans.most_common()})

out = {"n": len(sample), "models": [c.name for c in clients],
       "dist_overtriage_on": dict(orig), "dist_overtriage_off": dict(abl),
       "transitions": {f"{o}->{a}": n for (o, a), n in trans.items()}}
(PROCESSED / "_ablation_overtriage.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"\nsaved -> {PROCESSED / '_ablation_overtriage.json'}")
