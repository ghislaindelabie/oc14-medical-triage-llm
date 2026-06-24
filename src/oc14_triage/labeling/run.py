"""Orchestrate triage labelling, labeler calibration, and dataset assembly.

    # key-free smoke (mock 3 models, 5 cases):
    uv run python -m oc14_triage.labeling.run label --mock --limit 5
    uv run python -m oc14_triage.labeling.run build            # labeled -> eval-gold + SFT-train
    # with real keys in ~/.env:
    uv run python -m oc14_triage.labeling.run calibrate --limit 200   # labeler MCQA floor check
    uv run python -m oc14_triage.labeling.run label --limit 3075      # full labelling
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from ..config import PROCESSED, ROOT, SEED
from ..data.templates import RECO, chat_example, triage_response
from .aggregate import Aggregate, Label, consensus, parse_label
from .cases import load_mcqu_questions, load_triage_cases
from .clients import PRICES, MockClient, available_clients, missing_keys
from .rubric import SYSTEM_PROMPT, build_user_prompt

LABELED = PROCESSED / "triage_labeled.jsonl"
REPORT = PROCESSED / "_triage_label_report.json"
EVAL_GOLD = PROCESSED / "triage_eval_gold.jsonl"
SFT_TRAIN = PROCESSED / "triage_sft_train.jsonl"

# Canned answers for key-free end-to-end tests (cover gold / majority / flagged / inconsistent).
def _j(urg, esi):
    return f'{{"is_triage_case": true, "urgency": "{urg}", "esi": {esi}, "justification": "test {urg}."}}'


_MAX, _MOD, _DIF = _j("urgence maximale", 2), _j("urgence modérée", 3), _j("urgence différée", 5)
_INCONS = _j("urgence maximale", 4)  # esi 4 -> différée, so inconsistent with stated maximale


def _mock_clients() -> list:
    # Cycling lists -> per-case variety: case0 all-MAX (gold), case1 all-MOD (gold),
    # case2 DIF/DIF/inconsistent (majority -> train, flagged), case3 three-way split (flagged).
    return [
        MockClient("openai", [_MAX, _MOD, _DIF, _MAX]),
        MockClient("mistral", [_MAX, _MOD, _DIF, _MOD]),
        MockClient("anthropic", [_MAX, _MOD, _INCONS, _DIF]),
    ]


def _label_one(clients, case) -> tuple:
    """Call every client on one case and return (case, consensus). Thread-pool worker."""
    labels: list[Label] = []
    for cl in clients:
        try:
            raw = cl.complete(SYSTEM_PROMPT, build_user_prompt(case["text"]))
            labels.append(parse_label(cl.name, raw))
        except Exception as e:  # noqa: BLE001 — record + continue (one bad call ≠ abort)
            labels.append(Label(cl.name, False, None, None, False, error=str(e)[:120]))
    return case, consensus(case["case_id"], labels)


def _print_cost(clients, n_cases) -> None:
    total = 0.0
    for cl in clients:
        price = PRICES.get(cl.name)
        if not price or not getattr(cl, "calls", 0):
            continue
        pin, pout = price
        cost = (cl.in_full * pin + cl.in_cread * pin * 0.1 + cl.in_cwrite * pin * 1.25
                + cl.out_tok * pout) / 1e6
        total += cost
        cache = f", cached_read={cl.in_cread:,}" if cl.in_cread else " (no cache hit)"
        print(f"  {cl.name}[{cl.model}]: {cl.calls} calls, in={cl.in_tok:,}{cache} "
              f"out={cl.out_tok:,} -> ${cost:.2f}")
    if total and n_cases:
        print(f"  MEASURED cost this run ({n_cases} cases): ${total:.2f}")
        print(f"  -> extrapolated to 3,075: ${total * 3075 / n_cases:.2f} sync "
              f"/ ${total * 3075 / n_cases / 2:.2f} batch")


def cmd_label(args) -> None:
    cases = load_triage_cases(limit=args.limit)
    print(f"loaded {len(cases)} triage cases")
    if args.dry_run:
        print("\n===== SYSTEM (rubric, cached prefix) =====\n" + SYSTEM_PROMPT)
        for c in cases[:1]:
            print("\n===== USER (per-case) =====\n" + build_user_prompt(c["text"]))
        return
    clients = _mock_clients() if args.mock else available_clients()
    if not clients:
        raise SystemExit(f"No API keys set ({', '.join(missing_keys())}). Use --mock or set keys.")
    print(f"clients: {[c.name for c in clients]}")
    PROCESSED.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    mode = "w"
    if LABELED.exists() and not args.fresh:  # resume — skip already-labelled cases, append
        done = {json.loads(x)["case_id"] for x in LABELED.read_text(encoding="utf-8").split("\n") if x.strip()}
        mode = "a"
    todo = [c for c in cases if c["case_id"] not in done]
    if done:
        print(f"resume: {len(done)} already labelled, {len(todo)} to go")

    agg = Aggregate()
    wlock = threading.Lock()
    n = 0
    with open(LABELED, mode, encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(_label_one, clients, c) for c in todo]):
            c, con = fut.result()
            with wlock:
                agg.consensuses.append(con)
                fh.write(json.dumps({
                    "case_id": c["case_id"], "text": c["text"],
                    "urgency": con.urgency, "esi": con.esi, "unanimous": con.unanimous,
                    "is_gold": con.is_gold, "flagged": con.flagged,
                    "labels": [vars(x) for x in con.labels],
                }, ensure_ascii=False) + "\n")
                n += 1
                if n % 50 == 0:
                    print(f"  {n}/{len(todo)} labelled")
    REPORT.write_text(json.dumps(agg.report(), indent=2, ensure_ascii=False))
    print("report (this run):", json.dumps(agg.report(), ensure_ascii=False))
    _print_cost(clients, len(todo))


def cmd_build(args) -> None:
    """Turn the labelled consensus into a held-out eval-gold set + SFT training rows."""
    if not LABELED.exists():
        raise SystemExit(f"{LABELED} not found — run `label` first.")
    rows = [json.loads(x) for x in LABELED.read_text(encoding="utf-8").split("\n") if x.strip()]
    gold = [r for r in rows if r.get("is_gold")]
    train_src = [r for r in rows if r.get("urgency") and not r.get("is_gold")]
    rng = random.Random(SEED)
    rng.shuffle(gold)
    n_eval = min(len(gold), args.eval_size)
    eval_rows, leftover_gold = gold[:n_eval], gold[n_eval:]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(EVAL_GOLD, "w", encoding="utf-8") as fh:
        for r in eval_rows:
            fh.write(json.dumps({"case_id": r["case_id"], "user": r["text"],
                                 "gold_urgency": r["urgency"], "gold_esi": r["esi"]},
                                ensure_ascii=False) + "\n")
    # SFT training rows: leftover gold + majority cases, rendered in the triage response structure.
    with open(SFT_TRAIN, "w", encoding="utf-8") as fh:
        for r in leftover_gold + train_src:
            lvl = r["urgency"]
            justif = next((x["justification"] for x in r["labels"]
                           if x.get("urgency") == lvl and x.get("justification")), "")
            assistant = triage_response(lvl, justif or "Sur la base du tableau clinique présenté.",
                                        RECO["fr"][lvl], "fr")
            fh.write(json.dumps(chat_example(r["text"], assistant, "fr", "llm_triage", "triage"),
                                ensure_ascii=False) + "\n")
    print(f"eval_gold: {len(eval_rows)} -> {EVAL_GOLD.name} | sft_train: "
          f"{len(leftover_gold) + len(train_src)} -> {SFT_TRAIN.name}")


_LETTER = re.compile(r"\b([A-E])\b")


def cmd_calibrate(args) -> None:
    qs = load_mcqu_questions(limit=args.limit)
    print(f"loaded {len(qs)} MCQU questions (real answer keys)")
    clients = _mock_clients() if args.mock else available_clients()
    if not clients:
        raise SystemExit(f"No API keys set ({', '.join(missing_keys())}). Use --mock or set keys.")
    sys_p = "Réponds à la question médicale à choix unique. Donne UNIQUEMENT la lettre (A-E)."
    scores = {c.name: 0 for c in clients}
    for q in qs:
        for cl in clients:
            try:
                ans = cl.complete(sys_p, q["prompt"])
            except Exception:  # noqa: BLE001
                continue
            m = _LETTER.search((ans or "").strip().upper())
            if m and m.group(1) == q["answer"]:
                scores[cl.name] += 1
    n = len(qs) or 1
    out = {name: round(s / n, 3) for name, s in scores.items()}
    print("MCQU accuracy vs real keys:", json.dumps(out, ensure_ascii=False))
    (PROCESSED / "_labeler_mcqu_calibration.json").write_text(
        json.dumps({"n": len(qs), "accuracy": out}, indent=2, ensure_ascii=False))


def main() -> None:
    load_dotenv(ROOT / ".env")  # project-local keys + OC14_*_MODEL ids
    ap = argparse.ArgumentParser(description="OC14 triage labelling pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_label = sub.add_parser("label")
    p_label.add_argument("--limit", type=int)
    p_label.add_argument("--mock", action="store_true")
    p_label.add_argument("--dry-run", action="store_true")
    p_label.add_argument("--workers", type=int, default=12, help="concurrent cases")
    p_label.add_argument("--fresh", action="store_true", help="ignore existing labels, relabel from scratch")
    p_label.set_defaults(func=cmd_label)

    p_build = sub.add_parser("build")
    p_build.add_argument("--eval-size", type=int, default=300)
    p_build.set_defaults(func=cmd_build)

    p_cal = sub.add_parser("calibrate")
    p_cal.add_argument("--limit", type=int, default=200)
    p_cal.add_argument("--mock", action="store_true")
    p_cal.set_defaults(func=cmd_calibrate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
