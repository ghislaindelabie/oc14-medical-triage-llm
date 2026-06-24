"""Parse model answers, check ESI<->3 consistency, build 3-model consensus, Fleiss' kappa.

Pure functions — no network, no SDKs. Unit-tested with synthetic model answers.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

from .rubric import URGENCY_LEVELS, esi_to_urgency

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_URGENCY_CANON = {u.lower(): u for u in URGENCY_LEVELS}


@dataclass(frozen=True)
class Label:
    """One model's parsed answer for one case."""
    model: str
    is_triage_case: bool
    urgency: str | None  # one of URGENCY_LEVELS, or None
    esi: int | None
    consistent: bool  # does esi bucket to urgency?
    justification: str = ""
    red_flags: tuple[str, ...] = ()
    error: str | None = None  # set if parsing failed


def parse_label(model: str, raw: str) -> Label:
    """Parse a model's raw text answer into a Label (robust to code fences / extra prose)."""
    m = _JSON_RE.search(raw or "")
    if not m:
        return Label(model, False, None, None, False, error="no JSON found")
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return Label(model, False, None, None, False, error=f"bad JSON: {e}")
    urg = str(d.get("urgency", "")).strip().lower()
    urgency = _URGENCY_CANON.get(urg)
    esi = d.get("esi")
    try:
        esi = int(esi) if esi is not None else None
    except (ValueError, TypeError):
        esi = None
    is_triage = bool(d.get("is_triage_case", True))
    consistent = (urgency is not None and esi in (1, 2, 3, 4, 5)
                  and esi_to_urgency(esi) == urgency)
    rf = d.get("red_flags") or []
    return Label(model=model, is_triage_case=is_triage, urgency=urgency, esi=esi,
                 consistent=consistent, justification=str(d.get("justification", ""))[:300],
                 red_flags=tuple(str(x) for x in rf)[:10],
                 error=None if urgency else "unparseable urgency")


@dataclass(frozen=True)
class Consensus:
    case_id: str
    labels: tuple[Label, ...]
    urgency: str | None  # majority urgency (None if no majority / not triage)
    esi: int | None  # majority ESI
    unanimous: bool  # all raters agree on urgency
    n_agree: int  # size of the largest urgency bloc
    all_triage: bool
    all_consistent: bool
    is_gold: bool  # unanimous + all consistent + all triage -> held-out gold candidate
    flagged: bool  # something off -> exclude or review


def consensus(case_id: str, labels: list[Label]) -> Consensus:
    votes = [x.urgency for x in labels if x.urgency]
    all_triage = bool(labels) and all(x.is_triage_case for x in labels)
    all_consistent = bool(labels) and all(x.consistent for x in labels)
    complete = len(votes) == len(labels) and len(labels) >= 1
    urgency = esi = None
    n_agree = 0
    unanimous = False
    if votes:
        c = Counter(votes)
        urgency, n_agree = c.most_common(1)[0]
        unanimous = len(c) == 1 and complete
        esis = [x.esi for x in labels if x.urgency == urgency and x.esi]
        esi = Counter(esis).most_common(1)[0][0] if esis else None
    is_gold = bool(unanimous and all_consistent and all_triage and len(labels) >= 3)
    return Consensus(case_id=case_id, labels=tuple(labels), urgency=urgency, esi=esi,
                     unanimous=unanimous, n_agree=n_agree, all_triage=all_triage,
                     all_consistent=all_consistent, is_gold=is_gold,
                     flagged=not (complete and all_triage and unanimous))


def fleiss_kappa(items: list[list[str]], categories: tuple[str, ...] = URGENCY_LEVELS) -> float | None:
    """Fleiss' kappa over items, each a list of n rater labels (same n per item). None if undefined."""
    items = [it for it in items if len(it) >= 2]
    if not items:
        return None
    n = len(items[0])
    if any(len(it) != n for it in items) or n < 2:
        return None
    N = len(items)
    cat = list(categories)
    counts = [[row.count(c) for c in cat] for row in items]
    P_i = [(sum(x * x for x in row) - n) / (n * (n - 1)) for row in counts]
    P_bar = sum(P_i) / N
    p_j = [sum(counts[i][j] for i in range(N)) / (N * n) for j in range(len(cat))]
    P_e = sum(p * p for p in p_j)
    if P_e >= 1.0:
        return 1.0
    return round((P_bar - P_e) / (1 - P_e), 3)


@dataclass
class Aggregate:
    consensuses: list[Consensus] = field(default_factory=list)

    def report(self) -> dict:
        n = len(self.consensuses)
        gold = [c for c in self.consensuses if c.is_gold]
        complete = [[x.urgency for x in c.labels if x.urgency] for c in self.consensuses
                    if len([x for x in c.labels if x.urgency]) == len(c.labels) and len(c.labels) >= 2]
        return {
            "n_cases": n,
            "gold_unanimous": len(gold),
            "majority_only": sum(1 for c in self.consensuses if c.urgency and not c.is_gold),
            "flagged": sum(1 for c in self.consensuses if c.flagged),
            "non_triage": sum(1 for c in self.consensuses if not c.all_triage),
            "fleiss_kappa": fleiss_kappa(complete),
            "gold_urgency_dist": dict(Counter(c.urgency for c in gold)),
        }
