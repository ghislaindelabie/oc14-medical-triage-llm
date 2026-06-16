#!/usr/bin/env python3
"""Build ONE self-contained, mobile-friendly HTML report from the OC14 research
markdown docs and publish it into the vault, where `vault-reader` auto-serves it
on the tailnet (https://p710.tail3089b5.ts.net:8445/doc/oc14-finetune-llm-report).

Canonical "publish an HTML report for phone reading" method on the P710 — see the
global ~/.claude/CLAUDE.md section "Publishing HTML reports".

Run:  uv run --with markdown python3 docs/research/build_report_html.py
"""
import html
import re
from pathlib import Path

import markdown

SRC = Path(__file__).resolve().parent  # docs/research/
# Publish target: a folder in the vault (sibling of the OC13 report). vault-reader
# discovers any *.html under ~/vault and serves it at /doc/<filename-stem>.
OUT = Path("/home/gdelabie/vault/oc14-finetune-llm/OC14-finetune-llm-report.html")

ORDER = [
    "00-OVERALL-APPROACH.md",
    "01-qwen3-1.7b-base-model.md",
    "02-unsloth-sft-dpo-qwen3.md",
    "03-trl-peft-sft-dpo-guide.md",
    "04-oc14-dataset-construction-recipe.md",
    "05-serving-deployment-vllm-runpod.md",
    "06-presidio-anonymization-gdpr-medical-dataset.md",
    "07-reference-notebooks-and-repos.md",
    "08-evaluation-clinical-safety-jama-evidence.md",
    "09-red-team-challenges.md",
]

SHORT = {
    "00-OVERALL-APPROACH.md": "Overall approach (start here)",
    "01-qwen3-1.7b-base-model.md": "Qwen3-1.7B model",
    "02-unsloth-sft-dpo-qwen3.md": "Unsloth (SFT + DPO)",
    "03-trl-peft-sft-dpo-guide.md": "TRL + PEFT guide",
    "04-oc14-dataset-construction-recipe.md": "Dataset recipe",
    "05-serving-deployment-vllm-runpod.md": "Serving & deploy",
    "06-presidio-anonymization-gdpr-medical-dataset.md": "Presidio & GDPR",
    "07-reference-notebooks-and-repos.md": "Reference notebooks",
    "08-evaluation-clinical-safety-jama-evidence.md": "Eval & safety",
    "09-red-team-challenges.md": "Red-team challenges",
}


def first_h1(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def prefix_anchors(s: str, pfx: str) -> str:
    s = re.sub(r'id="([^"]+)"', lambda m: f'id="{pfx}-{m.group(1)}"', s)
    s = re.sub(r'href="#([^"]+)"', lambda m: f'href="#{pfx}-{m.group(1)}"', s)
    return s


def add_backtoc(body: str) -> str:
    """Append a small 'back to Contents' link inside every H2 (section) heading."""
    link = ' <a class="toclink" href="#contents" title="Back to contents">↩ TOC</a>'
    return re.sub(r"(<h2\b[^>]*>)(.*?)(</h2>)",
                  lambda m: f"{m.group(1)}{m.group(2)}{link}{m.group(3)}", body)


docs = []
for i, fname in enumerate(ORDER):
    p = SRC / fname
    if not p.exists():
        continue
    raw = p.read_text(encoding="utf-8")
    pfx = f"doc{i:02d}"
    md = markdown.Markdown(extensions=["extra", "toc", "sane_lists", "admonition"],
                           extension_configs={"toc": {"toc_depth": "2-3"}})
    body = add_backtoc(prefix_anchors(md.convert(raw), pfx))
    sections = []
    for tok in md.toc_tokens:
        if tok.get("level") == 1:
            for k in tok.get("children", []):
                sections.append((k["id"], k["name"]))
        else:
            sections.append((tok["id"], tok["name"]))
    docs.append({
        "id": pfx, "num": fname[:2], "title": first_h1(raw, fname),
        "short": SHORT.get(fname, fname), "body": body, "sections": sections,
    })

contents_items = []
for j, d in enumerate(docs):
    sec_links = "\n".join(
        f'<li><a href="#{d["id"]}-{html.escape(sid)}">{html.escape(name)}</a></li>'
        for sid, name in d["sections"]
    )
    open_attr = " open" if j == 0 else ""
    contents_items.append(f"""
    <details class="toc-doc"{open_attr}>
      <summary><span class="badge">{html.escape(d['num'])}</span> <a class="doc-jump" href="#{d['id']}">{html.escape(d['short'])}</a></summary>
      <ul class="toc-sub">
        {sec_links}
      </ul>
    </details>""")
contents_html = "\n".join(contents_items)

sections_html = []
for d in docs:
    sections_html.append(f"""
  <section class="doc" id="{d['id']}">
    <div class="doc-meta"><span class="badge">{html.escape(d['num'])}</span><a class="totop" href="#contents">⤴ Contents</a></div>
    {d['body']}
  </section>""")
sections_body = "\n<hr class='doc-sep'>\n".join(sections_html)

CSS = """
:root{
  --bg:#ffffff; --fg:#1c1e21; --muted:#6b7280; --line:#e5e7eb; --card:#f8fafc;
  --accent:#1d4ed8; --accent-soft:#eff6ff; --code-bg:#f3f4f6; --warn:#b45309; --warn-bg:#fffbeb;
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#0f1115; --fg:#e6e8eb; --muted:#9aa3af; --line:#262a31; --card:#161a20;
    --accent:#7aa2ff; --accent-soft:#15203a; --code-bg:#11151b; --warn:#fbbf24; --warn-bg:#231d10; }
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);
  font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  word-wrap:break-word;overflow-wrap:break-word}
.wrap{max-width:860px;margin:0 auto;padding:0 18px 120px}
header.top{position:sticky;top:0;z-index:20;background:var(--bg);border-bottom:1px solid var(--line);
  padding:10px 18px;display:flex;align-items:center;gap:12px}
header.top .h{font-weight:700;font-size:15px;flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
header.top a.cbtn{font-size:14px;text-decoration:none;color:#fff;background:var(--accent);
  padding:6px 12px;border-radius:8px;white-space:nowrap}
h1,h2,h3,h4{line-height:1.28;margin:1.5em 0 .5em;scroll-margin-top:60px}
h1{font-size:1.6rem;border-bottom:2px solid var(--line);padding-bottom:.25em}
h2{font-size:1.32rem;border-bottom:1px solid var(--line);padding-bottom:.2em}
h3{font-size:1.12rem}
a{color:var(--accent)}
p,li{font-size:1rem}
code{background:var(--code-bg);padding:.12em .35em;border-radius:5px;font-size:.86em;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--code-bg);border:1px solid var(--line);border-radius:10px;padding:12px;
  overflow-x:auto;-webkit-overflow-scrolling:touch}
pre code{background:none;padding:0;font-size:.82em;white-space:pre}
blockquote{margin:1em 0;padding:.6em 1em;background:var(--warn-bg);border-left:4px solid var(--warn);
  border-radius:0 8px 8px 0;color:var(--fg)}
blockquote p{margin:.3em 0}
table{border-collapse:collapse;width:100%;display:block;overflow-x:auto;
  -webkit-overflow-scrolling:touch;margin:1em 0;font-size:.92rem}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{background:var(--card)}
hr{border:none;border-top:1px solid var(--line);margin:1.5em 0}
hr.doc-sep{border-top:3px solid var(--line);margin:2.5em 0}
.intro{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:18px 0}
.intro h1{border:none;margin:.1em 0 .4em;font-size:1.5rem}
#contents{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:18px 0;scroll-margin-top:60px}
#contents>h2{margin-top:.2em;border:none}
.toc-doc{border-top:1px solid var(--line);padding:6px 0}
.toc-doc:first-of-type{border-top:none}
.toc-doc summary{cursor:pointer;font-weight:600;font-size:1.02rem;list-style:none;display:flex;gap:8px;align-items:center}
.toc-doc summary::-webkit-details-marker{display:none}
.toc-doc summary::before{content:"▸";color:var(--muted);transition:transform .15s}
.toc-doc[open] summary::before{transform:rotate(90deg)}
.doc-jump{text-decoration:none}
.toc-sub{margin:.3em 0 .6em 1.6em;padding:0;list-style:none}
.toc-sub li{margin:.12em 0}
.toc-sub a{text-decoration:none;color:var(--muted);font-size:.93rem}
.toc-sub a:hover{color:var(--accent)}
.badge{display:inline-block;min-width:1.7em;text-align:center;background:var(--accent-soft);color:var(--accent);
  border-radius:6px;padding:1px 6px;font-size:.78rem;font-weight:700;font-variant-numeric:tabular-nums}
.doc-meta{display:flex;align-items:center;gap:10px;margin:.4em 0 0}
a.totop{font-size:.82rem;color:var(--muted);text-decoration:none;margin-left:auto}
a.toclink{font-size:.62em;font-weight:400;text-decoration:none;color:var(--muted);
  margin-left:.5em;vertical-align:middle;opacity:.55;white-space:nowrap}
a.toclink:hover{opacity:1;color:var(--accent)}
.fab{position:fixed;right:16px;bottom:16px;z-index:30;background:var(--accent);color:#fff;
  min-width:46px;height:46px;padding:0 14px;border-radius:23px;display:flex;align-items:center;justify-content:center;
  text-decoration:none;font-size:14px;font-weight:600;box-shadow:0 3px 10px rgba(0,0,0,.25)}
.note{color:var(--muted);font-size:.9rem}
"""

JS = """
var fab=document.getElementById('fab');
addEventListener('scroll',function(){fab.style.opacity=scrollY>400?'1':'0';},{passive:true});
fab.style.opacity='0';fab.style.transition='opacity .2s';
"""

doc_count = len(docs)
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<title>OC14 · Fine-tune your own LLM — Research &amp; Plan</title>
<style>{CSS}</style>
</head>
<body>
<a id="top"></a>
<header class="top">
  <div class="h">OC14 · Medical-triage LLM POC</div>
  <a class="cbtn" href="#contents">Contents</a>
</header>
<main class="wrap">
  <div class="intro" id="overview">
    <h1>OC14 — Fine-tune your own LLM</h1>
    <p><strong>Project:</strong> a proof-of-concept bilingual (FR/EN) medical-triage assistant for a fictional French hospital — built by fine-tuning <strong>Qwen3-1.7B</strong> (SFT + LoRA → DPO), served via <strong>vLLM</strong> on RunPod, wrapped in FastAPI + Docker + GitHub Actions CI/CD, with a 20-page report.</p>
    <p class="note">This page bundles {doc_count} documents: the hardened overall approach, eight topic deep-dives, and the red-team review. Read <strong>00 — Overall approach</strong> first; the rest are reference. Tap <strong>Contents</strong> (top-right) or the <strong>↩ TOC</strong> link on any heading to jump back to navigation.</p>
  </div>

  <nav id="contents">
    <h2>Contents</h2>
    {contents_html}
  </nav>

  {sections_body}
</main>
<a class="fab" id="fab" href="#contents" aria-label="Back to contents">↩ TOC</a>
<script>{JS}</script>
</body>
</html>
"""

# Reconcile dangling intra-doc links (hand-written TOCs vs generated slugs).
all_ids = set(re.findall(r'id="([^"]+)"', page))
def _norm(s): return re.sub(r"-+", "-", s)
_norm_map = {}
for _i in all_ids:
    _norm_map.setdefault(_norm(_i), _i)
def _fix(m):
    target = m.group(1)
    if target in all_ids:
        return m.group(0)
    if _norm(target) in _norm_map:
        return f'href="#{_norm_map[_norm(target)]}"'
    cands = sorted((i for i in all_ids if i.startswith(target)), key=len)
    return f'href="#{cands[0]}"' if cands else m.group(0)
page = re.sub(r'href="#([^"]+)"', _fix, page)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(page, encoding="utf-8")
_dangle = [t for t in set(re.findall(r'href="#([^"]+)"', page)) if t not in all_ids]
print(f"Published {OUT}  ({len(page):,} bytes, {doc_count} docs, {len(_dangle)} dangling anchors)")
