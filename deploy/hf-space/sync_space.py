"""Sync the CHSA triage demo to its Hugging Face Space — the single source of truth for the
deploy, used by both `.github/workflows/deploy-space.yml` and manual/local runs.

It mirrors the hand-deploy: the Space-root files (`app.py`, `requirements.txt`, `README.md`)
live in `deploy/hf-space/`, and the runtime package `src/oc14_triage/` is uploaded to the
Space at `src/oc14_triage/` (minus the non-runtime subpackages `data/`, `labeling/`,
`serving/`). Code only — Space secrets/variables (VLLM_API_KEY, VLLM_BASE_URL, …) live on the
Space and are NEVER touched here.

Auth: reads the write-scoped token from the HF_TOKEN env var (a GitHub Actions secret in CI).
The token is never printed. Idempotent: re-running with unchanged files is a no-op commit.
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi

SPACE_REPO_ID = "ghislaindelabie/oc14-triage-demo"

# Files that belong at the Space repo ROOT (Space entrypoint + build config).
_ROOT_DIR = "deploy/hf-space"
_ROOT_FILES = ("app.py", "requirements.txt", "README.md")

# The runtime package, uploaded verbatim to the Space at the same path.
_SRC_PKG = "src/oc14_triage"
# Non-runtime subpackages excluded from the Space (data-prep / labelling / local serving wrapper
# are never imported by app.py's path and would only bloat the image / pull heavy deps).
_EXCLUDE_SUBPKGS = ("data", "labeling", "serving")


def _repo_root() -> str:
    """Project root = two levels up from this file (deploy/hf-space/sync_space.py)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set in the environment.", file=sys.stderr)
        return 1

    root = _repo_root()
    api = HfApi(token=token)

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    operations: list = []

    # 1) Space-root files from deploy/hf-space/.
    for name in _ROOT_FILES:
        path = os.path.join(root, _ROOT_DIR, name)
        if not os.path.isfile(path):
            print(f"ERROR: missing {os.path.join(_ROOT_DIR, name)}", file=sys.stderr)
            return 1
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=path))

    # 2) Runtime package src/oc14_triage/** (skip excluded subpackages and caches).
    pkg_dir = os.path.join(root, _SRC_PKG)
    for dirpath, dirnames, filenames in os.walk(pkg_dir):
        rel_from_pkg = os.path.relpath(dirpath, pkg_dir)
        top = rel_from_pkg.split(os.sep)[0]
        if top in _EXCLUDE_SUBPKGS or "__pycache__" in dirpath:
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            if top in _EXCLUDE_SUBPKGS:
                continue
        for fn in filenames:
            if fn.endswith((".pyc",)) or fn == "__pycache__":
                continue
            abs_path = os.path.join(dirpath, fn)
            rel = os.path.relpath(abs_path, root)  # e.g. src/oc14_triage/agent/ui.py
            operations.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=abs_path))

    uploaded_src = {op.path_in_repo for op in operations if op.path_in_repo.startswith(_SRC_PKG + "/")}

    # 3) Delete any stale files under src/oc14_triage/ on the Space that we are NOT uploading
    #    (e.g. a module removed upstream, or a previously-shipped excluded subpackage).
    try:
        existing = api.list_repo_files(repo_id=SPACE_REPO_ID, repo_type="space")
    except Exception as exc:  # noqa: BLE001 — first-ever deploy: nothing to prune
        print(f"note: could not list existing Space files ({exc}); skipping prune.")
        existing = []
    for f in existing:
        if f.startswith(_SRC_PKG + "/") and f not in uploaded_src:
            operations.append(CommitOperationDelete(path_in_repo=f))

    api.create_commit(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        operations=operations,
        commit_message="ci: sync Space from main (code only; secrets/env untouched)",
    )
    n_add = sum(1 for op in operations if isinstance(op, CommitOperationAdd))
    n_del = sum(1 for op in operations if isinstance(op, CommitOperationDelete))
    print(f"Synced {SPACE_REPO_ID}: {n_add} file(s) uploaded, {n_del} stale file(s) removed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
