"""Freeze a dev/eval split of the evaluable localization tasks.

Stratified by repo, fixed seed. dev ~1/3 (hyperparameter tuning), eval ~2/3 (touch once, paper
numbers). Writes analysis/split.json. Run ONCE; commit the output; never regenerate with a new seed.

The evaluable task list is regenerated FROM THE COMPETITION DATA (data/tasks.jsonl + data/graphs/):
a task is evaluable if its patch touches an existing (non-new) Python module that is present in the
shipped graph. No derived labels are shipped; the SHA-256 of analysis/split.json (in the README) is
the verification that your regeneration matches ours.

Reproduce: python analysis/make_split.py    (needs ./data with tasks.jsonl + graphs/)
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from ground_truth import gold_nodes  # noqa: E402

SEED = 20260927
DEV_FRACTION = 0.32


def _data_dir():
    for p in (Path("/kaggle/input/competitions/gemma-4-developer-agent"),
              Path("/kaggle/input/gemma-4-developer-agent"), ROOT / "data", Path("data")):
        if (p / "tasks.jsonl").exists():
            return p
    return None


def _evaluable_from_data(d):
    """The evaluable instance_ids, recomputed from the competition data. Returns None if any graph
    is missing (incomplete local data) so the caller can fall back."""
    ids = []
    for line in open(d / "tasks.jsonl"):
        t = json.loads(line)
        key = t["repo"].split("/")[-1] + "_" + t["base_commit"]
        gp = d / f"graphs/{key}.json"
        if not gp.exists():
            return None
        nodes = json.load(open(gp))["nodes"]
        gold_fn, gold_mod, meta, _ = gold_nodes(t["patch"], nodes)
        if gold_mod and not meta["all_new"]:
            ids.append(t["instance_id"])
    return ids


def _evaluable_fallback():
    """Our already-computed evaluable list (used only when local data is incomplete)."""
    f = ROOT / "results/full_probe_tasks.json"
    return [t["id"] for t in json.load(open(f))] if f.exists() else None


def main():
    d = _data_dir()
    ids = _evaluable_from_data(d) if d else None
    src = "data/ (graphs)"
    if not ids:
        ids = _evaluable_fallback()
        src = "results/full_probe_tasks.json (fallback; local data incomplete)"
    assert ids, "need ./data (tasks.jsonl + graphs/) or notes/full_probe_tasks.json to derive the task list"

    by_repo = defaultdict(list)
    for i in ids:
        by_repo[i.rsplit("_", 1)[0]].append(i)  # repo shortname = instance_id prefix
    rng = random.Random(SEED)
    dev, evalset = [], []
    for repo in sorted(by_repo):
        r = sorted(by_repo[repo])
        rng.shuffle(r)
        k = round(DEV_FRACTION * len(r))
        dev += r[:k]
        evalset += r[k:]

    split = {
        "seed": SEED, "dev_fraction": DEV_FRACTION, "stratify": "repo",
        "note": "Hyperparameters (PPR alpha, seed k, BM25 params, two-stage N/top-k, query expansion) "
                "are tuned on dev ONLY. Paper numbers come from eval. The prior all-124 numbers are exploratory.",
        "n_dev": len(dev), "n_eval": len(evalset),
        "dev_ids": sorted(dev), "eval_ids": sorted(evalset),
    }
    (ROOT / "analysis/split.json").write_text(json.dumps(split, indent=1))
    print(f"source: {src}")
    dc = Counter(i.rsplit("_", 1)[0] for i in dev)
    ec = Counter(i.rsplit("_", 1)[0] for i in evalset)
    print(f"dev={len(dev)} eval={len(evalset)}")
    for r in sorted(set(list(dc) + list(ec))):
        print(f"  {r:10s} dev={dc[r]:3d} eval={ec[r]:3d}")
    print("-> analysis/split.json")


if __name__ == "__main__":
    main()
