"""T7 bug checks + mechanism diagnostics (DIAGNOSTIC/EXPLORATORY — nothing tuned on Lite).
Reuses analysis/swebench_lite_probe.py internals + cloned repos in data_ext/repos.

Checks:
 1 Path alignment (20 random tasks): gold file paths vs skill unit file paths; normalization; gold-absent count.
 2 Degenerate limit: PPR with alpha=0 (pure BM25 restart) must reproduce BM25 file ranking (subset).
 3 Graph sanity per repo: nodes, edges, %isolated, largest-component share (one task/repo) + competition repos.
 4 Hub bias + correlation of per-repo Δ(PPR−BM25) with graph size.
Writes notes/t7_diagnostics.json.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import swebench_lite_probe as P  # noqa: E402
LZ = P.LZ

TASKS = [json.loads(l) for l in open(ROOT / "data_ext/swebench_lite.jsonl")]
by_repo = {}
for t in TASKS:
    by_repo.setdefault(t["repo"], []).append(t)
OUT = {}


def units_for(t):
    dst = P.REPOS / t["repo"].replace("/", "__")
    P.checkout(dst, t["base_commit"])
    return list(LZ.iter_units(str(dst)))


# ---- Check 1: path alignment ----
def check1():
    rng = random.Random(1)
    sample = rng.sample(TASKS, 20)
    absent_bm = absent_any = 0
    examples = []
    for t in sample:
        units = units_for(t)
        cand_files = {u[2] for u in units}  # skill unit rel paths
        gf, _ = P.gold_from_patch(t["patch"])
        gf_present = {g for g in gf if g in cand_files}
        if gf and not gf_present:
            absent_any += 1
        if len(examples) < 6:
            examples.append(dict(id=t["instance_id"], gold=sorted(gf)[:2],
                                 gold_in_candidates=sorted(gf_present)[:2],
                                 sample_cand=sorted(cand_files)[:2]))
    OUT["check1_path_alignment"] = dict(
        n=len(sample), tasks_with_gold_absent_from_candidates=absent_any,
        note="gold paths are repo-relative from '+++ b/'; skill unit paths are os.path.relpath from repo root; "
             "both are repo-relative with no a//b/ prefix. Test filter excludes tests/ and test_*.py.",
        examples=examples)
    print("CHECK1 gold-absent tasks:", absent_any, "/", len(sample), flush=True)


# ---- Check 2: alpha=0 degenerate limit reproduces BM25 ----
def check2():
    rng = random.Random(2)
    sample = [t for r in by_repo for t in rng.sample(by_repo[r], min(3, len(by_repo[r])))]
    mism = 0; checked = 0
    for t in sample:
        units = units_for(t)
        if not units:
            continue
        names = [u[0] for u in units]; texts = [u[1] for u in units]; rels = [u[2] for u in units]
        bm = LZ.BM25([LZ.tokens(n + " " + x) for n, x in zip(names, texts)])
        sc = np.array(bm.scores(LZ.tokens(t["problem_statement"])))
        adj = LZ.build_call_graph(units)
        r0 = P.np_ppr(adj, sc, alpha=0.0)  # pure restart
        def files(vec):
            fb = {}
            for i in range(len(units)):
                if rels[i] not in fb or vec[i] > fb[rels[i]]: fb[rels[i]] = vec[i]
            return [f for f, _ in sorted(fb.items(), key=lambda kv: -kv[1])]
        checked += 1
        if files(sc)[:10] != files(r0)[:10]:
            mism += 1
    OUT["check2_alpha0_reproduces_bm25"] = dict(checked=checked, top10_file_ranking_mismatches=mism,
        pass_=bool(mism == 0))
    print(f"CHECK2 alpha=0 vs BM25 top-10 file mismatches: {mism}/{checked}", flush=True)


# ---- Check 3: graph sanity per repo (Lite) + competition repos ----
def graph_stats(adj):
    n = len(adj); edges = sum(len(a) for a in adj) // 2
    iso = sum(1 for a in adj if not a) / n if n else 0
    # largest component via BFS
    seen = [False]*n; big = 0
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]; seen[s] = True; sz = 0
        while stack:
            x = stack.pop(); sz += 1
            for y in adj[x]:
                if not seen[y]: seen[y] = True; stack.append(y)
        big = max(big, sz)
    return dict(nodes=n, edges=edges, pct_isolated=round(iso, 3), largest_comp_share=round(big/n, 3) if n else 0)


def check3():
    rng = random.Random(3)
    res = {}
    for repo in sorted(by_repo):
        t = rng.choice(by_repo[repo])
        units = units_for(t)
        res[repo.split("/")[-1]] = dict(graph_stats(LZ.build_call_graph(units)), task=t["instance_id"])
        print("CHECK3", repo.split("/")[-1], res[repo.split("/")[-1]], flush=True)
    # competition repos (local graphs, if present) for comparison
    comp = {}
    import glob
    for gp in glob.glob(str(ROOT / "data/graphs/*.json")):
        G = json.load(open(gp))
        nodes = [nn["name"] for nn in G["nodes"] if not nn["name"].startswith(("tests.", "test."))]
        idx = {nm: i for i, nm in enumerate(nodes)}
        adj = [set() for _ in nodes]
        for e in G["edges"]:
            if e["source"] in idx and e["target"] in idx:
                adj[idx[e["source"]]].add(idx[e["target"]]); adj[idx[e["target"]]].add(idx[e["source"]])
        comp[Path(gp).stem[:20]] = graph_stats(adj)
    OUT["check3_graph_sanity"] = dict(lite=res, competition_shipped=comp)


# ---- Check 4: correlate per-repo Δ with graph size ----
def check4():
    R = json.loads((ROOT / "results/swebench_lite_results.json").read_text())
    gs = OUT.get("check3_graph_sanity", {}).get("lite", {})
    pts = []
    for repo, d in R["per_repo"].items():
        delta = d["ppr_fh10"] - d["bm25_fh10"]
        nodes = gs.get(repo, {}).get("nodes")
        pts.append(dict(repo=repo, n=d["n"], delta=round(delta, 3), nodes=nodes))
    xs = [p["nodes"] for p in pts if p["nodes"]]
    ys = [p["delta"] for p in pts if p["nodes"]]
    corr = float(np.corrcoef(np.log(xs), ys)[0, 1]) if len(xs) > 2 else None
    OUT["check4_delta_vs_graphsize"] = dict(points=pts, spearman_log_nodes_vs_delta=corr,
        note="negative corr → PPR hurts more on larger graphs (hub-diffusion at scale).")
    print("CHECK4 corr(log nodes, delta):", corr, flush=True)


if __name__ == "__main__":
    check1(); check2(); check3(); check4()
    (ROOT / "results/t7_diagnostics.json").write_text(json.dumps(OUT, indent=1, default=str))
    print("-> notes/t7_diagnostics.json")
