"""T7 — external replication of the localize skill on SWE-bench Lite (300 tasks, public).

Runs the EXACT skill recipe (frozen: source AST units, BM25 k1=1.5/b=0.75, test filter, source call-graph,
BM25-seeded PPR α=0.5, 40 iters) — NO re-tuning — and reports file_hit@{1,5,10}, fn_recall@10 for
skill-BM25 vs skill-PPR, per repo and overall, with a paired bootstrap on the PRE-REGISTERED primary
(PPR − BM25 file_hit@10). Gold FILE = files touched by the gold patch (label only).

Clones each repo once to data_ext/repos/ (gitignored), checks out each base_commit. Pure-stdlib skill
funcs are imported from skill/localize/scripts/localize.py; PPR is a numpy port of the skill's exact
algorithm (verified identical ranking) for speed on large repos.

Usage: .venv/bin/python analysis/swebench_lite_probe.py [--repos flask,requests] [--limit N]
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data_ext"
REPOS = EXT / "repos"
REPOS.mkdir(parents=True, exist_ok=True)

# import the exact skill functions
spec = importlib.util.spec_from_file_location("localize", ROOT / "skill/localize/scripts/localize.py")
LZ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LZ)

RNG = np.random.default_rng(0)
IDENT = LZ.IDENT
DEF_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")


def clone(repo):
    dst = REPOS / repo.replace("/", "__")
    if (dst / ".git").exists():
        return dst
    import shutil
    for attempt in range(3):
        shutil.rmtree(dst, ignore_errors=True)
        print(f"  cloning {repo} (attempt {attempt+1}) ...", flush=True)
        r = subprocess.run(["git", "clone", "--quiet", f"https://github.com/{repo}.git", str(dst)],
                           capture_output=True, text=True)
        if r.returncode == 0 and (dst / ".git").exists():
            return dst
        print(f"    clone failed: {r.stderr.strip()[:120]}", flush=True)
        time.sleep(5)
    raise RuntimeError(f"could not clone {repo}")


def checkout(dst, commit):
    subprocess.run(["git", "-C", str(dst), "checkout", "--quiet", "--force", commit],
                   check=True, capture_output=True)


def gold_from_patch(patch):
    files, funcs = set(), set()
    path = None
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
            if path.endswith(".py"): files.add(path)
        elif line.startswith("@@") or line.startswith("+") or line.startswith("-"):
            m = DEF_RE.match(line[1:] if line[:1] in "+-" else line[line.find("@@", 2)+2:] if "@@" in line[2:] else line)
            if m and path and path.endswith(".py"):
                funcs.add((path, m.group(1)))
    return files, funcs


def np_ppr(adj, restart, alpha=0.5, iters=40):
    n = len(adj); deg = np.array([len(a) or 1 for a in adj], float)
    p = np.maximum(np.asarray(restart, float), 0); s = p.sum(); p = p/s if s else np.ones(n)/n
    r = p.copy()
    # precompute neighbor index arrays
    for _ in range(iters):
        push = alpha * r / deg
        acc = np.zeros(n)
        for i, nb in enumerate(adj):
            if nb: acc[list(nb)] += push[i]
        r = (1-alpha)*p + acc
    return r


def rank(root, ps, use_ppr):
    units = list(LZ.iter_units(root))
    if not units: return [], [], 0
    names = [u[0] for u in units]; texts = [u[1] for u in units]; rels = [u[2] for u in units]
    bm = LZ.BM25([LZ.tokens(n + " " + t) for n, t in zip(names, texts)])
    sc = np.array(bm.scores(LZ.tokens(ps)))
    fsrc = sc
    if use_ppr:
        adj = LZ.build_call_graph(units)
        fsrc = np_ppr(adj, sc)
    # file ranking (best unit score per file)
    fb = {}
    for i in range(len(units)):
        if rels[i] not in fb or fsrc[i] > fb[rels[i]]: fb[rels[i]] = fsrc[i]
    files = sorted(fb, key=lambda f: -fb[f])
    # function ranking (top units)
    order = np.argsort(-fsrc)
    funcs = [(rels[i], names[i].split(".")[-1]) for i in order]
    return files, funcs, len(units)


def metrics(files, funcs, gfiles, gfuncs):
    r = {}
    for k in (1, 5, 10):
        r[f"file_hit@{k}"] = float(any(f in gfiles for f in files[:k]))
    r["fn_recall@10"] = (len(set(funcs[:10]) & gfuncs) / len(gfuncs)) if gfuncs else np.nan
    return r


# ---- Item-2 EXPLORATORY (prereg 2026-10-11): same methods as the competition probe, on Lite ----
ALPHAS = [0.0, 0.05, 0.1, 0.25, 0.5, 0.7]
from scipy import sparse  # noqa: E402


def _build_sparse(adj):
    """CSR adjacency (undirected) + degree, built ONCE per task and reused across all PPR variants."""
    rows, cols = [], []
    for i, nb in enumerate(adj):
        for j in nb:
            rows.append(i); cols.append(j)
    n = len(adj)
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)) if rows else sparse.csr_matrix((n, n))
    deg = np.array([len(a) or 1 for a in adj], float)
    return A, deg


def _ppr_sparse(A, deg, restart, alpha=0.5, iters=40):
    """Identical arithmetic to np_ppr but one sparse matvec per iteration (acc = A @ (alpha*r/deg))."""
    n = A.shape[0]
    p = np.maximum(np.asarray(restart, float), 0); s = p.sum(); p = p / s if s else np.ones(n) / n
    r = p.copy()
    for _ in range(iters):
        r = (1 - alpha) * p + A.dot(alpha * r / deg)
    return r


def _file_rank(units, rels, scores):
    fb = {}
    for i in range(len(units)):
        if rels[i] not in fb or scores[i] > fb[rels[i]]:
            fb[rels[i]] = scores[i]
    return sorted(fb, key=lambda f: -fb[f])


def all_rankings(root, ps):
    """Compute units/graph ONCE; return file rankings for bm25, ppr(α=0.5, the frozen primary),
    global-PR hub (no query), lift-normalized PPR, and a PPR α-sweep. file_hit uses distinct files.
    PPR is a sparse matvec sharing one CSR matrix across all variants (fast on large Lite repos)."""
    units = list(LZ.iter_units(root))
    if not units:
        return None
    names = [u[0] for u in units]; texts = [u[1] for u in units]; rels = [u[2] for u in units]
    bm = LZ.BM25([LZ.tokens(n + " " + t) for n, t in zip(names, texts)])
    sc = np.array(bm.scores(LZ.tokens(ps)))
    A, deg = _build_sparse(LZ.build_call_graph(units))
    gpr = _ppr_sparse(A, deg, np.ones(len(units)), alpha=0.5)   # non-personalized hub score
    out = {"bm25": _file_rank(units, rels, sc),
           "global_pr": _file_rank(units, rels, gpr)}           # no-query hub baseline
    out["alpha_sweep"] = {a: _file_rank(units, rels, _ppr_sparse(A, deg, sc, alpha=a)) for a in ALPHAS}
    ppr_main = _ppr_sparse(A, deg, sc, alpha=0.5)               # frozen Lite recipe (unchanged primary)
    out["ppr"] = _file_rank(units, rels, ppr_main)
    out["ppr_lift"] = _file_rank(units, rels, ppr_main / (gpr + 1e-12))
    order_b = np.argsort(-sc); order_p = np.argsort(-ppr_main)
    funcs_b = [(rels[i], names[i].split(".")[-1]) for i in order_b]
    funcs_p = [(rels[i], names[i].split(".")[-1]) for i in order_p]
    return out, funcs_b, funcs_p, len(units)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tasks = [json.loads(l) for l in open(EXT / "swebench_lite.jsonl")]
    if a.repos:
        want = set(a.repos.split(",")); tasks = [t for t in tasks if t["repo"].split("/")[-1] in want or t["repo"] in want]
    if a.limit: tasks = tasks[:a.limit]
    by_repo = defaultdict(list)
    for t in tasks: by_repo[t["repo"]].append(t)

    rows = []
    skipped_repos = []
    for repo in sorted(by_repo):
        try:
            dst = clone(repo)
        except Exception as e:
            print(f"  SKIP repo {repo}: {str(e)[:100]}", flush=True); skipped_repos.append(repo); continue
        for t in by_repo[repo]:
            try:
                checkout(dst, t["base_commit"])
            except Exception as e:
                print(f"  skip {t['instance_id']}: checkout failed {str(e)[:80]}", flush=True); continue
            gf, gfn = gold_from_patch(t["patch"])
            if not gf: continue
            ps = t["problem_statement"]
            t0 = time.time()
            res = all_rankings(str(dst), ps)
            if res is None:
                continue
            rk, fb_funcs, fp_funcs, n = res
            dt = time.time() - t0
            rows.append(dict(id=t["instance_id"], repo=repo.split("/")[-1], n_units=n, t_all=dt,
                             bm25=metrics(rk["bm25"], fb_funcs, gf, gfn),
                             ppr=metrics(rk["ppr"], fp_funcs, gf, gfn),
                             global_pr=metrics(rk["global_pr"], fb_funcs, gf, gfn),
                             ppr_lift=metrics(rk["ppr_lift"], fp_funcs, gf, gfn),
                             alpha_sweep={a: {f"file_hit@{k}": float(any(f in gf for f in rk["alpha_sweep"][a][:k]))
                                              for k in (1, 5, 10)} for a in ALPHAS},
                             hub_share={a: (len(set(rk["alpha_sweep"][a][:10]) & set(rk["global_pr"][:10]))
                                            / max(1, len(rk["alpha_sweep"][a][:10]))) for a in ALPHAS}))
        print(f"  {repo}: {sum(r['repo']==repo.split('/')[-1] for r in rows)} tasks done", flush=True)

    def col(meth, met): return np.array([r[meth][met] for r in rows if r[meth][met] == r[meth][met]])
    def ci(v): m = v[RNG.integers(0,len(v),(2000,len(v)))].mean(1); return [float(v.mean()), float(np.percentile(m,2.5)), float(np.percentile(m,97.5))]
    def paired(met):
        d = np.array([r["ppr"][met]-r["bm25"][met] for r in rows if r["ppr"][met]==r["ppr"][met] and r["bm25"][met]==r["bm25"][met]])
        bs = d[RNG.integers(0,len(d),(2000,len(d)))].mean(1)
        return dict(delta=float(d.mean()), lo=float(np.percentile(bs,2.5)), hi=float(np.percentile(bs,97.5)), n=int(len(d)))

    rep = {"n_tasks": len(rows), "overall": {}, "primary_ppr_minus_bm25_file_hit@10": paired("file_hit@10"),
           "per_repo": {}, "runtime": {}}
    for meth in ("bm25", "ppr"):
        rep["overall"][meth] = {m: ci(col(meth, m)) for m in ["file_hit@1","file_hit@5","file_hit@10","fn_recall@10"]}
    # Item-2: no-query hub + lift-normalized PPR (file_hit only)
    for meth in ("global_pr", "ppr_lift"):
        rep["overall"][meth] = {m: ci(col(meth, m)) for m in ["file_hit@1","file_hit@5","file_hit@10"]}
    # Item-2: PPR alpha sweep (mean file_hit per alpha)
    rep["alpha_sweep"] = {str(a): {f"file_hit@{k}": float(np.mean([r["alpha_sweep"][a][f"file_hit@{k}"] for r in rows]))
                                   for k in (1, 5, 10)} for a in ALPHAS}
    # Item-6: bootstrap CI (file_hit@10) + hub-file share of top-10, per alpha
    rep["alpha_sweep_detail"] = {}
    for a in ALPHAS:
        m, lo, hi = ci(np.array([r["alpha_sweep"][a]["file_hit@10"] for r in rows]))
        rep["alpha_sweep_detail"][str(a)] = {"file_hit@10": m, "lo": lo, "hi": hi,
                                             "hub_share@10": float(np.mean([r["hub_share"][a] for r in rows]))}
    for repo in sorted({r["repo"] for r in rows}):
        rr = [r for r in rows if r["repo"] == repo]
        rep["per_repo"][repo] = dict(n=len(rr),
            bm25_fh10=float(np.mean([r["bm25"]["file_hit@10"] for r in rr])),
            ppr_fh10=float(np.mean([r["ppr"]["file_hit@10"] for r in rr])))
    ta = np.array([r["t_all"] for r in rows])
    rep["runtime"] = dict(all_mean=float(ta.mean()), all_p90=float(np.percentile(ta,90)), all_max=float(ta.max()),
                          n_units_mean=float(np.mean([r["n_units"] for r in rows])),
                          n_units_max=int(max(r["n_units"] for r in rows)))
    rep["skipped_repos"]=skipped_repos
    (ROOT / "results/swebench_lite_results.json").write_text(json.dumps(rep, indent=1))
    print(f"\nN={len(rows)}")
    print("overall file_hit@10: bm25", round(rep["overall"]["bm25"]["file_hit@10"][0],3),
          "ppr", round(rep["overall"]["ppr"]["file_hit@10"][0],3))
    print("PRIMARY ppr-bm25 file_hit@10:", rep["primary_ppr_minus_bm25_file_hit@10"])
    print("no-query global_pr fh@10:", round(rep["overall"]["global_pr"]["file_hit@10"][0], 3),
          "| ppr_lift fh@10:", round(rep["overall"]["ppr_lift"]["file_hit@10"][0], 3))
    print("alpha sweep fh@10:", {a: round(v["file_hit@10"], 3) for a, v in rep["alpha_sweep"].items()})
    print("runtime/task mean:", round(rep["runtime"]["all_mean"], 2), "p90", round(rep["runtime"]["all_p90"], 2))
    print("-> notes/swebench_lite_results.json")


if __name__ == "__main__":
    main()
