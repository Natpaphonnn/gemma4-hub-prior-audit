"""Split-aware localization probe with a two-stage retriever and a dev-only hyperparameter sweep.

Protocol (per Nat): tune every hyperparameter on DEV (analysis/split.json) only; report DEV and EVAL
separately with the dev-selected config; paper numbers come from EVAL. Prior all-124 numbers are
exploratory. p-values on eval use Holm–Bonferroni over a fixed set of comparisons.

Self-contained (numpy + matplotlib only, inline BM25 + PPR); runs locally on ./data or on a Kaggle
notebook with the competition mounted. Reproduce: `.venv/bin/python analysis/probe_v3.py`.
"""
import json
import os
import re
import subprocess
import tarfile
import tempfile
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ paths
def _find_data_dir():
    cands = [Path("/kaggle/input/competitions/gemma-4-developer-agent"),
             Path("/kaggle/input/gemma-4-developer-agent")]
    try:
        cands.append(Path(__file__).resolve().parent.parent / "data")
    except NameError:
        pass
    cands.append(Path("data"))
    for p in cands:
        if (p / "tasks.jsonl").exists():
            return p
    ki = Path("/kaggle/input")
    if ki.exists():
        hits = list(ki.rglob("tasks.jsonl"))
        if hits:
            return hits[0].parent
    return None


DATA_DIR = _find_data_dir()
assert DATA_DIR, "tasks.jsonl not found"
ON_KAGGLE = str(DATA_DIR).startswith("/kaggle")
HERE = Path("/kaggle/working") if ON_KAGGLE else Path(__file__).resolve().parent.parent
OUT_DIR = Path("/kaggle/working") if ON_KAGGLE else (HERE / "results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# split: kernel builds embed it here; locally we read analysis/split.json.
EMBEDDED_SPLIT = None  # <-- kernel build replaces this line with the split literal
SPLIT = EMBEDDED_SPLIT
if SPLIT is None:
    for p in [Path("split.json"), Path("/kaggle/working/split.json"),
              (HERE / "analysis/split.json") if not ON_KAGGLE else Path("split.json")]:
        if p.exists():
            SPLIT = json.loads(p.read_text())
            break
assert SPLIT, "split not found (embed it or provide analysis/split.json)"
DEV, EVAL = set(SPLIT["dev_ids"]), set(SPLIT["eval_ids"])
print(f"DATA_DIR={DATA_DIR} dev={len(DEV)} eval={len(EVAL)}", flush=True)

RNG = np.random.default_rng(0)
N_BOOT = 2000
FILE_KS, FN_KS = (1, 5, 10), (5, 10, 20)
TASKS = [json.loads(l) for l in open(DATA_DIR / "tasks.jsonl")]
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
TEMPLATE_RE = re.compile(r"<!--.*?-->", re.S)
DEF_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")
STOP = set("the and for with that this from are not was but you can use have when into then its will "
           "has all should would could add fix use these those".split())


# ------------------------------------------------------------------ BM25
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1, self.b, self.docs = k1, b, corpus
        self.dl = np.array([len(d) for d in corpus], dtype=float)
        self.avgdl = self.dl.mean() if len(corpus) else 0.0
        self.f, df = [], {}
        for d in corpus:
            tf = {}
            for w in d:
                tf[w] = tf.get(w, 0) + 1
            self.f.append(tf)
            for w in tf:
                df[w] = df.get(w, 0) + 1
        N = len(corpus)
        self.idf = {w: max(1e-6, np.log((N - n + 0.5) / (n + 0.5) + 1)) for w, n in df.items()}

    def scores(self, query):
        q = [w for w in query if w in self.idf]
        s = np.zeros(len(self.docs))
        for i, tf in enumerate(self.f):
            dl = self.dl[i]
            for w in q:
                if w in tf:
                    fr = tf[w]
                    s[i] += self.idf[w] * fr * (self.k1 + 1) / (fr + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


# ------------------------------------------------------------------ ground truth (same as probe_kernel)
MODULE_STRIP = re.compile(r"^(src|docs_src)/")


def file_to_module(path):
    p = MODULE_STRIP.sub("", path)
    p = p[:-3] if p.endswith(".py") else p
    return p.replace("/", ".").removesuffix(".__init__")


def iter_file_hunks(patch):
    path, is_new, old, add, have = None, False, [], [], False
    for line in patch.splitlines():
        if line.startswith("+++ "):
            if have:
                yield path, is_new, old, add
            p = line[4:].strip()
            path = p[2:] if p.startswith("b/") else p
            is_new, old, add, have = False, [], [], True
        elif line.startswith("--- "):
            if "/dev/null" in line:
                is_new = True
        elif have and line.startswith("@@"):
            continue
        elif have and line.startswith("-") and not line.startswith("---"):
            old.append(line[1:])
        elif have and line.startswith("+") and not line.startswith("+++"):
            add.append(line[1:])
        elif have and line.startswith(" "):
            old.append(line[1:])
    if have:
        yield path, is_new, old, add


def distinctive(lines, min_len=8):
    out = []
    for ln in lines:
        s = ln.strip()
        if len(s) >= min_len and s not in (")", "):", "]", "}") and not s.startswith("#"):
            out.append(s)
    return out


def gold_nodes(patch, nodes):
    by_name = {n["name"]: n for n in nodes}
    names = list(by_name)
    gold_fn, gold_mod = set(), set()
    n_files = n_new = 0
    for path, is_new, old, add in iter_file_hunks(patch):
        if not path.endswith(".py"):
            n_files += 1
            continue
        n_files += 1
        if is_new:
            n_new += 1
            continue
        mod = file_to_module(path)
        in_mod = [nm for nm in names if nm == mod or nm.startswith(mod + ".")]
        if not in_mod:
            continue
        gold_mod.add(mod)
        probe = distinctive(old) or distinctive(add)
        matched = {nm for nm in in_mod if any(s in (by_name[nm]["text"] or "") for s in probe)}
        gold_fn |= matched
        if not matched:
            gold_fn |= {nm for nm in in_mod if nm.count(".") == mod.count(".") + 1}
    meta = dict(n_files=n_files, n_new_files=n_new, all_new=(n_new == n_files and n_files > 0))
    return gold_fn, gold_mod, meta


# ------------------------------------------------------------------ helpers
def is_test_node(name):
    return name.startswith(("tests.", "test.", "testing.")) or ".test_" in name or name.split(".")[-1].startswith("test_")


def tokens(text):
    out = []
    for w in IDENT.findall(text):
        for p in re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", w):
            if len(p) > 1:
                out.append(p.lower())
        out.append(w.lower())
    return [t for t in out if t not in STOP]


def under_gold(name, gold_mod):
    return any(name == m or name.startswith(m + ".") for m in gold_mod)


def node_file(name, prefixes):
    segs = name.split(".")
    for i, s in enumerate(segs):
        if s[:1].isupper():
            return ".".join(segs[:i])
    return name if name in prefixes else ".".join(segs[:-1]) if len(segs) > 1 else name


def personalized_pagerank(names, adj, personal, alpha, iters=40):
    idx = {n: i for i, n in enumerate(names)}
    nb = [[idx[m] for m in adj.get(n, ()) if m in idx] for n in names]
    deg = np.array([len(x) for x in nb], dtype=float)
    p = np.maximum(personal, 0)
    p = p / p.sum() if p.sum() > 0 else np.ones(len(names)) / len(names)
    r = p.copy()
    for _ in range(iters):
        push = r / np.maximum(deg, 1)
        acc = np.zeros(len(names))
        for i, neigh in enumerate(nb):
            if neigh:
                acc[neigh] += push[i]
        r = (1 - alpha) * p + alpha * acc
    return r


def rank_metrics(ranked, gold_fn, gold_mod):
    r = {}
    for k in FILE_KS:
        r[f"file_hit@{k}"] = float(any(under_gold(n, gold_mod) for n in ranked[:k]))
    for k in FN_KS:
        r[f"fn_recall@{k}"] = (len(set(ranked[:k]) & gold_fn) / len(gold_fn)) if gold_fn else np.nan
    return r


# ------------------------------------------------------------------ git memory (co-change / churn / commit-msg)
GIT_TMP = "/kaggle/working" if ON_KAGGLE else str(HERE / "results/_gittmp")
Path(GIT_TMP).mkdir(parents=True, exist_ok=True)
GIT_LOG_N = 500  # commits back from HEAD to read


def _git(d, *args):
    return subprocess.run(["git", "-c", "safe.directory=*", "-C", d, *args],
                          capture_output=True, text=True).stdout


def git_signals(instance_id):
    """Extract inference-legal repo memory from the snapshot's git history (HEAD backward).
    Returns (commits, churn) where commits=[{msg, files:[py paths]}], churn=Counter(path->#commits).
    All paths are repo-relative (e.g. src/requests/models.py). Empty if no snapshot/history."""
    snap = DATA_DIR / f"snapshots/{instance_id}.tgz"
    if not snap.exists():
        return [], Counter()
    with tempfile.TemporaryDirectory(dir=GIT_TMP) as tmp:
        try:
            with tarfile.open(snap) as tf:
                gm = [m for m in tf.getmembers() if "/.git/" in m.name or m.name.endswith("/.git")]
                tf.extractall(tmp, members=gm or None)
        except Exception:
            return [], Counter()
        root = None
        for r, dirs, _ in os.walk(tmp):
            if ".git" in dirs:
                root = r
                break
        if root is None:
            return [], Counter()
        log = _git(root, "log", "HEAD", "--name-only", "--pretty=format:\x01%s", "-n", str(GIT_LOG_N))
        commits, cur = [], None
        for line in log.splitlines():
            if line.startswith("\x01"):
                if cur:
                    commits.append(cur)
                cur = {"msg": line[1:], "files": []}
            elif line.endswith(".py") and cur is not None:
                cur["files"].append(line)
        if cur:
            commits.append(cur)
    churn = Counter(f for c in commits for f in c["files"])
    return commits, churn


def repo_path_to_module(path):
    """Map a git repo path (src/requests/models.py, docs_src/x/y.py, rich/table.py) to node-name module."""
    return file_to_module(path)


# ------------------------------------------------------------------ per-task precompute (hyperparam-independent)
def precompute(t):
    key = f'{t["repo"].split("/")[-1]}_{t["base_commit"]}'
    gp, ep = DATA_DIR / f"graphs/{key}.json", DATA_DIR / f"embeddings/{key}.npz"
    if not (gp.exists() and ep.exists()):
        return None
    G = json.load(open(gp))
    Z = np.load(ep)
    nodes0 = G["nodes"]
    gold_fn, gold_mod, meta = gold_nodes(t["patch"], nodes0)
    if not gold_mod or meta["all_new"]:
        return None
    ps = TEMPLATE_RE.sub("", t["problem_statement"])
    idents = set(IDENT.findall(ps))
    # candidates exclude test nodes (shown fair: 0 gold are tests)
    nodes = [n for n in nodes0 if not is_test_node(n["name"])]
    names = [n["name"] for n in nodes]
    texts = [n["text"] or "" for n in nodes]
    idx = {n: i for i, n in enumerate(names)}
    prefixes = {n.rsplit(".", 1)[0] for n in names}
    files = [node_file(n, prefixes) for n in names]
    vecs = np.stack([Z[n] for n in names])
    vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    adj = {}
    for e in G["edges"]:
        if e["source"] in idx and e["target"] in idx:
            adj.setdefault(e["source"], set()).add(e["target"])
            adj.setdefault(e["target"], set()).add(e["source"])
    bm = BM25([tokens(nm + " " + tx) for nm, tx in zip(names, texts)])
    sc = bm.scores(tokens(ps))
    sc_bm = BM25([tokens(nm + " " + tx) for nm, tx in zip(names, texts)])  # reused for stage2
    gfn = gold_fn & set(names)
    sym_in_ps = any(re.search(rf"\b{re.escape(g.split('.')[-1])}\b", ps)
                    for g in gold_fn if not g.split(".")[-1].startswith("__"))

    # ---- git memory: file(module)-level signals, mapped onto nodes via node_file ----
    commits, churn = git_signals(t["instance_id"])
    node_mod = files  # node_file(name) per node (a module string)
    # churn prior: module -> #commits touching it
    churn_mod = defaultdict(float)
    for path, c in churn.items():
        churn_mod[repo_path_to_module(path)] += c
    # commit-message retrieval: BM25 over commit messages, query = statement; a module's score = sum of
    # retrieval score over commits that touched it
    commitmsg_mod = defaultdict(float)
    if commits:
        cm = BM25([tokens(c["msg"]) for c in commits])
        cs = cm.scores(tokens(ps))
        for ci, c in enumerate(commits):
            if cs[ci] <= 0:
                continue
            for path in set(c["files"]):
                commitmsg_mod[repo_path_to_module(path)] += cs[ci]
    # co-change: modules frequently co-committed with the BM25 top-k node files
    top_mods = []
    seen = set()
    for i in np.argsort(-sc):
        m = node_mod[i]
        if m not in seen:
            seen.add(m)
            top_mods.append(m)
        if len(top_mods) >= 5:
            break
    top_set = set(top_mods)
    cochange_mod = defaultdict(float)
    for c in commits:
        mods = {repo_path_to_module(p) for p in c["files"]}
        if mods & top_set:
            for m in mods:
                if m not in top_set:
                    cochange_mod[m] += 1.0

    def mod_vec(dmap):
        return np.array([dmap.get(node_mod[i], 0.0) for i in range(len(names))])

    return dict(id=t["instance_id"], repo=t["repo"].split("/")[-1], names=names, texts=texts,
                idx=idx, files=files, vn=vn, adj=adj, sc=sc, bm=sc_bm, ps=ps, idents=idents,
                gold_fn=gfn, gold_mod=gold_mod, ps_words=len(ps.split()),
                n_files=meta["n_files"], sym_in_ps=sym_in_ps, n_commits=len(commits),
                churn=mod_vec(churn_mod), commitmsg=mod_vec(commitmsg_mod), cochange=mod_vec(cochange_mod))


# ------------------------------------------------------------------ methods (some parametrized)
def rank_bm25(P):
    return [P["names"][i] for i in np.argsort(-P["sc"])]


def rank_emb_seed(P, seed_k=3):
    order = np.argsort(-P["sc"])
    seeds = list(order[:seed_k])
    best = (P["vn"][seeds] @ P["vn"].T).max(0) if seeds else np.zeros(len(P["names"]))
    sc = P["sc"]
    return [P["names"][i] for i in np.argsort(-(0.5 * (sc / (sc.max() or 1)) + 0.5 * best))]


def rank_ppr(P, alpha):
    r = personalized_pagerank(P["names"], P["adj"], P["sc"], alpha)
    return [P["names"][i] for i in np.argsort(-r)], r


def rank_two_stage(P, alpha, N, expand):
    # stage 1: PPR node scores -> file scores (max over nodes) -> top-N files
    _, r = rank_ppr(P, alpha)
    fscore = {}
    for i, f in enumerate(P["files"]):
        if r[i] > fscore.get(f, -1):
            fscore[f] = r[i]
    topfiles = set(sorted(fscore, key=lambda f: -fscore[f])[:N])
    # stage 2: BM25 over nodes within top-N files, optional query expansion with statement symbols
    q = tokens(P["ps"])
    if expand:
        short = {n.split(".")[-1] for n in P["names"]}
        q = q + [w.lower() for w in P["idents"] if w in short]
    sc2 = P["bm"].scores(q)
    pool = [i for i, f in enumerate(P["files"]) if f in topfiles]
    pool.sort(key=lambda i: -sc2[i])
    rest = [i for i in np.argsort(-P["sc"]) if P["files"][i] not in topfiles]
    order = pool + list(rest)
    return [P["names"][i] for i in order]


def _rank_of(scores):
    order = np.argsort(-scores)
    rank = np.empty(len(scores), int)
    rank[order] = np.arange(len(scores))
    return rank


def rank_rrf(P, signals, k=60):
    total = np.zeros(len(P["names"]))
    for s in signals:
        if np.any(s):
            total += 1.0 / (k + _rank_of(s))
    return rank_bm25(P) if not np.any(total) else [P["names"][i] for i in np.argsort(-total)]


def rank_git_fused(P, k=60):
    return rank_rrf(P, [P["sc"], P["churn"], P["commitmsg"], P["cochange"]], k)


def rank_git_stage(P, alpha, N, k=60):
    """Two-stage: stage-1 files ranked by RRF of {PPR, churn, commitmsg, cochange} at file level;
    stage-2 BM25 within top-N files."""
    _, ppr = rank_ppr(P, alpha)
    fileset = list({f for f in P["files"]})
    fidx = {f: i for i, f in enumerate(fileset)}
    # aggregate node signals to file level (max)
    def to_file(vec):
        fs = np.full(len(fileset), 0.0)
        for i, f in enumerate(P["files"]):
            if vec[i] > fs[fidx[f]]:
                fs[fidx[f]] = vec[i]
        return fs
    sigs = [to_file(ppr), to_file(P["churn"]), to_file(P["commitmsg"]), to_file(P["cochange"])]
    tot = np.zeros(len(fileset))
    for s in sigs:
        if np.any(s):
            tot += 1.0 / (k + _rank_of(s))
    topfiles = set(fileset[i] for i in np.argsort(-tot)[:N])
    sc2 = P["bm"].scores(tokens(P["ps"]))
    pool = [i for i, f in enumerate(P["files"]) if f in topfiles]
    pool.sort(key=lambda i: -sc2[i])
    rest = [i for i in np.argsort(-P["sc"]) if P["files"][i] not in topfiles]
    return [P["names"][i] for i in pool + list(rest)]


def rank_oracle(P):
    gm = [n for n in P["names"] if under_gold(n, P["gold_mod"])]
    if not gm:
        return P["names"]
    sub = BM25([tokens(n + " " + P["texts"][P["idx"][n]]) for n in gm])
    oc = [gm[i] for i in np.argsort(-sub.scores(tokens(P["ps"])))]
    return oc


# ------------------------------------------------------------------ eval harness
def boot_ci(vals):
    vals = np.array([v for v in vals if v == v])
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    m = vals[RNG.integers(0, len(vals), (N_BOOT, len(vals)))].mean(1)
    return float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b):
    d = np.array([x - y for x, y in zip(a, b) if (x == x and y == y)])
    if not len(d):
        return None
    bs = d[RNG.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    return dict(delta=float(d.mean()), lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)), p=float(2 * min((bs <= 0).mean(), (bs >= 0).mean())), n=int(len(d)))


def holm(pairs):
    items = sorted([(k, v) for k, v in pairs.items() if v], key=lambda kv: kv[1]["p"])
    m = len(items)
    out = {}
    for rank, (k, v) in enumerate(items):
        adj = min(1.0, v["p"] * (m - rank))
        out[k] = dict(v, p_holm=adj, sig=bool(adj < 0.05))
    return out


def evaluate(P, ranked):
    return rank_metrics(ranked, P["gold_fn"], P["gold_mod"])


def main():
    Ps = [p for p in (precompute(t) for t in TASKS) if p]
    dev = [p for p in Ps if p["id"] in DEV]
    ev = [p for p in Ps if p["id"] in EVAL]
    print(f"precomputed {len(Ps)} (dev {len(dev)}, eval {len(ev)})", flush=True)

    # ---- sweep on DEV ----
    # graph_ppr alpha by file_hit@10
    ppr_grid = [0.3, 0.5, 0.7]
    ppr_dev = {a: np.nanmean([evaluate(p, rank_ppr(p, a)[0])["file_hit@10"] for p in dev]) for a in ppr_grid}
    best_alpha = max(ppr_dev, key=ppr_dev.get)
    # two_stage by fn_recall@10
    ts_grid = list(product([0.3, 0.5, 0.7], [3, 5, 10], [False, True]))
    ts_dev = {}
    for a, N, ex in ts_grid:
        ts_dev[(a, N, ex)] = np.nanmean([evaluate(p, rank_two_stage(p, a, N, ex))["fn_recall@10"] for p in dev])
    best_ts = max(ts_dev, key=ts_dev.get)
    # git_fused RRF k by fn_recall@10; git_stage (alpha,N,k) by fn_recall@10
    k_grid = [15, 30, 60]
    gf_dev = {k: np.nanmean([evaluate(p, rank_git_fused(p, k))["fn_recall@10"] for p in dev]) for k in k_grid}
    best_kf = max(gf_dev, key=gf_dev.get)
    gs_grid = list(product([0.3, 0.5, 0.7], [3, 5, 10], [30, 60]))
    gs_dev = {g: np.nanmean([evaluate(p, rank_git_stage(p, *g))["fn_recall@10"] for p in dev]) for g in gs_grid}
    best_gs = max(gs_dev, key=gs_dev.get)
    # coverage: how many tasks actually had git history
    covered = sum(p["n_commits"] > 5 for p in Ps)
    sweep = {"ppr_alpha_dev_file_hit10": ppr_dev, "best_alpha": best_alpha,
             "best_two_stage": {"alpha": best_ts[0], "N": best_ts[1], "expand": best_ts[2]},
             "git_fused_k_dev": gf_dev, "best_kf": best_kf,
             "git_stage_dev": {str(k): v for k, v in sorted(gs_dev.items(), key=lambda x: -x[1])[:5]},
             "best_git_stage": best_gs,
             "tasks_with_history": covered, "tasks_total": len(Ps)}
    print("SWEEP:", json.dumps(sweep, default=float, indent=1), flush=True)

    a_ts, N_ts, ex_ts = best_ts
    a_gs, N_gs, k_gs = best_gs

    def _grep(P):
        cnt = [sum(1 for w in P["idents"] if w == P["names"][i].split(".")[-1]) for i in range(len(P["names"]))]
        cand = [i for i in range(len(P["names"])) if cnt[i] > 0]
        cand.sort(key=lambda i: (-cnt[i], len(P["names"][i])))
        return [P["names"][i] for i in cand]

    def rankers(P):
        return {
            "bm25": rank_bm25(P),
            "graph_ppr": rank_ppr(P, best_alpha)[0],
            "two_stage": rank_two_stage(P, a_ts, N_ts, ex_ts),
            "git_fused": rank_git_fused(P, best_kf),
            "git_stage": rank_git_stage(P, a_gs, N_gs, k_gs),
            "oracle": rank_oracle(P),
        }

    methods = ["bm25", "graph_ppr", "two_stage", "git_fused", "git_stage", "oracle"]
    metrics = [f"file_hit@{k}" for k in FILE_KS] + [f"fn_recall@{k}" for k in FN_KS]
    cache = {split: {p["id"]: rankers(p) for p in group} for split, group in (("dev", dev), ("eval", ev))}
    Pby = {p["id"]: p for p in Ps}

    def col(split, method, metric):
        group = dev if split == "dev" else ev
        return [evaluate(Pby[p["id"]], cache[split][p["id"]][method])[metric] for p in group]

    report = {"split": {"dev": len(dev), "eval": len(ev)}, "sweep": sweep, "dev": {}, "eval": {}, "paired_eval": {}}
    for split in ("dev", "eval"):
        for m in methods:
            report[split][m] = {met: dict(zip(("mean", "lo", "hi"), boot_ci(col(split, m, met)))) for met in metrics}

    # headline comparisons on EVAL, Holm-corrected (git-memory focus)
    comps = [("git_fused", "bm25", "file_hit@10"), ("git_fused", "bm25", "fn_recall@10"),
             ("git_stage", "bm25", "fn_recall@10"), ("git_stage", "graph_ppr", "fn_recall@10"),
             ("git_stage", "two_stage", "fn_recall@10"), ("git_fused", "bm25", "file_hit@5"),
             ("git_stage", "oracle", "fn_recall@10")]
    raw = {}
    for a, b, met in comps:
        raw[f"{a}-{b}|{met}"] = paired(col("eval", a, met), col("eval", b, met))
    report["paired_eval"] = holm(raw)

    (OUT_DIR / "cochange_results.json").write_text(json.dumps(report, indent=1, default=float))

    # print tables
    for split in ("dev", "eval"):
        print(f"\n=== {split.upper()} (n={report['split'][split]}) ===", flush=True)
        print(f"{'method':10s}" + "".join(f"{m:>14s}" for m in metrics), flush=True)
        for m in methods:
            print(f"{m:10s}" + "".join(f"{report[split][m][met]['mean']:>14.2f}" for met in metrics), flush=True)
    print("\n=== EVAL paired (Holm, 7 comparisons) ===", flush=True)
    for k, v in report["paired_eval"].items():
        print(f"  {k:28s} Δ={v['delta']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] p_holm={v['p_holm']:.3f} {'*' if v['sig'] else ''}", flush=True)
    print("-> cochange_results.json", flush=True)


if __name__ == "__main__":
    main()
