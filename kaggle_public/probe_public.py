"""Split-aware localization probe with a two-stage retriever and a dev-only hyperparameter sweep.

Protocol (per Nat): tune every hyperparameter on DEV (analysis/split.json) only; report DEV and EVAL
separately with the dev-selected config; paper numbers come from EVAL. Prior all-124 numbers are
exploratory. p-values on eval use Holm–Bonferroni over a fixed set of comparisons.

Self-contained (numpy + matplotlib only, inline BM25 + PPR); runs locally on ./data or on a Kaggle
notebook with the competition mounted. Reproduce: `.venv/bin/python analysis/probe_v3.py`.
Outputs contain instance_ids and per-task hit flags only; no task content.
"""
import json
import re
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
OUT_DIR = Path("/kaggle/working") if ON_KAGGLE else (HERE / "notes")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# PUBLIC notebook: the dev/eval split is REGENERATED from the mounted competition data in main()
# (deterministic, SEED baked) — no split or labels are embedded or shipped.
SPLIT_SEED, SPLIT_DEV_FRACTION = 20260927, 0.32
DEV, EVAL = set(), set()
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


def rank_metrics(ranked, gold_fn, gold_mod, prefixes=None):
    """file_hit@k uses TOP-K DISTINCT FILES (nodes deduplicated to their file/module in rank order),
    the single definition shared with the SWE-bench Lite probe. If prefixes is None, falls back to the
    legacy top-k-nodes definition (kept only for reproducing pre-review numbers)."""
    r = {}
    if prefixes is not None:
        files_ranked = []
        for n in ranked:
            f = node_file(n, prefixes)
            if f not in files_ranked:
                files_ranked.append(f)
        for k in FILE_KS:
            r[f"file_hit@{k}"] = float(any(under_gold(f, gold_mod) for f in files_ranked[:k]))
    else:
        for k in FILE_KS:
            r[f"file_hit@{k}"] = float(any(under_gold(n, gold_mod) for n in ranked[:k]))
    for k in FN_KS:
        r[f"fn_recall@{k}"] = (len(set(ranked[:k]) & gold_fn) / len(gold_fn)) if gold_fn else np.nan
    return r


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
    adj_dir = {}  # directed (source->target only), for the directed-PPR ablation
    for e in G["edges"]:
        if e["source"] in idx and e["target"] in idx:
            adj.setdefault(e["source"], set()).add(e["target"])
            adj.setdefault(e["target"], set()).add(e["source"])
            adj_dir.setdefault(e["source"], set()).add(e["target"])
    bm = BM25([tokens(nm + " " + tx) for nm, tx in zip(names, texts)])
    sc = bm.scores(tokens(ps))
    sc_bm = BM25([tokens(nm + " " + tx) for nm, tx in zip(names, texts)])  # reused for stage2
    gfn = gold_fn & set(names)
    sym_in_ps = any(re.search(rf"\b{re.escape(g.split('.')[-1])}\b", ps)
                    for g in gold_fn if not g.split(".")[-1].startswith("__"))
    return dict(id=t["instance_id"], repo=t["repo"].split("/")[-1], names=names, texts=texts,
                idx=idx, files=files, vn=vn, adj=adj, adj_dir=adj_dir, sc=sc, bm=sc_bm, ps=ps, idents=idents,
                gold_fn=gfn, gold_mod=gold_mod, ps_words=len(ps.split()), prefixes=prefixes,
                n_files=meta["n_files"], sym_in_ps=sym_in_ps)


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


def rank_oracle(P):
    gm = [n for n in P["names"] if under_gold(n, P["gold_mod"])]
    if not gm:
        return P["names"]
    sub = BM25([tokens(n + " " + P["texts"][P["idx"][n]]) for n in gm])
    oc = [gm[i] for i in np.argsort(-sub.scores(tokens(P["ps"])))]
    return oc


# ---- Item-2 (EXPLORATORY, prereg 2026-10-11) extra methods -------------------
def _file_reps(P):
    """first node seen per file, in node index order (for turning a file order into a node ranking)."""
    rep = {}
    for i, f in enumerate(P["files"]):
        rep.setdefault(f, P["names"][i])
    return rep


def global_pagerank(P, alpha):
    """Non-personalized PageRank (uniform restart) on the symmetrized graph — pure hub score."""
    uniform = np.ones(len(P["names"]))
    return personalized_pagerank(P["names"], P["adj"], uniform, alpha)


def rank_global_pr(P, alpha):
    """No-query hub baseline: rank by global PageRank (ignores the problem statement)."""
    r = global_pagerank(P, alpha)
    return [P["names"][i] for i in np.argsort(-r)]


def rank_ppr_lift(P, alpha):
    """Lift-normalized PPR: PPR mass / global-PR mass, to divide out the hub prior."""
    _, r = rank_ppr(P, alpha)
    g = global_pagerank(P, alpha)
    lift = r / (g + 1e-12)
    return [P["names"][i] for i in np.argsort(-lift)]


def rank_cosine_only(P, seed_k=3):
    """Embedding WITHOUT the BM25 score term: rank by cosine to the BM25-top-seed nodes only
    (0*BM25 + 1*cos), to isolate the embedding's own contribution over BM25."""
    order = np.argsort(-P["sc"])
    seeds = list(order[:seed_k])
    best = (P["vn"][seeds] @ P["vn"].T).max(0) if seeds else np.zeros(len(P["names"]))
    return [P["names"][i] for i in np.argsort(-best)]


def rank_ppr_directed(P, alpha):
    """PPR on the DIRECTED calls graph (source->target only), vs the default symmetrized graph."""
    r = personalized_pagerank(P["names"], P["adj_dir"], P["sc"], alpha)
    return [P["names"][i] for i in np.argsort(-r)]


def rank_bm25_dedup(P):
    """File-level BM25: each file is one document (concat of its nodes' name+text); rank distinct files."""
    from collections import defaultdict
    fdocs = defaultdict(list)
    for i, f in enumerate(P["files"]):
        fdocs[f].extend(tokens(P["names"][i] + " " + P["texts"][i]))
    files = list(fdocs)
    fb = BM25([fdocs[f] for f in files])
    fsc = np.array(fb.scores(tokens(P["ps"])))
    forder = [files[i] for i in np.argsort(-fsc)]
    rep = _file_reps(P)
    return [rep[f] for f in forder if f in rep]


def rank_dev_freq(P, dev_freq):
    """No-query baseline: rank this task's files by gold-frequency observed in the DEV split."""
    order = sorted(set(P["files"]), key=lambda f: -dev_freq.get(f, 0))
    rep = _file_reps(P)
    return [rep[f] for f in order if f in rep]


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p-value (binomial about 0.5), pure-stdlib."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


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
    return rank_metrics(ranked, P["gold_fn"], P["gold_mod"], P.get("prefixes"))


def _regen_split_from_data():
    """Graph-only evaluable filter (matches analysis/make_split.py): a task is evaluable if its patch
    touches an existing Python module present in the shipped graph. Deterministic, SEED baked."""
    import random
    by = {}
    for t in TASKS:
        key = t["repo"].split("/")[-1] + "_" + t["base_commit"]
        gp = DATA_DIR / f"graphs/{key}.json"
        if not gp.exists():
            continue
        nodes = json.load(open(gp))["nodes"]
        gfn, gmod, meta = gold_nodes(t["patch"], nodes)
        if gmod and not meta["all_new"]:
            by.setdefault(t["instance_id"].rsplit("_", 1)[0], []).append(t["instance_id"])
    rng = random.Random(SPLIT_SEED)
    dev, evalset = [], []
    for repo in sorted(by):
        r = sorted(by[repo]); rng.shuffle(r); k = round(SPLIT_DEV_FRACTION * len(r))
        dev += r[:k]; evalset += r[k:]
    # IDENTICAL structure to analysis/make_split.py (same fields/order/note) so the SHA is comparable.
    split = {
        "seed": SPLIT_SEED, "dev_fraction": SPLIT_DEV_FRACTION, "stratify": "repo",
        "note": "Hyperparameters (PPR alpha, seed k, BM25 params, two-stage N/top-k, query expansion) "
                "are tuned on dev ONLY. Paper numbers come from eval. The prior all-124 numbers are exploratory.",
        "n_dev": len(dev), "n_eval": len(evalset),
        "dev_ids": sorted(dev), "eval_ids": sorted(evalset),
    }
    blob = json.dumps(split, indent=1)
    (OUT_DIR / "split.json").write_text(blob)
    import hashlib
    sha = hashlib.sha256(blob.encode()).hexdigest()
    print(f"regenerated split from data: dev={len(dev)} eval={len(evalset)} -> split.json", flush=True)
    print(f"SPLIT_SHA256={sha}", flush=True)
    print("SHA MATCHES committed 49e7303c..." if sha.startswith("49e7303c") else f"SHA MISMATCH (expected 49e7303c..., got {sha[:8]}...)", flush=True)
    return set(dev), set(evalset)


def main():
    global DEV, EVAL
    DEV, EVAL = _regen_split_from_data()
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
    sweep = {"ppr_alpha_dev_file_hit10": ppr_dev, "best_alpha": best_alpha,
             "two_stage_dev_fn_recall10": {str(k): v for k, v in ts_dev.items()},
             "best_two_stage": {"alpha": best_ts[0], "N": best_ts[1], "expand": best_ts[2]}}
    print("SWEEP:", json.dumps(sweep, default=float, indent=1), flush=True)

    a_ts, N_ts, ex_ts = best_ts

    def _grep(P):
        cnt = [sum(1 for w in P["idents"] if w == P["names"][i].split(".")[-1]) for i in range(len(P["names"]))]
        cand = [i for i in range(len(P["names"])) if cnt[i] > 0]
        cand.sort(key=lambda i: (-cnt[i], len(P["names"][i])))
        return [P["names"][i] for i in cand]

    def rankers(P):
        return {
            "grep": _grep(P),
            "bm25": rank_bm25(P),
            "emb_seed": rank_emb_seed(P),
            "graph_ppr": rank_ppr(P, best_alpha)[0],
            "two_stage": rank_two_stage(P, a_ts, N_ts, ex_ts),
            "oracle": rank_oracle(P),
        }

    methods = ["grep", "bm25", "emb_seed", "graph_ppr", "two_stage", "oracle"]
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

    # headline comparisons on EVAL, Holm-corrected
    comps = [("two_stage", "bm25", "fn_recall@10"), ("two_stage", "graph_ppr", "fn_recall@10"),
             ("two_stage", "bm25", "file_hit@10"), ("graph_ppr", "bm25", "file_hit@10"),
             ("emb_seed", "bm25", "file_hit@10"), ("emb_seed", "bm25", "fn_recall@10"),
             ("two_stage", "oracle", "fn_recall@10")]
    raw = {}
    for a, b, met in comps:
        va = col("eval", a, met)
        vb = col("eval", b, met)
        raw[f"{a}-{b}|{met}"] = paired(va, vb)
    report["paired_eval"] = holm(raw)

    # per-task EVAL metrics for every method (for exact McNemar on binary file_hit + bootstrap on fn_recall)
    methods_all = methods
    report["per_task_eval"] = {
        p["id"]: {m: {mt: evaluate(Pby[p["id"]], cache["eval"][p["id"]][m]).get(mt)
                      for mt in ["file_hit@1", "file_hit@5", "file_hit@10", "fn_recall@5", "fn_recall@10", "fn_recall@20"]}
                  for m in methods_all if m in cache["eval"][p["id"]]}
        for p in ev}

    # ---- Item-2 EXPLORATORY analyses (prereg 2026-10-11) ----
    from collections import Counter
    dev_freq = Counter()
    for p in dev:
        for m in p["gold_mod"]:
            dev_freq[m] += 1
    fk = [f"file_hit@{k}" for k in FILE_KS]
    item2_methods = {
        "bm25_dedup": lambda P: rank_bm25_dedup(P),
        "global_pr": lambda P: rank_global_pr(P, best_alpha),
        "ppr_lift": lambda P: rank_ppr_lift(P, best_alpha),
        "dev_freq_top": lambda P: rank_dev_freq(P, dev_freq),
    }
    extra_rk = {name: {p["id"]: fn(Pby[p["id"]]) for p in ev} for name, fn in item2_methods.items()}
    report["item2_eval"] = {
        name: {met: dict(zip(("mean", "lo", "hi"),
                             boot_ci([evaluate(Pby[p["id"]], extra_rk[name][p["id"]])[met] for p in ev])))
               for met in fk}
        for name in item2_methods}

    # alpha sweep (PPR diffusion weight) on eval, distinct-file file_hit
    report["alpha_sweep_eval"] = {
        a: {met: float(np.nanmean([evaluate(Pby[p["id"]], rank_ppr(Pby[p["id"]], a)[0])[met] for p in ev]))
            for met in fk}
        for a in [0.0, 0.05, 0.1, 0.25, 0.5, 0.7]}

    # ---- Item-4 (prereg addendum 2026-10-15): PPR full metrics + per-task McNemar at frozen 0.5 and
    # dev-selected 0.7; plus cosine-only and directed-graph PPR ablations. ----
    report["ppr_by_alpha"] = {}
    report["mcnemar_ppr_alpha"] = {}
    for a in (0.5, 0.7):
        rk = {p["id"]: rank_ppr(Pby[p["id"]], a)[0] for p in ev}
        report["ppr_by_alpha"][str(a)] = {
            "agg": {met: float(np.nanmean([evaluate(Pby[p["id"]], rk[p["id"]])[met] for p in ev])) for met in metrics},
            "per_task_file_hit": {p["id"]: {f"file_hit@{k}": evaluate(Pby[p["id"]], rk[p["id"]])[f"file_hit@{k}"]
                                            for k in FILE_KS} for p in ev},
        }
        for k in FILE_KS:
            bb = cc = 0
            for p in ev:
                pid = p["id"]
                pp = evaluate(Pby[pid], rk[pid])[f"file_hit@{k}"]
                bm = evaluate(Pby[pid], cache["eval"][pid]["bm25"])[f"file_hit@{k}"]
                if pp > bm:
                    bb += 1
                elif bm > pp:
                    cc += 1
            report["mcnemar_ppr_alpha"][f"ppr{a}-bm25|file_hit@{k}"] = {"b": bb, "c": cc, "p": mcnemar_exact(bb, cc)}
    item4_methods = {
        "cosine_only": lambda P: rank_cosine_only(P),
        "ppr_directed_0.5": lambda P: rank_ppr_directed(P, 0.5),
        "ppr_directed_0.7": lambda P: rank_ppr_directed(P, 0.7),
    }
    i4rk = {name: {p["id"]: fn(Pby[p["id"]]) for p in ev} for name, fn in item4_methods.items()}
    report["item4_eval"] = {
        name: {met: float(np.nanmean([evaluate(Pby[p["id"]], i4rk[name][p["id"]])[met] for p in ev]))
               for met in metrics}
        for name in item4_methods}

    # exact McNemar on eval binary file_hit, Holm-corrected (regenerates notes/mcnemar_holm.json)
    def fh_at(pid, meth, k):
        ranked = extra_rk[meth][pid] if meth in extra_rk else cache["eval"][pid][meth]
        return evaluate(Pby[pid], ranked)[f"file_hit@{k}"]
    mcn = {}
    for a, b in [("graph_ppr", "bm25"), ("emb_seed", "bm25"), ("two_stage", "bm25"),
                 ("ppr_lift", "bm25"), ("bm25_dedup", "bm25")]:
        for k in FILE_KS:
            bb = cc = 0
            for p in ev:
                xa, xb = fh_at(p["id"], a, k), fh_at(p["id"], b, k)
                bb += xa > xb
                cc += xb > xa
            mcn[f"{a}-{b}|file_hit@{k}"] = {"b": int(bb), "c": int(cc),
                                            "p_raw": mcnemar_exact(int(bb), int(cc)), "n": len(ev)}
    order = sorted(mcn, key=lambda kk: mcn[kk]["p_raw"]); mm = len(order)
    for rk_, kk in enumerate(order):
        mcn[kk]["p_holm"] = min(1.0, mcn[kk]["p_raw"] * (mm - rk_))
        mcn[kk]["sig"] = bool(mcn[kk]["p_holm"] < 0.05)
    report["mcnemar_eval"] = mcn
    (OUT_DIR / "mcnemar_holm.json").write_text(json.dumps(mcn, indent=1, default=float))

    (OUT_DIR / "probe_v3_results.json").write_text(json.dumps(report, indent=1, default=float))

    # print tables
    for split in ("dev", "eval"):
        print(f"\n=== {split.upper()} (n={report['split'][split]}) ===", flush=True)
        print(f"{'method':10s}" + "".join(f"{m:>14s}" for m in metrics), flush=True)
        for m in methods:
            print(f"{m:10s}" + "".join(f"{report[split][m][met]['mean']:>14.2f}" for met in metrics), flush=True)
    print("\n=== EVAL paired (Holm-corrected, 7 comparisons) ===", flush=True)
    for k, v in report["paired_eval"].items():
        print(f"  {k:28s} Δ={v['delta']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] p_holm={v['p_holm']:.3f} {'*' if v['sig'] else ''}", flush=True)
    print("-> probe_v3_results.json", flush=True)


if __name__ == "__main__":
    main()
