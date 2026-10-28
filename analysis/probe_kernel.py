"""Self-contained localisation probe. Runs locally (reads ./data) OR on a Kaggle notebook
(reads /kaggle/input/gemma-4-developer-agent, all files pre-mounted — no download).

Zero third-party deps beyond numpy + matplotlib (both preinstalled on Kaggle). BM25 is inlined
so the kernel needs no internet. Writes results + figures to OUT_DIR.

Local:  .venv/bin/python analysis/probe_kernel.py
Kaggle: pushed as a script kernel with competition data attached, internet off.
"""
import json
import os
import re
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- paths
def _find_data_dir():
    cands = [Path("/kaggle/input/gemma-4-developer-agent")]
    try:
        cands.append(Path(__file__).resolve().parent.parent / "data")
    except NameError:
        pass
    cands.append(Path("data"))
    for p in cands:
        if (p / "tasks.jsonl").exists():
            return p
    # auto-discover anywhere under /kaggle/input
    ki = Path("/kaggle/input")
    if ki.exists():
        print("/kaggle/input contents:", [str(x) for x in ki.rglob("*") if x.is_dir()][:20], flush=True)
        hits = list(ki.rglob("tasks.jsonl"))
        print("tasks.jsonl found at:", [str(h) for h in hits], flush=True)
        if hits:
            return hits[0].parent
    return None


DATA_DIR = _find_data_dir()
assert DATA_DIR, "tasks.jsonl not found (see /kaggle/input listing above)"
ON_KAGGLE = str(DATA_DIR).startswith("/kaggle")
OUT_DIR = Path("/kaggle/working") if ON_KAGGLE else (Path(__file__).resolve().parent.parent / "results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"DATA_DIR={DATA_DIR}  OUT_DIR={OUT_DIR}  on_kaggle={ON_KAGGLE}", flush=True)

RNG = np.random.default_rng(0)
N_BOOT = 2000
FILE_KS, FN_KS = (1, 5, 10), (5, 10, 20)
TASKS = [json.loads(l) for l in open(DATA_DIR / "tasks.jsonl")]

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
TEMPLATE_RE = re.compile(r"<!--.*?-->", re.S)
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
STOP = set("the and for with that this from are not was but you can use have when into then its will "
           "has all should would could add fix use int str these those the".split())


# ---------------------------------------------------------------- inline BM25
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = corpus
        self.dl = np.array([len(d) for d in corpus], dtype=float)
        self.avgdl = self.dl.mean() if len(corpus) else 0.0
        self.f = []
        df = {}
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
                    freq = tf[w]
                    s[i] += self.idf[w] * freq * (self.k1 + 1) / (freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


# ---------------------------------------------------------------- ground truth
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


DEF_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")


def gold_nodes(patch, nodes):
    by_name = {n["name"]: n for n in nodes}
    names = list(by_name)
    gold_fn, gold_mod, gold_hdr = set(), set(), set()  # gold_hdr = independent label (hunk-header symbols)
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
        # independent ground truth: def/class names appearing on changed or hunk-context lines
        hdr_syms = {m.group(1) for ln in old + add for m in [DEF_RE.match(ln)] if m}
        gold_hdr |= {nm for nm in in_mod if nm.split(".")[-1] in hdr_syms}
    meta = dict(n_files=n_files, n_new_files=n_new, all_new=(n_new == n_files and n_files > 0))
    return gold_fn, gold_mod, meta, gold_hdr


# ---------------------------------------------------------------- helpers
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


def personalized_pagerank(names, adj, personal, alpha=0.5, iters=40):
    """Diffuse a BM25 restart distribution over the (undirected) calls graph. Fair 'smart' graph use."""
    idx = {n: i for i, n in enumerate(names)}
    nb = [[idx[m] for m in adj.get(n, ()) if m in idx] for n in names]
    deg = np.array([len(x) for x in nb], dtype=float)
    p = np.maximum(personal, 0)
    p = p / p.sum() if p.sum() > 0 else np.ones(len(names)) / len(names)
    r = p.copy()
    for _ in range(iters):
        push = r / np.maximum(deg, 1)
        nxt = (1 - alpha) * p
        acc = np.zeros(len(names))
        for i, neigh in enumerate(nb):
            if neigh:
                acc[neigh] += push[i]
        r = nxt + alpha * acc
    return r


def rank_metrics(ranked, gold_fn, gold_mod):
    r = {}
    for k in FILE_KS:
        r[f"file_hit@{k}"] = float(any(under_gold(n, gold_mod) for n in ranked[:k]))
    for k in FN_KS:
        r[f"fn_recall@{k}"] = (len(set(ranked[:k]) & gold_fn) / len(gold_fn)) if gold_fn else np.nan
    return r


def load_graph_emb(task):
    key = f'{task["repo"].split("/")[-1]}_{task["base_commit"]}'
    gp, ep = DATA_DIR / f"graphs/{key}.json", DATA_DIR / f"embeddings/{key}.npz"
    if not (gp.exists() and ep.exists()):
        return None, None
    return json.load(open(gp)), np.load(ep)


def build_task(t):
    G, Z = load_graph_emb(t)
    if G is None:
        return {"skip": True, "id": t["instance_id"], "reason": "no graph file"}
    nodes = G["nodes"]
    gold_fn, gold_mod, meta, gold_hdr = gold_nodes(t["patch"], nodes)
    if not gold_mod or meta["all_new"]:
        return {"skip": True, "id": t["instance_id"], "reason": "no in-graph module" if not gold_mod else "all-new-files"}
    ps = TEMPLATE_RE.sub("", t["problem_statement"])
    idents = set(IDENT.findall(ps))
    adj = {}
    for e in G["edges"]:
        adj.setdefault(e["source"], set()).add(e["target"])
        adj.setdefault(e["target"], set()).add(e["source"])

    def run(names, texts, vecs):
        idx = {n: i for i, n in enumerate(names)}
        short = [n.split(".")[-1] for n in names]
        vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

        # --- grep: identifier match count
        cnt = {n: sum(1 for w in idents if w == s) for n, s in zip(names, short)}
        grep = sorted([n for n in names if cnt[n] > 0], key=lambda n: (-cnt[n], len(n)))

        # --- bm25 over name+text
        bm = BM25([tokens(nm + " " + tx) for nm, tx in zip(names, texts)])
        sc = bm.scores(tokens(ps))
        order = np.argsort(-sc)
        bm25 = [names[i] for i in order]

        # --- bm25 + naive 1-hop calls expansion (re-rank by bm25(node)+0.5*max neighbour bm25)
        comb = np.array([sc[i] + 0.5 * max((sc[idx[m]] for m in adj.get(n, ()) if m in idx), default=0.0)
                         for i, n in enumerate(names)])
        bm25g = [names[i] for i in np.argsort(-comb)]

        # --- emb NAIVE tool-faithful: resolve every PS identifier to a node key, average vectors, cosine
        suffix = {}
        for i, n in enumerate(names):
            suffix.setdefault(short[i], []).append(i)
        hit = []
        for w in idents:
            if w in idx:
                hit.append(idx[w])
            elif w in suffix:
                hit += suffix[w]
        hit = list(dict.fromkeys(hit))
        if hit:
            qv = vn[hit].mean(0)
            emb = [names[i] for i in np.argsort(-(vn @ qv))]
        else:
            emb = []

        # --- emb_seed FAIR intended-use: seed from BM25 top-3, expand via search_similar_code
        #     (cosine-nearest to each seed vector), rank pool by bm25 + max seed cosine.
        seeds = list(order[:3])
        seed_sim = vn[seeds] @ vn.T if seeds else np.zeros((0, len(names)))
        best_sim = seed_sim.max(0) if len(seeds) else np.zeros(len(names))
        emb_seed = [names[i] for i in np.argsort(-(0.5 * (sc / (sc.max() or 1)) + 0.5 * best_sim))]

        # --- graph_ppr FAIR: personalized PageRank with BM25 restart distribution
        ppr = personalized_pagerank(names, adj, sc)
        graph_ppr = [names[i] for i in np.argsort(-ppr)]

        # --- graph_symbol FAIR: expand caller/callee of statement-mentioned symbols, within same file,
        #     rank that pool by bm25 (rest after by bm25).
        sym_seeds = [n for n in names if short[idx[n]] in idents]
        pool = set(sym_seeds)
        for n in sym_seeds:
            file_prefix = n.rsplit(".", 1)[0]
            for m in adj.get(n, ()):
                if m in idx and m.startswith(file_prefix.rsplit(".", 1)[0]):
                    pool.add(m)
        in_pool = sorted(pool, key=lambda n: -sc[idx[n]])
        rest = [n for n in bm25 if n not in pool]
        graph_symbol = in_pool + rest if pool else bm25

        return (dict(grep=grep, bm25=bm25, bm25g=bm25g, emb=emb, emb_seed=emb_seed,
                     graph_ppr=graph_ppr, graph_symbol=graph_symbol), len(hit))

    out = {"id": t["instance_id"], "repo": t["repo"], "ps_words": len(ps.split()),
           "n_gold_fn": len(gold_fn), "meta": meta}
    for tag, keep_test in (("with_test", True), ("no_test", False)):
        sel = [n for n in nodes if keep_test or not is_test_node(n["name"])]
        names = [n["name"] for n in sel]
        texts = [n["text"] or "" for n in sel]
        vecs = np.stack([Z[n] for n in names])
        gfn = gold_fn & set(names)
        rankers, n_hit = run(names, texts, vecs)
        res = {m: rank_metrics(r, gfn, gold_mod) for m, r in rankers.items()}
        gm_names = [n for n in names if under_gold(n, gold_mod)]
        if gm_names:
            bm = BM25([tokens(n + " " + texts[names.index(n)]) for n in gm_names])
            oc = [gm_names[i] for i in np.argsort(-bm.scores(tokens(ps)))]
            res["oracle"] = rank_metrics(oc, gfn, gold_mod)
        out[tag] = {"n_nodes": len(names), "n_gold_fn": len(gfn), "n_idents_resolved": n_hit, "res": res}
    out["symbol_mentioned"] = any(
        re.search(rf"\b{re.escape(g.split('.')[-1])}\b", ps) for g in gold_fn if not g.split(".")[-1].startswith("__"))
    # ground-truth inter-check: graph-text gold vs independent hunk-header-symbol gold
    inter = len(gold_fn & gold_hdr)
    out["gt_check"] = dict(n_textgold=len(gold_fn), n_hdrgold=len(gold_hdr), n_overlap=inter,
                           jaccard=(inter / len(gold_fn | gold_hdr)) if (gold_fn | gold_hdr) else 1.0,
                           hdr_covered=(inter / len(gold_hdr)) if gold_hdr else None)
    # test-node-in-gold: is excluding test candidates fair? (are any gold nodes test nodes?)
    out["n_gold_test"] = sum(is_test_node(g) for g in gold_fn)
    return out


def boot_ci(vals):
    vals = np.array([v for v in vals if v == v])
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    means = vals[RNG.integers(0, len(vals), (N_BOOT, len(vals)))].mean(1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_delta(a, b):
    d = np.array([x - y for x, y in zip(a, b) if (x == x and y == y)])
    if not len(d):
        return None
    bs = d[RNG.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    return dict(delta=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
               p_le0=float((bs <= 0).mean()), n=int(len(d)))


def main():
    tasks, skipped = [], []
    for t in TASKS:
        r = build_task(t)
        (skipped if r.get("skip") else tasks).append(r)
    from collections import Counter
    report = {"n_tasks": len(tasks), "n_skipped": len(skipped),
              "skip_reasons": dict(Counter(s["reason"] for s in skipped)),
              "data_dir": str(DATA_DIR), "main": {}, "paired": {}, "subgroups": {}}
    print(f"evaluated {len(tasks)}  skipped {len(skipped)} {report['skip_reasons']}", flush=True)

    methods = ["grep", "bm25", "emb", "emb_seed", "bm25g", "graph_ppr", "graph_symbol", "oracle"]
    metrics = [f"file_hit@{k}" for k in FILE_KS] + [f"fn_recall@{k}" for k in FN_KS]

    # ground-truth quality + fairness aggregates
    jac = [t["gt_check"]["jaccard"] for t in tasks]
    hdrcov = [t["gt_check"]["hdr_covered"] for t in tasks if t["gt_check"]["hdr_covered"] is not None]
    report["gt_inter_check"] = dict(
        n=len(tasks), jaccard_mean=float(np.mean(jac)), jaccard_median=float(np.median(jac)),
        hdr_covered_mean=float(np.mean(hdrcov)) if hdrcov else None,
        tasks_full_agreement=int(sum(j == 1.0 for j in jac)))
    report["test_in_gold"] = dict(
        tasks_with_test_gold=int(sum(t["n_gold_test"] > 0 for t in tasks)),
        total_gold_test_nodes=int(sum(t["n_gold_test"] for t in tasks)))

    def collect(method, metric, subset=None, tag="no_test"):
        return [t[tag]["res"][method][metric] for t in (subset or tasks)
                if method in t[tag]["res"] and metric in t[tag]["res"][method]]

    for method in methods:
        for m in metrics:
            mean, lo, hi = boot_ci(collect(method, m))
            report["main"].setdefault(method, {})[m] = dict(mean=mean, lo=lo, hi=hi)

    for m in ("file_hit@10", "fn_recall@10"):
        report.setdefault("ablation_testnodes", {})[m] = dict(
            with_test=boot_ci(collect("bm25", m, tag="with_test")),
            no_test=boot_ci(collect("bm25", m, tag="no_test")))

    for metric in ("file_hit@10", "fn_recall@10"):
        for a, b in [("bm25", "emb"), ("bm25", "grep"), ("bm25g", "bm25"),
                     ("emb_seed", "bm25"), ("graph_ppr", "bm25"), ("graph_symbol", "bm25"),
                     ("emb_seed", "emb"), ("oracle", "bm25")]:
            va = {t["id"]: t["no_test"]["res"].get(a, {}).get(metric, np.nan) for t in tasks}
            vb = {t["id"]: t["no_test"]["res"].get(b, {}).get(metric, np.nan) for t in tasks}
            ids = [i for i in va if va[i] == va[i] and vb[i] == vb[i]]
            report["paired"].setdefault(metric, {})[f"{a}-{b}"] = paired_delta([va[i] for i in ids], [vb[i] for i in ids])

    def subgroup(name, fn):
        groups = {}
        for t in tasks:
            groups.setdefault(fn(t), []).append(t)
        report["subgroups"][name] = {
            str(g): dict(n=len(ts),
                         bm25_file_hit10=boot_ci(collect("bm25", "file_hit@10", ts)),
                         bm25_fn_recall10=boot_ci(collect("bm25", "fn_recall@10", ts)),
                         emb_fn_recall10=boot_ci(collect("emb", "fn_recall@10", ts)))
            for g, ts in groups.items()}

    subgroup("repo", lambda t: t["repo"].split("/")[-1])
    subgroup("ps_len", lambda t: "<30w" if t["ps_words"] < 30 else ">=30w")
    subgroup("symbol_mentioned", lambda t: "sym_in_PS" if t["symbol_mentioned"] else "sym_absent")
    subgroup("patch_files", lambda t: "1_file" if t["meta"]["n_files"] == 1 else "multi_file")

    (OUT_DIR / "full_probe_results.json").write_text(json.dumps(report, indent=1, default=float))
    (OUT_DIR / "full_probe_tasks.json").write_text(json.dumps(tasks, indent=1, default=float))
    print(json.dumps(report["main"], indent=1, default=float), flush=True)

    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = {"grep": "grep", "emb": "emb(naive)", "emb_seed": "emb(seeded)", "bm25": "BM25",
                  "bm25g": "BM25+1hop", "graph_ppr": "graph(PPR)", "graph_symbol": "graph(sym)",
                  "oracle": "oracle(file)"}
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, met in zip(axes.flat, metrics):
            ys, los, his = [], [], []
            for mth in methods:
                d = report["main"][mth][met]
                ys.append(d["mean"] if d["mean"] == d["mean"] else 0)
                los.append((d["mean"] - d["lo"]) if d["mean"] == d["mean"] else 0)
                his.append((d["hi"] - d["mean"]) if d["mean"] == d["mean"] else 0)
            ax.bar(range(len(methods)), ys, yerr=[los, his], capsize=3, color="#4C78A8")
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([labels[m] for m in methods], rotation=30, ha="right", fontsize=8)
            ax.set_title(met, fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Localisation retrievers (N={len(tasks)}, test nodes excluded, 95% CI)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_methods.png", dpi=130)

        subs = ["repo", "ps_len", "symbol_mentioned", "patch_files"]
        fig, axes = plt.subplots(1, len(subs), figsize=(15, 4))
        for ax, s in zip(axes, subs):
            g = report["subgroups"][s]
            nm = list(g)
            ys = [g[n]["bm25_fn_recall10"][0] for n in nm]
            lo = [g[n]["bm25_fn_recall10"][0] - g[n]["bm25_fn_recall10"][1] for n in nm]
            hi = [g[n]["bm25_fn_recall10"][2] - g[n]["bm25_fn_recall10"][0] for n in nm]
            ax.bar(range(len(nm)), ys, yerr=[lo, hi], capsize=3, color="#F58518")
            ax.set_xticks(range(len(nm)))
            ax.set_xticklabels([f"{n}\n(n={g[n]['n']})" for n in nm], fontsize=8)
            ax.set_title(f"by {s}", fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("BM25 function recall@10 by subgroup (95% CI)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_subgroups.png", dpi=130)
        print("figures written", flush=True)
    except Exception as e:
        print("plot skipped:", e, flush=True)


if __name__ == "__main__":
    main()
