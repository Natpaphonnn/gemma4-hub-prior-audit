"""localize.py — dependency-free BM25 localizer for the swegemma sandbox.

Runs INSIDE the agent's /workspace container. Parses the repo's Python source into function/class
units (AST), builds BM25 over unit name+source, filters out test code, and prints the top-k
file:function candidates with line ranges. Pure standard library only (no pip, no /wheels needed) so it
works offline in the sandbox.

Usage (the agent runs this via run_command, or writes it to /tmp and runs it):
    python3 localize.py --root /workspace --k 10 "PROBLEM STATEMENT TEXT"
    python3 localize.py --root /workspace --k 10 --statement-file /tmp/ps.txt

Output is compact (a ranked list, no source dumps) to stay within the 32k context budget.
"""
import argparse
import ast
import io
import json
import math
import os
import re
import sys
import tokenize

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
STOP = set("the and for with that this from are not was but you can use have when into then its will "
           "has all should would could add fix use these those".split())
TEST_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$")


def tokens(text):
    out = []
    for w in IDENT.findall(text):
        for p in re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", w):
            if len(p) > 1:
                out.append(p.lower())
        out.append(w.lower())
    return [t for t in out if t not in STOP]


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1, self.b, self.docs = k1, b, corpus
        self.dl = [len(d) for d in corpus]
        self.avgdl = (sum(self.dl) / len(corpus)) if corpus else 0.0
        self.f, df = [], {}
        for d in corpus:
            tf = {}
            for w in d:
                tf[w] = tf.get(w, 0) + 1
            self.f.append(tf)
            for w in tf:
                df[w] = df.get(w, 0) + 1
        N = len(corpus)
        self.idf = {w: max(1e-6, math.log((N - n + 0.5) / (n + 0.5) + 1)) for w, n in df.items()}

    def scores(self, query):
        q = [w for w in query if w in self.idf]
        out = []
        for i, tf in enumerate(self.f):
            s, dl = 0.0, self.dl[i]
            for w in q:
                if w in tf:
                    fr = tf[w]
                    s += self.idf[w] * fr * (self.k1 + 1) / (fr + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


def module_name(rel_path):
    p = re.sub(r"^(src|docs_src)/", "", rel_path)
    p = p[:-3] if p.endswith(".py") else p
    return p.replace("/", ".").rstrip(".").removesuffix(".__init__")


def iter_units(root):
    """Yield (name, text, rel_path, lineno, end_lineno) for each function/class in the repo's .py files."""
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            if TEST_RE.search("/" + rel.replace(os.sep, "/")):
                continue
            try:
                src = open(full, encoding="utf-8", errors="ignore").read()
                tree = ast.parse(src)
            except Exception:
                continue
            lines = src.splitlines()
            mod = module_name(rel.replace(os.sep, "/"))

            def walk(node, prefix):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        name = f"{prefix}.{child.name}" if prefix else child.name
                        lo = child.lineno
                        hi = getattr(child, "end_lineno", lo)
                        text = "\n".join(lines[lo - 1:hi])
                        yield (f"{mod}.{name}", text, rel.replace(os.sep, "/"), lo, hi)
                        if isinstance(child, ast.ClassDef):
                            yield from walk(child, name)
            yield from walk(tree, "")


def build_call_graph(units):
    """Lightweight undirected call graph from source: edge between unit i and j if j's short name is
    referenced as an identifier in i's body (or vice versa). Approximates the shipped `calls` graph."""
    short = [u[0].split(".")[-1] for u in units]
    short_to_idx = {}
    for i, s in enumerate(short):
        short_to_idx.setdefault(s, []).append(i)
    adj = [set() for _ in units]
    for i, (_, text, *_ ) in enumerate(units):
        refs = set(IDENT.findall(text or ""))
        for r in refs:
            for j in short_to_idx.get(r, ()):
                if j != i:
                    adj[i].add(j); adj[j].add(i)
    return adj


def personalized_pagerank(adj, restart, alpha=0.5, iters=40):
    n = len(adj)
    deg = [len(a) or 1 for a in adj]
    p = [max(x, 0.0) for x in restart]
    s = sum(p) or 1.0
    p = [x / s for x in p]
    r = p[:]
    for _ in range(iters):
        nxt = [(1 - alpha) * p[i] for i in range(n)]
        for i in range(n):
            share = alpha * r[i] / deg[i]
            for j in adj[i]:
                nxt[j] += share
        r = nxt
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("statement", nargs="?", default="")
    ap.add_argument("--root", default="/workspace")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--statement-file")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ppr", action="store_true",
                    help="re-rank FILES by BM25-seeded personalized PageRank over a source call-graph (C2 recipe)")
    ap.add_argument("--alpha", type=float, default=0.5)
    a = ap.parse_args()
    ps = open(a.statement_file).read() if a.statement_file else a.statement
    if not ps.strip():
        print("localize: empty statement", file=sys.stderr)
        sys.exit(2)

    units = list(iter_units(a.root))
    if not units:
        print("localize: no python units found under", a.root, file=sys.stderr)
        sys.exit(1)
    names = [u[0] for u in units]
    bm = BM25([tokens(n + " " + t) for n, t, *_ in units])
    sc = bm.scores(tokens(ps))
    ppr = None
    if a.ppr:
        adj = build_call_graph(units)
        ppr = personalized_pagerank(adj, sc, alpha=a.alpha)
    order = sorted(range(len(units)), key=lambda i: -sc[i])

    # file ranking = best (PPR if --ppr, else BM25) unit score per file
    fscore_src = ppr if ppr is not None else sc
    file_best = {}
    for i in range(len(units)):
        rel = units[i][2]
        if rel not in file_best or fscore_src[i] > file_best[rel]:
            file_best[rel] = fscore_src[i]
    top_files = sorted(file_best, key=lambda f: -file_best[f])[:a.k]

    if a.json:
        print(json.dumps({
            "top_files": top_files[:a.k],
            "top_functions": [{"name": names[i], "file": units[i][2],
                               "lines": [units[i][3], units[i][4]], "score": round(sc[i], 3)}
                              for i in order[:a.k]],
        }, indent=1))
        return
    print(f"# localize: top {a.k} candidate files")
    for f in top_files:
        print(f"  {f}")
    print(f"\n# top {a.k} candidate functions/classes (file:lines  symbol)")
    for i in order[:a.k]:
        n, _, rel, lo, hi = units[i]
        print(f"  {rel}:{lo}-{hi}  {n.split('.')[-1]}   (score {sc[i]:.2f})")


if __name__ == "__main__":
    main()
