"""Descriptive statistics over data/tasks.jsonl (129 tasks). Pure Python, no LLM.

Usage: .venv/bin/python analysis/task_stats.py  -> prints summary, writes notes/task_table.csv
"""
import collections
import csv
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = [json.loads(l) for l in open(ROOT / "data/tasks.jsonl")]

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")
DEF_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")
TEMPLATE_RE = re.compile(r"<!--.*?-->", re.S)

BUG_WORDS = re.compile(r"\b(bug|fix(es|ed)?|error|exception|traceback|crash|broken|incorrect|wrong|regression|fails?|issue)\b", re.I)
FEAT_WORDS = re.compile(r"\b(add(s|ed)?|support|feature|new|implement|introduce|allow|enable|option)\b", re.I)


def parse_patch(patch: str):
    """Return per-file dicts: path, added, removed, hunks, is_new, symbols touched (from hunk ctx + def lines)."""
    files, cur = [], None
    for line in patch.splitlines():
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            path = line[4:].strip()
            path = path[2:] if path.startswith("b/") else path
            cur = dict(path=path, added=0, removed=0, hunks=0, is_new=False, symbols=set())
            files.append(cur)
            continue
        if cur is None:
            continue
        m = HUNK_RE.match(line)
        if m:
            cur["hunks"] += 1
            if m.group(1) == "0" and (m.group(2) or "") == "0":
                cur["is_new"] = True
            d = DEF_RE.match(m.group(5) or "")
            if d:
                cur["symbols"].add(d.group(1))
            continue
        if line.startswith("+") or line.startswith("-"):
            cur["added" if line[0] == "+" else "removed"] += 1
            d = DEF_RE.match(line[1:])
            if d:
                cur["symbols"].add(d.group(1))
    return files


def pct(x, n):
    return f"{100 * x / n:.1f}%"


def dist(xs):
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return f"mean={st.mean(xs):.1f} median={st.median(xs)} p25={q(.25)} p75={q(.75)} p90={q(.9)} max={xs[-1]}"


def main():
    rows = []
    for t in TASKS:
        ps = t["problem_statement"]
        ps_clean = TEMPLATE_RE.sub("", ps)
        files = parse_patch(t["patch"])
        src = [f for f in files if f["path"].endswith(".py")]
        symbols = set().union(*[f["symbols"] for f in files]) if files else set()
        low = ps_clean.lower()
        file_mentioned = any(Path(f["path"]).name.lower() in low or Path(f["path"]).stem.lower() in re.findall(r"\w+", low) for f in src)
        path_mentioned = any(f["path"].lower() in low for f in files)
        sym_mentioned = any(re.search(rf"\b{re.escape(s)}\b", ps_clean) for s in symbols)
        bug = len(BUG_WORDS.findall(ps_clean))
        feat = len(FEAT_WORDS.findall(ps_clean))
        rows.append(dict(
            instance_id=t["instance_id"], repo=t["repo"], created_at=t["created_at"][:10],
            ps_chars=len(ps), ps_chars_clean=len(ps_clean.strip()), ps_words=len(ps_clean.split()),
            has_hints=bool(t["hints_text"].strip()), has_test_patch=bool(t["test_patch"].strip()),
            n_files=len(files), n_py_files=len(src),
            n_new_files=sum(f["is_new"] for f in files),
            added=sum(f["added"] for f in files), removed=sum(f["removed"] for f in files),
            hunks=sum(f["hunks"] for f in files), n_symbols=len(symbols),
            max_depth=max((f["path"].count("/") for f in files), default=0),
            file_kinds="|".join(sorted({f["path"].split("/")[0] if "/" in f["path"] else "(root)" for f in files})),
            file_mentioned=file_mentioned, path_mentioned=path_mentioned, symbol_mentioned=sym_mentioned,
            type="bug" if bug > feat else "feature" if feat > bug else "unclear",
            touches_docs_src=any(f["path"].startswith("docs_src/") for f in files),
            touches_non_py=any(not f["path"].endswith(".py") for f in files),
            files="|".join(f["path"] for f in files), symbols="|".join(sorted(symbols)),
        ))

    n = len(rows)
    out = ROOT / "results/task_table.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"N={n}  -> {out.relative_to(ROOT)}\n")
    print("Tasks per repo:", dict(collections.Counter(r["repo"] for r in rows)))
    years = collections.Counter(r["created_at"][:4] for r in rows)
    print("created_at year:", dict(sorted(years.items())), "| range", min(r["created_at"] for r in rows), "→", max(r["created_at"] for r in rows))
    print("\nproblem_statement words (template comments stripped):", dist([r["ps_words"] for r in rows]))
    print("  <30 words:", pct(sum(r["ps_words"] < 30 for r in rows), n))
    print("has hints:", pct(sum(r["has_hints"] for r in rows), n), "| has test_patch:", pct(sum(r["has_test_patch"] for r in rows), n))
    print("\nfiles changed:", dist([r["n_files"] for r in rows]))
    print("  single-file:", pct(sum(r["n_files"] == 1 for r in rows), n), "| creates new file:", pct(sum(r["n_new_files"] > 0 for r in rows), n))
    print("lines changed (+/-):", dist([r["added"] + r["removed"] for r in rows]))
    print("hunks:", dist([r["hunks"] for r in rows]))
    print("functions/classes touched (heuristic):", dist([r["n_symbols"] for r in rows]))
    print("max path depth (# of '/'):", dict(sorted(collections.Counter(r["max_depth"] for r in rows).items())))
    print("top-level dirs touched:", collections.Counter(k for r in rows for k in r["file_kinds"].split("|")).most_common(10))
    print("touches docs_src/:", pct(sum(r["touches_docs_src"] for r in rows), n), "| touches non-.py:", pct(sum(r["touches_non_py"] for r in rows), n))
    print("\ntype (keyword heuristic):", dict(collections.Counter(r["type"] for r in rows)))
    print("edited .py file name mentioned in PS:", pct(sum(r["file_mentioned"] for r in rows), n))
    print("edited full path mentioned in PS:", pct(sum(r["path_mentioned"] for r in rows), n))
    print("edited function/class name mentioned in PS:", pct(sum(r["symbol_mentioned"] for r in rows), n))
    print("\nPer repo:")
    for repo in sorted({r["repo"] for r in rows}):
        rr = [r for r in rows if r["repo"] == repo]
        print(f"  {repo:18s} n={len(rr):3d} words_med={st.median([r['ps_words'] for r in rr]):6.0f} "
              f"files_med={st.median([r['n_files'] for r in rr])} lines_med={st.median([r['added'] + r['removed'] for r in rr])} "
              f"test_patch={pct(sum(r['has_test_patch'] for r in rr), len(rr))} sym_mentioned={pct(sum(r['symbol_mentioned'] for r in rr), len(rr))} "
              f"file_mentioned={pct(sum(r['file_mentioned'] for r in rr), len(rr))}")


if __name__ == "__main__":
    main()
