"""Paper figures (CPU). Reads notes/probe_v3_results.json + notes/mde_table.json.

(a) fig_paper_retrievers.png — EVAL retriever comparison (file_hit@10, fn_recall@10) with 95% CIs.
(b) fig_paper_mde.png       — MDE vs N (unpaired + paired r=4%,8%), with the A-C=0.02 effect line and
                              the plausible public-N band (50-100).
(c) fig_paper_gap.png       — localization->resolution gap: file_hit@10, fn_recall@10, oracle-file
                              fn_recall@10, and the agent resolve rate (A=0.08) on one axis.

Run: .venv/bin/python analysis/make_paper_figs.py -> notes/fig_paper_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "results/probe_v3_results.json").read_text())
M = json.loads((ROOT / "results/mde_table.json").read_text())
EV = R["eval"]
BLUE, ORANGE, GREEN, GREY = "#4C78A8", "#F58518", "#54A24B", "#9C9C9C"

# ---- (a) retriever comparison on EVAL with CIs ----
methods = [("grep", "grep"), ("bm25", "BM25"), ("emb_seed", "emb tool\n(seeded)"),
           ("graph_ppr", "graph PPR"), ("oracle", "oracle\n(file)")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, met, ttl in [(axes[0], "file_hit@10", "File localization (file_hit@10)"),
                     (axes[1], "fn_recall@10", "Function localization (fn_recall@10)")]:
    xs = range(len(methods))
    ys = [EV[m][met]["mean"] for m, _ in methods]
    lo = [EV[m][met]["mean"] - EV[m][met]["lo"] for m, _ in methods]
    hi = [EV[m][met]["hi"] - EV[m][met]["mean"] for m, _ in methods]
    cols = [GREY, BLUE, ORANGE, GREEN, "#333333"]
    ax.bar(xs, ys, yerr=[lo, hi], capsize=4, color=cols)
    ax.set_xticks(list(xs)); ax.set_xticklabels([t for _, t in methods], fontsize=9)
    ax.set_ylim(0, 1); ax.set_title(ttl, fontsize=11); ax.grid(axis="y", alpha=0.3)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.02, f"{y:.2f}", ha="center", fontsize=8)
fig.suptitle(f"Retrieval on the frozen EVAL split (N={R['split']['eval']}, test nodes excluded, 95% CI)", fontsize=12)
fig.tight_layout(); fig.savefig(ROOT / "results/fig_paper_retrievers.png", dpi=140); print("-> fig_paper_retrievers.png")

# ---- (b) MDE vs N (inferred graded N=58 marked) ----
Ns = [50, 58, 85, 129, 300, 500]
unp = [M["unpaired"][str(n)] for n in Ns]
pr8 = [M["paired"][str(n)].get("0.08") for n in Ns]
mde58 = M["unpaired"]["58"]
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(Ns, unp, "o-", color=BLUE, label="unpaired MDE (exact Fisher)")
ax.plot(Ns, [p if p is not None else np.nan for p in pr8], "s--", color=ORANGE, label="paired MDE (disc. 8%)")
ax.axhline(0.02, color="red", lw=1.4, label="observed A−C = 0.02  (≈ one task at N=58)")
ax.axvline(58, color="purple", ls=":", lw=1.5)
ax.plot([58], [mde58], "D", color="purple", ms=8, zorder=5,
        label=f"inferred graded N = 58  (MDE = {mde58:.2f})")
ax.annotate(f"N=58\nMDE={mde58:.2f}", (58, mde58), textcoords="offset points", xytext=(10, 2),
            fontsize=8, color="purple")
ax.set_xlabel("number of tasks N"); ax.set_ylabel("minimum detectable effect (resolve-rate points)")
ax.set_title("The public leaderboard is under-powered for the effects competitors chase\n(80% power, α=0.05, p=0.08)", fontsize=11)
ax.set_ylim(0, 0.25); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
for n, u in zip(Ns, unp):
    if n == 58:
        continue  # the N=58 value is already shown by the purple 'N=58 / MDE=0.21' annotation
    ax.annotate(f"{u:.02f}", (n, u), textcoords="offset points", xytext=(0, 6), fontsize=7, color=BLUE)
fig.tight_layout(); fig.savefig(ROOT / "results/fig_paper_mde.png", dpi=140); print("-> fig_paper_mde.png")

# ---- (c) localization -> resolution gap ----
stages = ["file_hit@10\n(find file)", "fn_recall@10\n(find function)", "oracle file→fn\n(fn | gold file)",
          "resolve rate\n(agent, A)"]
vals = [EV["bm25"]["file_hit@10"]["mean"], EV["bm25"]["fn_recall@10"]["mean"],
        EV["oracle"]["fn_recall@10"]["mean"], 0.08]
cols = [BLUE, BLUE, GREEN, ORANGE]
fig, ax = plt.subplots(figsize=(8, 4.6))
bars = ax.bar(range(len(stages)), vals, color=cols)
ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, fontsize=9)
ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
ax.set_title("Where a small quantized agent loses: retrieval vs. resolution\n"
             "(BM25 retrieval on EVAL; agent resolve rate on the public grader)", fontsize=11)
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax.text(3, 0.08 + 0.10, "offline retrieval gains\ndid NOT reach resolve rate\n(A=0.08, C=0.06)",
        ha="center", fontsize=8, color="red")
fig.tight_layout(); fig.savefig(ROOT / "results/fig_paper_gap.png", dpi=140); print("-> fig_paper_gap.png")
