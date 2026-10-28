"""Fig. 1 — the hub mechanism. Three panels vs the diffusion weight alpha, for the competition eval split
(4 small repos) and SWE-bench Lite (12 large repos):
  (left)   file_hit@10 with a 95% bootstrap CI band + each benchmark's no-query hub-prior line;
  (middle) file_hit@1;
  (right)  hub-file share of the top-10 (overlap of PPR's top-10 files with the hub prior's top-10).

As alpha rises, diffusion concentrates on hubs: harmless at @10 on the small competition repos (flat, near
a high hub prior) but collapsing on Lite (hubs are not the buggy files) — and the hub-share climbs on both.

Run: .venv/bin/python analysis/make_fig1_hub.py -> notes/fig_paper_hub.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
comp = json.load(open(ROOT / "results/probe_v3_results.json"))
lite = json.load(open(ROOT / "results/swebench_lite_results.json"))

comp_sw = comp["alpha_sweep_eval"]            # {alpha_str: {file_hit@k: mean}}
comp_det = comp["alpha_sweep_eval_detail"]    # {alpha_str: {file_hit@10, lo, hi, hub_share@10}}
lite_sw = lite["alpha_sweep"]
lite_det = lite["alpha_sweep_detail"]
alphas = sorted((float(a) for a in comp_sw), key=float)
axv = [str(a) if str(a) in comp_sw else f"{a:g}" for a in alphas]

CB, CR = "#1f77b4", "#d62728"


def band(ax, det, color, label):
    ys = [det[a]["file_hit@10"] for a in axv]
    lo = [det[a]["lo"] for a in axv]
    hi = [det[a]["hi"] for a in axv]
    ax.plot(alphas, ys, "o-", color=color, label=label)
    ax.fill_between(alphas, lo, hi, color=color, alpha=0.15)


comp_hub = comp["item2_eval"]["global_pr"]["file_hit@10"]["mean"]
lite_hub = lite["overall"]["global_pr"]["file_hit@10"][0]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))

# (left) file_hit@10 with CI band + hub-prior lines
band(ax1, comp_det, CB, "competition (4 small repos)")
band(ax1, lite_det, CR, "SWE-bench Lite (12 large repos)")
ax1.axhline(comp_hub, ls=":", color=CB, lw=1.3, label=f"competition hub prior {comp_hub:.2f}")
ax1.axhline(lite_hub, ls=":", color=CR, lw=1.3, label=f"Lite hub prior {lite_hub:.2f}")
ax1.set_title("file_hit@10 (95% CI)")
ax1.legend(fontsize=7, loc="center left")

# (middle) file_hit@1
ax2.plot(alphas, [comp_sw[a]["file_hit@1"] for a in axv], "o-", color=CB)
ax2.plot(alphas, [lite_sw[a]["file_hit@1"] for a in axv], "s-", color=CR)
ax2.set_title("file_hit@1")

# (right) hub-file share of top-10
ax3.plot(alphas, [comp_det[a]["hub_share@10"] for a in axv], "o-", color=CB)
ax3.plot(alphas, [lite_det[a]["hub_share@10"] for a in axv], "s-", color=CR)
ax3.set_title("hub-file share of top-10")

for ax in (ax1, ax2, ax3):
    ax.set_xlabel("diffusion weight  α   (α=0: BM25 seed only)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
ax1.set_ylabel("file_hit")
fig.suptitle("Diffusion finds hubs, not bugs: as α grows, PPR's top-10 fills with hub files — "
             "harmless on small repos, fatal on large ones", fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / "results/fig_paper_hub.png", dpi=140)
print(f"-> notes/fig_paper_hub.png  (comp hub {comp_hub:.3f}, lite hub {lite_hub:.3f})")
