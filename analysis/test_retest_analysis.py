"""§4 test-retest analysis (pre-registered: notes/prereg.md). Reads test_retest.csv (COMPLETE runs
with scores) and reports, per the pre-registration:
  - per-config mean and 95% t-interval (small-n, t not normal);
  - pooled within-config SD vs the binomial expectation sqrt(p(1-p)/N) at N=100 (N=129 sensitivity);
  - ICC(1,1): fraction of score variance that is between-config vs run-to-run (one-way random effects);
  - exact permutation test for each between-config mean difference (enumerate all relabelings);
  - runs-needed-per-config to detect a 0.02 / 0.01 between-config difference at 80% power (pooled SD).
Runs at whatever n the CSV holds (preview now at n=4; final at n=5). Writes notes/test_retest_analysis.json.

Run: .venv/bin/python analysis/test_retest_analysis.py
"""
import csv
import json
from itertools import combinations
from math import comb, sqrt
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
Z_A, Z_B = 1.959963985, 0.841621234  # alpha=0.05 two-sided, 80% power


def load():
    rows = list(csv.DictReader(open(ROOT / "test_retest.csv")))
    by = {}
    for r in rows:
        if r["status"] == "COMPLETE" and r["public_score"]:
            by.setdefault(r["config"], []).append(float(r["public_score"]))
    return by


def t_interval(x):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    if n < 2:
        return m, float("nan"), float("nan")
    se = x.std(ddof=1) / sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return m, m - h, m + h


def icc_1_1(groups):
    """One-way random-effects ICC(1,1). groups: list of equal-length score lists (configs x runs)."""
    k = len(groups); ns = [len(g) for g in groups]
    if len(set(ns)) != 1 or ns[0] < 2:
        return None
    n = ns[0]
    allx = np.array(groups, float)          # k x n
    grand = allx.mean()
    msb = n * ((allx.mean(1) - grand) ** 2).sum() / (k - 1)          # between configs
    msw = ((allx - allx.mean(1, keepdims=True)) ** 2).sum() / (k * (n - 1))  # within config
    denom = msb + (n - 1) * msw
    return float((msb - msw) / denom) if denom else None, float(msb), float(msw)


def exact_perm(a, b):
    """Exact two-sided permutation p for mean(a)-mean(b), enumerating all label assignments."""
    a, b = list(a), list(b); pool = a + b; na = len(a); N = len(pool)
    obs = abs(np.mean(a) - np.mean(b))
    cnt = tot = 0
    for idx in combinations(range(N), na):
        ga = [pool[i] for i in idx]; gb = [pool[i] for i in range(N) if i not in idx]
        if abs(np.mean(ga) - np.mean(gb)) >= obs - 1e-12:
            cnt += 1
        tot += 1
    return float(cnt / tot), int(comb(N, na)), float(obs)


def runs_needed(sigma, delta):
    """Runs per config for a two-sample comparison to detect delta at 80% power, given per-run SD sigma."""
    if delta <= 0 or sigma <= 0:
        return None
    return int(np.ceil(2 * (Z_A + Z_B) ** 2 * sigma ** 2 / delta ** 2))


def main():
    by = load()
    out = {"n_per_config": {c: len(v) for c, v in by.items()}, "per_config": {}, "between": {}}
    for c, v in sorted(by.items()):
        m, lo, hi = t_interval(v)
        out["per_config"][c] = {"runs": v, "mean": round(m, 4),
                                "sd": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None,
                                "t95": [round(lo, 4), round(hi, 4)]}
    # pooled within-config SD
    ss = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in by.values() if len(v) > 1)
    df = sum(len(v) - 1 for v in by.values() if len(v) > 1)
    pooled = sqrt(ss / df) if df else float("nan")
    out["pooled_within_sd"] = round(pooled, 4)
    for N in (100, 129):
        p = np.mean([x for v in by.values() for x in v])
        out[f"binomial_sd_N{N}"] = round(sqrt(p * (1 - p) / N), 4)
    # ICC (balanced only)
    gl = [v for v in by.values() if len(v) == min(len(x) for x in by.values())]
    if len({len(v) for v in by.values()}) == 1 and min(len(v) for v in by.values()) >= 2:
        icc, msb, msw = icc_1_1(list(by.values()))
        k = len(by); n = min(len(v) for v in by.values())
        # ICC(1,1) 95% CI via the F distribution (Shrout & Fleiss 1979)
        f_obs = msb / msw if msw else float("inf")
        df1, df2 = k - 1, k * (n - 1)
        fl = f_obs / stats.f.ppf(0.975, df1, df2)
        fu = f_obs * stats.f.ppf(0.975, df2, df1)
        icc_lo = (fl - 1) / (fl + (n - 1))
        icc_hi = (fu - 1) / (fu + (n - 1))
        out["icc_1_1"] = {"icc": round(icc, 4), "ci95": [round(icc_lo, 4), round(icc_hi, 4)],
                          "msb": round(msb, 6), "msw": round(msw, 6), "k_configs": k, "n_runs": n,
                          "reading": "fraction of score variance that is between-config vs run-to-run"}
    # between-config exact permutation
    cfgs = sorted(by)
    for a, b in combinations(cfgs, 2):
        p, nperm, obs = exact_perm(by[a], by[b])
        out["between"][f"{a}-{b}"] = {"mean_diff": round(float(np.mean(by[a]) - np.mean(by[b])), 4),
                                      "perm_p": round(p, 4), "n_perm": nperm}
    # runs-needed using pooled within SD
    out["runs_needed_per_config"] = {f"delta={d}": runs_needed(pooled, d) for d in (0.02, 0.01)}
    (ROOT / "results/test_retest_analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("-> notes/test_retest_analysis.json")


if __name__ == "__main__":
    main()
