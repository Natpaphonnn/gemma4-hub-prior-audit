"""Minimum detectable effect (MDE) for pass/fail agent-benchmark comparisons.

Two designs:
  unpaired: two independent proportions (worst case; what the public LB gives if we ignore that both
            configs run the SAME tasks). Normal-approx MDE at 80% power, alpha=0.05 two-sided.
  paired  : McNemar on the SAME tasks. Given a discordance rate r (fraction of the N tasks where the two
            configs disagree), the detectable marginal difference delta=(b-c)/N. Computed by EXACT
            McNemar (binomial) power via simulation + bisection on delta.

Public LB caveat: the grader returns only an aggregate rate, so discordance b,c cannot be observed
directly — the paired numbers are what WOULD be needed given an assumed discordance.

Run: .venv/bin/python analysis/mde_calc.py  -> prints tables, writes notes/mde_table.json
"""
import json
from math import sqrt
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(0)
Z_A, Z_B = 1.959963985, 0.841621234  # alpha=0.05 two-sided, power=0.80


def unpaired_mde_normal(N, p=0.08):
    """Normal-approx MDE (anti-conservative at low p / small N). Kept only for comparison."""
    return (Z_A + Z_B) * sqrt(2 * p * (1 - p) / N)


def _power_fisher(N, p1, p2, reps=2000):
    """Power of Fisher's exact test (alpha=0.05 two-sided) to separate p1,p2 at sample size N/group."""
    a = RNG.binomial(N, p1, reps)
    b = RNG.binomial(N, p2, reps)
    sig = 0
    for ai, bi in zip(a, b):
        _, pv = stats.fisher_exact([[int(ai), int(N - ai)], [int(bi), int(N - bi)]])
        sig += pv < 0.05
    return sig / reps


def unpaired_mde(N, p=0.08):
    """EXACT (Fisher) unpaired MDE: smallest delta with power>=0.80, by bisection. Correct at low p."""
    if _power_fisher(N, p, min(1.0, p + (1.0 - p))) < 0.80:
        return None
    lo, hi = 0.0, 1.0 - p
    for _ in range(18):
        mid = (lo + hi) / 2
        if _power_fisher(N, p, p + mid) >= 0.80:
            hi = mid
        else:
            lo = mid
    return hi


def mcnemar_p(b, c):
    """Exact two-sided McNemar (binomial) p-value for discordant counts b,c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial about 0.5
    p = 2 * stats.binom.cdf(k, n, 0.5)
    return min(1.0, p)


def paired_power(N, r, delta, reps=3000):
    """Power of exact McNemar at alpha=0.05 to detect marginal diff delta, given discordance rate r.
    p_b=(r+delta)/2, p_c=(r-delta)/2 per task (independent discordance indicators)."""
    pb, pc = (r + delta) / 2, (r - delta) / 2
    if pc < 0 or pb > 1:
        return np.nan
    b = RNG.binomial(N, pb, reps)
    c = RNG.binomial(N, pc, reps)
    sig = np.array([mcnemar_p(bi, ci) < 0.05 for bi, ci in zip(b, c)])
    return sig.mean()


def paired_mde(N, r):
    """Smallest delta with power>=0.80 (bisection on [0, r]); None if unreachable even at delta=r."""
    if paired_power(N, r, r) < 0.80:
        return None
    lo, hi = 0.0, r
    for _ in range(24):
        mid = (lo + hi) / 2
        if paired_power(N, r, mid) >= 0.80:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    Ns = [50, 58, 85, 100, 129, 300, 500]  # 58 = inferred public-LB graded-task count
    rates = [0.02, 0.04, 0.08]
    out = {"unpaired": {}, "unpaired_normal_approx": {}, "paired": {},
           "note": "baseline p=0.08; alpha=0.05 two-sided; power=0.80. 'unpaired' = EXACT Fisher MDE "
           "(correct at low p); 'unpaired_normal_approx' = normal approximation (anti-conservative, "
           "for comparison only). delta is an ABSOLUTE score difference (resolve-rate points). LB gives "
           "no per-task data → discordance is assumed, not observed."}
    print("UNPAIRED MDE (absolute score diff, 80% power, p=0.08) — EXACT Fisher vs normal approx:")
    for N in Ns:
        m = unpaired_mde(N)
        mn = unpaired_mde_normal(N)
        out["unpaired"][N] = round(m, 4)
        out["unpaired_normal_approx"][N] = round(mn, 4)
        print(f"  N={N:4d}: exact={m:.3f}  normal={mn:.3f}")
    print("\nPAIRED (McNemar) MDE by discordance rate r = fraction of N tasks where configs disagree:")
    print(f"  {'N':>5} " + "  ".join(f"r={r:.0%}" for r in rates))
    for N in Ns:
        row = []
        out["paired"][N] = {}
        for r in rates:
            m = paired_mde(N, r)
            out["paired"][N][r] = None if m is None else round(m, 4)
            row.append("n/a " if m is None else f"{m:.3f}")
        print(f"  {N:>5} " + "  ".join(f"{x:>6}" for x in row))
    # contextual read for A vs C (observed 0.02)
    print("\nObserved A-C = 0.02. Detectable?")
    for N in (50, 100):
        u = unpaired_mde(N)
        print(f"  N={N}: exact unpaired MDE={u:.3f} -> 0.02 is {u/0.02:.1f}x below MDE")
        for r in rates:
            m = paired_mde(N, r)
            tag = "n/a" if m is None else f"{m:.3f} ({'DETECTABLE' if 0.02>=m else 'below'})"
            print(f"       paired r={r:.0%}: MDE={tag}")
    (ROOT / "results/mde_table.json").write_text(json.dumps(out, indent=1))
    print("\n-> notes/mde_table.json")


if __name__ == "__main__":
    main()
