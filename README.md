# Diffusion finds hubs, not bugs — a fair audit of code-localization retrieval

Code and scripts for the Paper Track submission. **Code and aggregate results only — no competition
data and no per-task labels are included or redistributed.** A participant with competition access
regenerates every label, split, and per-task result by running these scripts. Licensed under
Apache-2.0 (`LICENSE`).

> Status: prepared for release, **not yet published**. Nothing here is public until the authors publish it.

## What this is

A fair audit of the shipped code-intelligence tools (grep, BM25, the embedding tool, a `calls`-graph
Personalized PageRank) for file/function localization, with a reliability analysis (minimum detectable
effect, test-retest of the grader) and an external replication on SWE-bench Lite.

**Headline.** Under the correct **distinct-file** metric, at the frozen diffusion weight α = 0.5 PPR
**exactly ties BM25** on file_hit@10 (0.847 = 0.847; exact McNemar 2 vs 2, p = 1.0); only a dev-selected
α = 0.7 nudges it to +0.024 (McNemar 3 vs 1, p = 0.625 — still null). A no-query **hub/popularity prior**
alone reaches file_hit@10 ≈ 0.71 on the four competition repos but only ≈ 0.10 on SWE-bench Lite's twelve
larger repos — so the same recipe **reverses sign** on Lite (PPR − BM25 file_hit@10 = −0.263
[−0.320, −0.207]). The public leaderboard grades an inferred **N = 58** tasks, so its exact MDE is ≈ 0.21:
it cannot resolve the ~0.02 effects competitors chase. Test-retest of the grader: ICC(1,1) ≈ 0.07
(~93 % of score variance is run-to-run).

One methodological point recurs: the file_hit@k **metric definition** (top-k nodes vs top-k distinct
files) flipped the headline from significant to null.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt    # numpy, rank_bm25, unidiff, matplotlib, scipy
```

Competition data is NOT shipped. Obtain it from Kaggle and extract it to `./data` (the scripts
auto-detect `./data` or a Kaggle mount). We do not redistribute it.

## Reproduce

```bash
# 1. Regenerate the gold file/function labels from the public patches (needs ./data; no labels shipped)
python analysis/ground_truth.py

# 2. Regenerate the frozen dev/eval split from the data (deterministic; seed baked in).
#    The evaluable task list is recomputed from ./data (tasks.jsonl + graphs/); no split file is shipped.
python analysis/make_split.py        # -> analysis/split.json
#    Verify your regeneration byte-for-byte:
shasum -a 256 analysis/split.json
#    expected SHA-256 = 49e7303c605619c09efe19368a22845526678e8c52d627aec35d36c7604fe867

# 3. Competition retrieval audit — distinct-file metric, α sweep, no-query hub baselines, lift-PPR,
#    exact McNemar, PPR at α=0.5/0.7, cosine-only + directed-graph PPR ablations.
#    Needs ALL graphs/embeddings, so run it on Kaggle with the competition mounted (see "Kaggle notebook"
#    below). Locally it runs on whatever graphs happen to be in ./data.
python analysis/probe_v3.py          # -> results/probe_v3_results.json, results/mcnemar_holm.json

# 4. Minimum detectable effect (exact Fisher + paired McNemar), N = 50..500 incl. the inferred N = 58
python analysis/mde_calc.py          # -> results/mde_table.json

# 5. External replication on SWE-bench Lite (clones public repos at base_commit; same distinct-file metric)
python analysis/swebench_lite_probe.py   # -> results/swebench_lite_results.json

# 6. Test-retest analysis of the grader (reads the shipped test_retest.csv)
python analysis/test_retest_analysis.py  # -> results/test_retest_analysis.json

# 7. The drop-in localizer skill (pure stdlib; standalone — no competition data needed)
python skill/localize/scripts/localize.py --root <repo_path> "ISSUE TEXT"        # BM25
python skill/localize/scripts/localize.py --root <repo_path> --ppr "ISSUE TEXT"  # BM25-seeded PPR

# 8. Figures
python analysis/make_fig1_hub.py     # -> results/fig_paper_hub.png  (Fig. 1, hub mechanism)
python analysis/make_paper_figs.py   # -> results/fig_paper_mde.png  (Fig. 2, MDE vs N)
```

## Kaggle notebook (full run)

The full competition audit needs every graph/embedding, so run it on a Kaggle **CPU** notebook with the
competition attached (internet off). `subkernel/kernel-metadata.json` is the script-kernel metadata; the
public "Copy & Run" notebook is linked in the submission writeup.

## Layout

- `analysis/` — probes, MDE calculator, split/label generators, Lite replication, figure scripts, diagnostics
- `skill/localize/` — the drop-in localizer (`SKILL.md` + `scripts/localize.py`, `--ppr` flag)
- `subkernel/` — Kaggle submission-build scripts + kernel metadata
- `results/` — **aggregate** result JSONs that back the paper (no per-task data): `probe_v3_results.json` (per-task stripped),
  `swebench_lite_results.json`, `mde_table.json`, `test_retest_analysis.json`, `mcnemar_holm.json`
- `prereg.md` — pre-registration (incl. integrity/exploratory disclosures)
- `test_retest.csv` — the grader test-retest log (per-run public scores)

## Honesty notes

- The competition dev/eval split is **exploratory**, not a blind hold-out (it was drawn after per-task
  results existed; see `prereg.md`). Only the SWE-bench Lite replication and the test-retest were
  pre-registered before the generating run.
- file_hit@k is measured on top-k **distinct files** throughout; node-based figures appear only where
  explicitly labelled as exploratory.
- The graded set is hidden (private repositories), so the agent-level comparison is out-of-distribution
  by construction and read as intention-to-treat (the grader exposes no traces).
