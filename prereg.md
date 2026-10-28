# Pre-registration — official-grader test-retest + C2 (T3) — 2026-09-29

Written BEFORE building/submitting C2 or any repeat. Frozen once Nat approves the schedule. No optional
stopping: the schedule is fixed; we report every result including nulls/negatives.

## Configs (identical except the one stated change)
- **A** = public walkthrough v2 baseline (no skill). Already submitted: id 56613027, public **0.08**.
- **C** = A + BM25 localize skill (Step-0 prompt). Already submitted: id 56626547, public **0.06**.
- **C2** = A + localize skill run with **`--ppr`** (BM25-seeded PPR file re-rank over a source call-graph)
  + test-node filter. Only difference from C is the `--ppr` flag in the Step-0 prompt + the updated script.
- All use temperature 0.2, **seed 42**, budget 4 min/80 turns.

## What repeats measure (determinism check — stated up front)
Seed is fixed (42) and temperature is 0.2. Therefore repeats of the SAME config do NOT vary sampling by
seed; they isolate the grader's **infrastructure non-determinism** (vLLM continuous-batching order +
hardware non-associativity + any concurrency in scoring). Two possible worlds, both reported:
- **World D (deterministic):** repeats identical (variance 0). Then LB scores are exact; a 1-task A−C gap
  is REAL but its significance is governed by N via MDE (still likely non-significant at N≈50–100).
- **World N (noisy):** repeats vary. Then the run-to-run SD is the noise band; LB gaps within it are not
  signal. We report the empirical SD in tasks.

## Hypotheses (pre-registered, directional where we commit)
- **H1 (primary, reliability):** the grader exhibits run-to-run variability. Prediction: SD of the public
  score across repeats of A is **> 0** (i.e. World N). We report the observed SD regardless.
- **H2 (C2 vs A, graph transfer):** we do **NOT** predict a positive effect. We pre-commit to reporting
  the raw C2−A difference and whether it exceeds (a) the repeat-derived noise band and (b) the MDE at the
  (to-be-confirmed) N. Null/negative is an acceptable, reportable outcome.
- **H3 (C vs A):** already observed C−A = −0.02 (1–2 tasks). Repeats test whether this is within noise.
- **Offline anchor (already measured, not re-tested on eval):** PPR beats BM25 on file_hit@10 (0.75→0.85,
  p_holm 0.054); this is the OFFLINE claim. H2 tests whether it appears at the AGENT level.

## Design & analysis plan
- The LB scores every config on the SAME tasks → the comparison is **PAIRED** (McNemar is the right test).
  **Limitation:** the grader returns only an aggregate rate k/N, **no per-task pass/fail**, so the
  discordant counts (b,c) are **not observable** — we cannot run McNemar directly and instead report the
  paired MDE for assumed discordance rates r∈{2%,4%,8%} (see `analysis/mde_calc.py`), with the unpaired MDE
  as a worst case. (Per-task data would require T5 traces on the free GPU; uncertain.)
- Runs per config: **5** (interleaved schedule) to estimate the grader's run-to-run SD (World D vs N).
- Steps: (1) infer N + granularity from the full set of observed scores; (2) report each config's repeat
  mean ± SD; (3) binomial/bootstrap CI on the mean; (4) MDE-vs-N table (paired + unpaired).
- Decision rule: call config X "better than Y" ONLY if mean(X)−mean(Y) exceeds the larger of
  {2×repeat-SD, paired-MDE@(N, plausible r)}. At the likely N (50–85) that threshold is not reachable for a
  0.02 effect → we expect and will report "not distinguishable at this N" (that is the finding, not a failure).

## Scope / honesty
Claims scoped to "this harness / gemma-4-31b-w4a16 / this public split". Report negative results. The
source-call-graph PPR in the skill is an APPROXIMATION of the shipped `calls`-graph used in the offline
probe (a stated limitation): C2 tests the PPR *recipe* at agent level, not the shipped graph directly.

## ADDENDUM 2026-09-29 (written BEFORE C2's score is known) — C2 interpretation rule
C2's localize `--ppr` builds a **source-derived** call-graph inside the sandbox; the offline claim
(PPR file_hit@10 = 0.847) used the **shipped** graph. These are different objects. Before interpreting
C2 we run the exact C2 skill code on all 85 EVAL tasks (`skill_eval_kernel`) and compare source-graph PPR
vs shipped-graph PPR vs BM25 (paired bootstrap, eval-once, no tuning). Pre-registered rule:
- **If source-graph PPR ≈ shipped-graph PPR on file_hit@10** (CI of the paired diff includes 0): C2 is a
  fair agent-level test of "does graph-based file localization move the resolution rate".
- **If source-graph PPR is clearly worse than shipped** (paired diff CI excludes 0, below): relabel C2 in
  `paper_plan.md` as the *source-rebuilt-graph* PPR skill; C2's LB score does NOT bear on the shipped-graph
  claim; and ADD a result "shipped graph > source-rebuilt graph" (a point about the value of the graph the
  JUDGES built — an application/relevance angle).
- Regardless, C2's LB score is read under the reliability lens (MDE): a 0.0x move vs A/C is expected to be
  within the noise floor at the plausible public N. We do NOT claim C2 "helps" or "hurts" beyond the MDE.

## ADDENDUM 2026-09-29 (T7, written BEFORE running) — external replication on SWE-bench Lite
Test generality: run the EXACT skill code (frozen from our dev split: source call-graph, BM25 k1=1.5/b=0.75,
PPR α=0.5, test-node filter — NO re-tuning) on SWE-bench Lite (300 tasks, princeton-nlp/SWE-bench_Lite).
Gold FILE = files touched by the gold patch (label only, never a feature).
- **PRIMARY (pre-registered) comparison:** source-graph PPR − BM25, **file_hit@10**, paired bootstrap 95% CI.
  Directional prediction: PPR ≥ BM25 (same sign as our competition eval, +0.094). We report the sign and CI
  whatever it is (a null/negative on Lite is a valid generality result).
- Secondary (report, not corrected): file_hit@1/5, fn_recall@10, per-repo breakdown, runtime.
- Decision: "generalizes" only if the primary CI excludes 0 in the predicted direction; else report as
  "does not replicate at file_hit@10 on Lite" honestly.

## ADDENDUM 2026-09-30 (T7 diagnostic, EXPLORATORY, written BEFORE running) — sparse call-graph
The skill's source graph is hyper-dense (identifier co-occurrence, avg degree ~200) vs the shipped graph
(~2). To test whether the PPR failure on Lite is due to DENSITY (not the diffusion idea), we build a
SPARSE call-graph from source (edges only where a unit's body AST contains a Call to another unit's name)
and run the same BM25-seeded PPR on it. Prediction: a sparse call-graph makes PPR closer to (not worse
than) BM25 on Lite. This is EXPLORATORY (mechanism only), NOT a new primary, and nothing is tuned; we
report the sign whatever it is. Result → `notes/t7_sparse.json`.

## ADDENDUM 2026-10-09 (test-retest fallback, written BEFORE seeing the remaining runs) — n=4/config
**Trigger (operational, NOT data-dependent):** if the interleaved test-retest schedule slips more than
3 days past its ~Oct 15 target for reaching n=5/config — because of grader errors and/or stuck-PENDING
submissions (e.g. C#4, PENDING >12 h and blocking new submissions) — we REDUCE the pre-registered target
to **n=4 per config** rather than delay the paper submission date.
- This is decided NOW, before the remaining runs (C#4 outcome, C2#4, C#5/C2#5) are observed, and is
  triggered by calendar slip only, never by the observed scores. At trigger time we stop at whatever runs
  have completed, requiring **≥4 valid runs for every config** (A already has 4).
- The §4 analysis is UNCHANGED except N_runs: per-config mean + bootstrap CI, between-config mean ordering
  with CIs (reported as-is, whichever way), per-config & pooled within-config SD vs binomial √(p(1−p)/N)
  (per run) / √(2p(1−p)/N) (run-to-run diff) at N=100 (N=129 sensitivity). We state n=4 and note the
  reduced power explicitly; the stuck/errored runs are reported as pipeline-reliability data, not dropped.
- If the schedule recovers within the 3-day window, we keep the original n=5 target.

## ADDENDUM 2026-10-11 (INTEGRITY CORRECTION — post-hoc disclosure, from independent review)
Honesty correction to the status of the COMPETITION-eval analysis (does NOT affect Lite or §4 prereg):
- The dev/eval split (`analysis/split.json`, commit **825a933**, 2026-09-27 20:48) was drawn **after** a probe
  had already written per-task BM25/PPR/embedding/oracle scores for all 124 tasks
  (`notes/full_probe_tasks.json` / `full_probe_results.json`, commit **045c891**, 20:42 — six minutes earlier;
  present even earlier at 8bba430). => The eval split is **EXPLORATORY, not a blind hold-out**; competition-eval
  numbers are not a pre-registered confirmation.
- The seven retriever comparisons are **pre-specified** in `analysis/probe_v3.py` (fixed before the eval re-run),
  NOT pre-registered before the data existed.
- GENUINELY pre-registered (prereg written before the generating run): **SWE-bench Lite** replication
  (this file, commit **cabc8e3** 2026-09-29 12:21, before probe script **2317052** 12:34); and the **§4 test-retest**
  schedule — with the disclosure that **A#1 and C#1 were already observed** before the test-retest prereg was written.
- All paper claims using "frozen/held-out/pre-registered/computed once" for the competition result are being
  rewritten to "exploratory"/"pre-specified" accordingly (Methods §2.3).

## ADDENDUM 2026-10-11 (Item-2 analyses, EXPLORATORY, written BEFORE running) — distinct-file metric, baselines, α sweep, lift-normalized PPR
These are EXPLORATORY (the competition split is not a blind hold-out; §2.1). Nothing is tuned to a target.
All run in one offline Kaggle kernel over the committed dev/eval split; we report every result, any sign.
Primary metric throughout: **file_hit@k on top-k DISTINCT files** (one definition, shared with Lite).
Methods to add in `analysis/probe_v3.py` and report on the eval split (dev used only where noted):
1. **distinct-file file_hit for ALL retrievers** (grep, BM25, emb-seed, PPR, two-stage, oracle) — replaces
   the pre-review node-based file_hit. If this changes the headline (incl. McNemar discordant b/c), we ADOPT
   the new numbers and adjust the text; no metric shopping (old node-based numbers kept in the result JSON for audit).
2. **BM25+dedup**: file-level BM25 (node scores aggregated to the file by max, distinct files ranked).
3. **No-query baselines** (ignore the problem statement; test a hub/popularity prior):
   (a) **dev-frequency top-10** — rank files by gold-frequency in the DEV split, return the same top-10 for every eval task;
   (b) **global PageRank top-10** — non-personalized PageRank on the (symmetrized) graph, return top-10 hubs.
4. **α sweep** α ∈ {0, 0.05, 0.1, 0.25, 0.5, 0.7} for PPR on the eval split (α = diffusion weight:
   r = (1−α)·seed + α·diffuse; α=0 ≈ BM25 seed, higher α = more diffusion). Report file_hit@{1,5,10} per α.
   (Lite α sweep with the same logic is a SEPARATE online run, not this offline kernel.)
5. **Lift-normalized PPR**: divide each node's PPR mass by its global-PageRank mass (hub prior) before ranking,
   to isolate query-specific lift from the hub prior. Report file_hit@{1,5,10}.
Prediction (not tuned): if the competition PPR gain is a hub artifact, (3b)/(4 high-α) should do well with NO query,
and (5) lift-normalization should remove PPR's apparent gain. We report regardless.

## ADDENDUM 2026-10-15 (Item-4 re-run, EXPLORATORY)
Fold into the SAME probe kernel (`kernel/probe_v3.py`, CPU, internet off), distinct-file metric:
- Report PPR full metrics (file_hit + fn_recall) AND per-task file_hit (for exact McNemar vs BM25) at the
  **frozen α = 0.5** and the **dev-selected α = 0.7** (`report["ppr_by_alpha"]`, `report["mcnemar_ppr_alpha"]`).
- Two new EXPLORATORY rankers (not pre-registered): **cosine-only** (0·BM25 + 1·cos from BM25-top-3 seeds,
  isolates the embedding over BM25) and **directed-graph PPR** (PPR on source→target edges only, vs the
  default symmetrized graph) → `report["item4_eval"]`.
- No metric shopping: the headline file_hit definition (top-k distinct files) is unchanged; these are added
  columns/ablations, adopted whatever they show.

## ADDENDUM 2026-10-15 (Item 7 — agent uptake/trace, PRE-REGISTERED before any run)
**Approved by Nat with limits (2026-10-15).** Written BEFORE running the agent, to pre-register the design.

**Goal.** Move the agent arm from intention-to-treat toward uptake: does attaching the `localize` skill
change (a) whether the skill is called, (b) whether the gold file is among the files the agent opens/edits,
and (c) per-task pass/fail — on a small dev sample.

**Design (fixed in advance).**
- **Tasks:** a fixed, pre-specified **stratified** sample of 20 DEV tasks (round-robin over per-repo sorted
  `analysis/split.json` dev_ids until 20 → 8 fastapi, 8 rich, 4 requests; the first-20-sorted would be all
  fastapi, so we stratify). DEV only; eval stays frozen for §3.1. The exact 20 ids:
  (20 dev ids, stratified 8 fastapi / 8 rich / 4 requests; the exact list is regenerated by
  analysis/make_split.py — first 8/8/4 of the per-repo sorted dev_ids, round-robin).
  _[2026-10-28 note: the explicit id list was moved out of the public copy (no instance_ids are
  released); the rule above — string-sorted per repo, round-robin fastapi → rich → requests —
  reproduces the original pre-registered list exactly.]_
- **Configs:** **A** (walkthrough baseline) vs **C2** (A + `localize --ppr`) only. No C.
- **Compute:** one Kaggle **L4x4** notebook attached to the competition, **≤ 6 GPU-h total** (≈ 40 agent
  runs). If the budget is hit, report whatever completed and mark the rest missing (no re-run for a better
  number).
- **Metrics (per task, per config):** skill-call rate (did the agent invoke `localize`?); gold-file-opened
  (is any gold file among the files the agent reads/edits?); pass/fail (harness verdict). Report as rates
  over the 20 tasks + a 2×(A,C2) table.
- **Analysis:** descriptive only (rates + McNemar on paired pass/fail across the 20 tasks). This is a small
  dev probe; **it must NOT change any headline claim** (§3.1–§3.5 stand regardless).
- **Reporting:** if completed by **2026-10-24**, add ≤ 1 paragraph + 1 small table within the ≤ 2,850-word
  budget; otherwise DROP it entirely (no partial claims in the paper).
- **Exploratory flag:** uptake/trace metrics are a new instrument; treated as exploratory evidence about
  mechanism, not confirmatory.

### Item 7 — DROPPED 2026-10-15 (CPU smoke test failed; no GPU spent)
The CPU-only smoke test (`subkernel/item7_smoke/`, Kaggle `gemma4-item7-cpu-smoke`, competition-attached,
internet off, **no GPU**) confirmed the grader harness is **unavailable**, so `swegemma eval` cannot run:
- `swegemma`, `adk-submission`, `adk-eval-core`: NOT on PyPI; NOT in the competition data (no wheel/source);
  NOT importable in the Kaggle competition-notebook image; `swegemma` CLI not on PATH. Only base `google.adk`
  is present (not the competition harness).
- Internet is off on competition notebooks, and we have no source to build the wheels → no install path.
The trace-parse logic itself validated OK on a synthetic SessionTrace. Per the plan, Item 7 is **dropped**;
the paper is unchanged (headline claims stand; §3.4 agent arm remains intention-to-treat). The authored
runner (`subkernel/item7_uptake/run_item7.py`) and this prereg are kept as a documented, reproducible plan.
