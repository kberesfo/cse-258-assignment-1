# Assignment 1 plan: reach ~22 / 33 (comprehensive exam bar)

**Goal:** 65–70% on Assignment 1 ≈ **22 / 33** (professor: a 4.0, a 3.5 and a 3.0 should pass).
**Deadline:** start of week 8 (exact time on Gradescope).
**Rule:** Homework 3 is already passed (6/6 thresholds). Do not resubmit or change it. Everything below is for Assignment 1 only.
**Status:** Tasks 1, 2 and 3 are at the **4.0** public tier; Tasks 4 and 6 at 3.5; Task 5 at 3.0. The bar is met with a full spare. Remaining: update the writeup (Task 4 paragraph, final numbers), tag, and optionally push Task 6 to 4.0.

---

## 1. How it's graded (as understood)

| Component | Marks | Source |
|---|---|---|
| Best 3 tasks × private (hidden) leaderboard tiers | 3 × 4 = 12 | professor's post (1 of the 4 for any valid upload) |
| Best 3 tasks × public leaderboard tiers (private tiers ÷ 2) | 3 × 2 = 6 | professor's post ("likely similar but divided by 2") |
| Best 3 tasks × leaderboard placement | 3 × 4 = 12 | original spec; total 3 × (4 + 2 + 4) + 3 = 33 |
| Writeup | 3 | spec ("generally get full marks automatically") |

**Placement marks (reported, not confirmed by me):** the leaderboard is split into uniformly sized buckets in 0.5-mark steps (8–9 buckets); top bucket 4/4, next 3.5/4, and so on. If "uniform" means equal numbers of students per bucket, 8 buckets ≈ 12.5% each: top 12.5% → 4.0, next 12.5% → 3.5, … Open question: per task, and on the public or hidden leaderboard?
**Best 3 tasks:** reported as taken automatically (the best three). Naming them in the writeup costs nothing; done in the header line.

Confirmed on Piazza: Tasks 1/3 MSE, Tasks 2/4/5 accuracy, Task 6 MAE; the visible leaderboard is **not** the hidden test; 0-ratings are skipped in the leaderboard MSE (@117).
Confirmed in lecture: exploiting how the test was constructed is allowed ("find my mistakes and exploit them").

- [ ] **Confirm on Piazza / with the professor:** placement buckets: equal counts or equal score ranges? Per task? Public or hidden board? Are the best 3 picked automatically?

### Tiers (private; public = half)

| Task | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 |
|---|---|---|---|---|---|---|
| 1 (MSE ↓, 0-ratings skipped) | 0.64 | 0.62 | 0.60 | 0.58 | 0.56 | 0.54 |
| 2 (acc ↑) | 0.87 | 0.90 | 0.91 | 0.92 | 0.93 | 0.94 |
| 3 (MSE ↓, 0-ratings skipped) | 0.81 | 0.78 | 0.75 | 0.73 | 0.71 | 0.69 |
| 4 (acc ↑) | 0.80 | 0.83 | 0.86 | 0.89 | 0.92 | 0.95 |
| 5 (acc ↑) | 0.81 | 0.83 | 0.85 | 0.87 | 0.91 | 0.92 |
| 6 (MAE ↓) | 13M | 12M | 11M | 10M | 9M | 8.5M |

---

## 2. Where I stand (public scores, 2026-11-12)

**Placement (2026-11-12):** ~631 on the leaderboard. Best per-task public ranks: Task 1 **92**, Task 3 **90**, Task 6 **89** (down from ~50 as others improve). With 8 equal buckets of ~79: top bucket (4/4) = rank ≤ 78; ranks 79–157 → 3.5/4. All three sit at ~14–15% → **3.5 each**, 11–14 places from 4.0. Ranks will keep sliding until the deadline; the hidden board may reorder.

| Task | Score | Tier | Margin | History |
|---|---|---|---|---|
| 1 | **0.5296** | **4.0** | 0.0104 under 0.54 | 0.6185 → 0.5917 (drop 0s) → 0.5501 (stack: bias + cosine kNN + stats) → 0.5453 (Jaccard + adjusted-cosine kNN) → 0.5443 (+ raw-rating kNN) → **0.5296** (top-k item-kNN, user-kNN, RatingProfile, 5-fold out-of-fold, tuned GB) |
| 2 | **0.9454** | **4.0** | +0.0054 over 0.94 | 0.9066 (BPR) → 0.9302 (LR combiner + cosine) → 0.9407 (+ seen-by-user rule) → 0.9410 (+ time features, BPR K=10) → **0.9454** (EASE + time-local cosine, GB combiner; `plan_t2.md` P6) |
| 3 | **0.6881** | **4.0** | 0.0019 under 0.69 (thin) | 0.7921 → 0.7626 (drop 0s) → 0.7229 (stack: time-bias + kNN + stats + session) → 0.7084 (Task 1's kNN sources) → **0.6881** (Task 1's new sources + time-weighted top-k + sequence residual) |
| 4 | **0.9345** | **3.5** | +0.0145 over 0.92 (4.0 needs 0.95, +0.0155) | 0.8706 (BPR + rule) → 0.8984 (GB combiner) → 0.9172 (+ seen-by-user) → 0.9277 (+ cosine features) → **0.9345** (+ time-local cosine; `plan_t4.md`) |
| 5 | 0.9030 | 3.0 | 3.5 needs 0.91 (+0.007) | 0.8668 → 0.8703 (rules) → 0.8905 (GB combiner) → 0.8945 (+ seen-by-user) → 0.9030 (+ cosine) |
| 6 | **8,542,010** | **3.5** | **42k (0.5%) from 4.0** at 8.5M | 13.01M → 11.39M (bound) → 11.21M (offset) → 10.65M (`GapRegressor`) → 8.76M (+ next book's reader times) → **8.54M** (<what changed>) |

**Best 3:** Tasks 1, 2, 3 at 4.0 → private 12 + public 6 = **18 / 18** if private matches public, writeup **3 / 3**, placement **? / 12**.
**Slack:** Tasks 4 and 6 are spare 3.5s. If Task 3 (the one thin margin) slips to 3.5 on private: 4.0 + 4.0 + 3.5 → private 11.5, above the professor's bar.
**Thin margins:** Task 3 (0.0019) only. Task 1 (0.0104) and Task 2 (0.0054, was 0.001) are solid 4.0s.

**Validation → public calibration** (how estimates held up):
| Task | Validation change | Public change | Note |
|---|---|---|---|
| 1 | outer holdout 0.5430 → 0.5299 (−2.4%) | 0.5443 → 0.5296 (−2.7%) | relative gain carries over, slightly larger on public |
| 2 | 0.9371 → 0.9383 (+0.001, four splits) → +0.0027 (P6, four splits) | 0.9407 → 0.9410 (+0.0003) → 0.9454 (+0.0044) | small gains shrank; the P6 gain grew (1.6×) |
| 3 | outer holdout 0.6931 → 0.6760 (−2.5%, 4/4 splits) | 0.7084 → 0.6881 (−2.9%) | relative gain carries over |
| 4 | 0.9173 → 0.9227 (+0.0053, both directions) | 0.9277 → 0.9345 (+0.0068) | additive gain carries over, slightly larger |
| 5 | 0.8932 → 0.9029 (+0.010) | 0.8945 → 0.9030 (+0.009) | additive gain carries over |
| 6 | 8.01M → 7.18M (−10.4%) | 10.65M → 8.76M (−17.7%) | relative gain larger on public |

**Rule predictions (`notebooks/rules.ipynb`) vs leaderboard:** Task 4 +1.97 predicted / +1.88 observed; Task 2 ~0.9406 / 0.9407; Task 5 ~0.8948 / 0.8945.

---

## 3. Lecture hints (professor, week 6) and how they played out

- **Top solutions ensemble standard models** (meta-model on user/item features + several models' predictions) → `StackedRating` and the pairwise combiners. SVD (K = 1, 5) as members: no gain. Top-k item-kNN, user-kNN and rating profiles as sources: −2.4% / −2.5% on Tasks 1 / 3.
- **MF and BPR with small K** → MF never beat the bias model on Task 1, alone or as a stack member. BPR K = 10 as a second member: +0.001 on Task 2, nothing on Task 5.
- **Weighted-average / Pearson models "probably not winners"** → on this data item-kNN residual features were the largest Task 1/3 gain.
- **Temporal / sequence tasks may gain little** → time-bias base only −0.6% on Task 3, but the user's *time-local* signal helped: sequence residual (Task 3), time-local cosine (+0.005 Task 4). Recent-history similarity gave nothing on Task 5. FPMC not attempted.
- **Task 6: MAE loss, time-series regression on the previous deltas** → `GapRegressor`, then the next book's reader times (−10.4% val, −17.7% public).
- **Exploit test-construction mistakes** → seen-by-user and unseen-item rules.

---

## 4. Next steps

### N1. Writeup (+3)
- [x] `writeup.md` drafted (approach, six task paragraphs, rules, what didn't work).
- [x] Task 4 paragraph: time-local cosine + EASE, val 0.9173 → 0.9242, public 0.9277 → 0.9345.
- [x] Task 2 paragraph: EASE, time-local cosine at the candidate's median time, GB; public 0.9454.
- [x] Task 6's public score (8.54M) and what changed (127 leaves, × 1.07).
- [x] Section 4 ("what didn't work"): calendar affinity, LightFM, time-local EASE.
- [x] Header: best three tasks (1, 2, 3). Approach: all combiners are now GB.
- [ ] `make submit` copies `writeup.md` → `dist/writeup.txt` (the copy in dist/ is stale).

### N2. Optional: Task 6 → 4.0 (42k / 0.5%): `plan_t6.md` round 5
- [ ] Target ≈ −0.4% on backtest A (public has moved 1.1–1.4× validation in the last two rounds).
- [ ] R1 diagnostics (error by gap bucket: gaps > 365 d are 10% of rows, 50% of total gap time)
      → R2 per-bucket scale → R3 next-book similarity to recent books (Tasks 2/4 lesson)
      → R4 cap at data end → R5 more seeds.
- [ ] Already done: GB tuning (round 3; only 127 leaves helped), scale (round 4).
- [x] Round 5 run: best −0.20% A / −0.41% B (last-reads similarity), half the target. R1: the
      remaining error is unforeseeable lapses; calibration is already good. R6 (+ cap, own
      scale): −0.22% / −0.42%. **Stopped, not shipped**: still 3.5 (≈ 8.52M), and placement only
      counts for the best three tasks.

### N3. Optional: Task 4 round 2 (margin only; `plan_t4.md`)
- [ ] Candidate reader timing near t; before / after t split; BPR-factor local similarity. Realistic total +0.005–0.010; 4.0 needs +0.0155.

### N4. Before the deadline
- [ ] Commit and tag the final state (`a1-final`); confirm the last upload matches `dist/`.
- [ ] Re-run `make check` once more on a clean checkout.

### Not pursuing (documented as future work)
- FPMC for Task 5 (factorized transitions; uncertain gain on weakly sequential data).
- Further Task 2 margin: done (P6, 0.9410 → 0.9454).
- EASE (Task 4: +0.001, below the +0.002 bar); LightFM (≤ +0.001, non-deterministic); fastFM (no Python 3.12 build).
- GANs / deep autoencoders (published re-evaluations: not consistently better than tuned simple baselines).

---

## 5. Done ✔
- [x] Tasks 1 & 3: drop 0-ratings (train + validate on nonzero rows).
- [x] Task 1 LFM: bias λ optimal; SVD best ~0.597 even with separate λ and full convergence; factors don't help alone.
- [x] Task 1 error analysis: 1–2 star ratings = 7% of rows, 36% of error → user-specific dislikes.
- [x] Task 1 stack v1: bias + item-kNN on residuals + `RatingStats`, GB on residuals (public 0.5443).
- [x] Tasks 1 & 3 stack v2 (`task1_ablation`, `task3_ablation`): `TopKItemKNN`, `UserKNN`, `RatingProfile`, 5-fold out-of-fold combiner, tuned GB; Task 3 adds time-weighted top-k and `SequenceResidual`. Task 3 leave-one-out: every source beats noise; 4/4 seeds. Public 0.5296 / 0.6881 (both 4.0).
- [x] Task 2: LR pairwise combiner on BPR, popularity, cosine sum/max/mean, time features relative to t_last, BPR K = 10 (val 0.9383, gain consistent over four splits). Jaccard, top-k cosine, span share: no gain.
- [x] Task 4: GB combiner + seen-by-user rule + cosine features (val 0.9173) + time-local cosine `LocalSimilarity` (val +0.0053; public 0.9345). Calendar affinity, LightFM, EASE: below the bar.
- [x] Task 5: GB combiner (transitions, t_last query time) + rules + cosine (val 0.9029). Recent-history similarity (m = 1–10) and BPR K = 10: no gain.
- [x] Task 6: `GapRegressor` (GB, MAE on log1p gaps; lags, medians, GapModel prediction) + the next book's reader times; first-seen bound as a feature only (as a hard floor it now costs accuracy). Val 7.18M; public 8.54M.
- [x] Seen-by-user rule (Tasks 2/4/5) and unseen-item rule (Task 5; guard on 2/4).
- [x] Class refactor: `RatingModel` / `FeatureSource` / `SurpriseModel` (`BiasModel`, `SVDModel`), `TimeBiasModel`, `ItemKNN`, `TopKItemKNN`, `UserKNN`, `RatingStats`, `RatingProfile`, `SessionResidual`, `SequenceResidual`, `ModelPrediction`, `StackedRating` (n_folds); `BPRModel`, `TransitionModel`, `PairFeatures`, `LocalSimilarity`; `GapModel`, `GapRegressor`, `split_last`; shared `models/similarity.py`.
- [x] Makefile: automatic `dist_history/` snapshots; `make changes`; `TASKS=` for single-task runs.
- [x] Notebooks rebuilt on the classes (Tasks 1–6), plus `rules.ipynb`, `task2_diagnostics.ipynb`, ablation notebooks.
- [x] Sectioned docstrings for Tasks 1–6.
- [x] Notes: techniques page (kNN correction, gap regression, BPR extensions).

## 6. Checklist before every upload
- [ ] `make check` (or `make check TASKS=n`): all prediction files report "matches".
- [ ] `make changes`: only the tasks you meant to change show changed rows.
- [ ] Re-run `rules.ipynb` if a ranking task changed.
- [ ] Record new public scores in section 2; commit and tag.

## 7. Path to 22
- Tiers (best 3 = Tasks 1, 2, 3): **18 / 18** if private matches public. With the writeup: **21 / 21** → **1 of 12** placement marks needed.
- Thin margins slip (Tasks 2 and 3 both to 3.5 on private, public stays): best 3 private 4.0 + 3.5 + 3.5 = 11 → 11 + 6 + 3 = **20** → ~2 placement marks needed.
- Under the reported buckets (4.0 down in 0.5 steps), even the lowest non-zero bucket on all three tasks gives 3 × 0.5 = 1.5 marks; 2 marks needs only one task one bucket above the bottom. The tiers carry the exam bar.
- Current estimate (N ≈ 631, ranks 89–92 → 3.5 each): placement ≈ **10.5 / 12** → total ≈ 18 + 3 + 10.5 = **31.5 / 33**. To hold 3.5, a task must stay within the top ~25% (rank ≤ 157), so it can absorb ~65 more students passing it. If placement counts all ~1,000 enrolled (non-submitters at the bottom), the top bucket is ~125 and these ranks would already be 4.0.