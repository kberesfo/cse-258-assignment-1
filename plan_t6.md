# Task 6 plan: reach the 4.0 tier (MAE ≤ 8.5M) and move up from rank 140

**Now (2026-11-12):** public **8,542,010** (tier 3.5; 4.0 needs 8.5M: **−0.5%**). See round 5.
**Before round 1:** public 8.76M MAE (tier 3.5; 4.0 needs 8.5M, about −3%). Backtest A 7,182,796,
backtest B 7,191,137.
**Model:** `GapRegressor`, gradient boosting with MAE loss on log1p(gap). Features: the user's
gaps, the next book's reader times relative to the user's last interaction, and GapModel.
**Lesson from earlier work:** the next item's reader times were the biggest single gain
(−12%: 8.15M → 7.18M without/with). The item side carries a lot of signal, so P2 and P3 add
more of it.

## Steps
All are options on `GapRegressor`; the defaults reproduce the shipped model. To ship, set
`TASK6_REGRESSOR_PARAMS` in `tasks/task6.py`.

### P1. More training targets: `n_steps`
- The regressor currently trains on one target per user (55k rows). Step k holds out each
  user's k-th last gap and builds features from the history before it; all steps are stacked.
- Backtest B (7.19M) is almost equal to A (7.18M), so features stay comparable across steps and
  2–3× the training rows should help the trees.

### P2. How quickly other readers reached the next item: `item_gaps`
- For the next item: median / quartiles / count of other readers' log gap from their previous
  interaction to this item. Some books are read straight after the previous one (series, hyped
  releases), others after a long pause.

### P3. Co-readers of the last item and the next item: `coread`
- Among users who read both the user's last item and the next item: how many, the share who read
  the next item after the last one, and quartiles of that delay (log days). Catches series and
  back-to-back reads.

### P4. Seed bagging: `n_bag`
- Average the log predictions of b regressors whose random seeds differ (the seed sets the
  early-stopping validation split). Small, steady gain at most.

### P3b. Co-read delay over the user's last k books: `coread_k`
- Round 1 found the co-read delay was the only new signal, but it exists for only 39% of rows.
  Pool it over the last k books: a co-reader v of history book a implies the next read at
  t_ua + (t_vb − t_va). On a 5k-user sample, k = 3 raises coverage 18% → 29%.

### P5. Tune the regressor (round 3, `notebooks/task6_tune.ipynb`)
- After co-read: public 8.64M, rank 140 → 107. Tier 4.0 needs ≤ 8.5M (−1.6%); public moved
  ~1.4× the validation gain last time, so the target is about −1.2% on backtest A.
- The regressor settings were never tuned. Features are cached once per backtest and every
  setting is scored as a 3-seed bag on the cached matrices (seed noise ~0.3%).
- T3 one setting at a time → T4 combine → T5 confirm on B → T6 scale diagnostic.

### P6. Prediction scale (round 4, `notebooks/task6_scale.ipynb`)
- Held-out gaps grow toward the end of each user's history: median 43.9 days (backtest B) →
  49.1 (A); the test is one step later (t_last median 2016-09 vs 2016-04 on A; 19.0% of test
  items first appear after t_last vs 17.1% on A). The model learns from the gap one step
  before the one it predicts, so it predicts too short (best scale on A ×1.05).
- New option `GapRegressor(scale=...)`; pick it on A (same relation as the test), check on B.
  Combined with 127 leaves.

### Later, if needed
- Look into the public/validation gap (8.76M vs 7.18M, 1.22×): are the test gaps later in
  calendar time? `to_end_d` is the feature that covers this.

### Round 5: the last −0.5% (`notebooks/task6_round5.ipynb`)
**Target:** public moved 1.1–1.4× the validation change (−1.4% vs −1.0%; −1.1% vs −0.73%), so
**≈ −0.4% on backtest A**, also negative on B. That's about seed-noise size for one seed (~0.3%),
so every comparison uses the same 5 seeds on cached features (paired), as in round 3.

**Facts (checked 2026-11-12):**
- Error is concentrated in long gaps: on backtest A, gaps > 365 days are 9.8% of rows but 50.5% of
  total gap time; > 90 days are 36.7% of rows and 87.2%. Short-gap users barely move MAE.
- The next book is never a re-read (0.00% of test and backtest rows): nothing to exploit.
- 93.1% of test next books have other readers in train (median 45). Similarity features would
  cover most rows; the co-read delay covers 63%.

**R1. Diagnostics first (no model change)**
- MAE share and median signed log error by true-gap bucket (< 30 d, 30–90, 90–365, > 365) and by
  predicted-gap bucket. Is the 1.07 scale right everywhere, or are long predictions still short?
- How often the prediction exceeds the time left in the data (`to_end_d`).
- Error by segment: next book first appears after t_last (bound active) / co-read defined / not.

**R2. Scale by predicted gap (post-hoc, cheapest)**
- One scale per predicted-gap bucket (2–3 buckets), picked on B and checked on A, as round 4 did
  for the single scale. Worth it only if R1 shows bias differs by bucket. Overfit risk: keep it
  to ≤ 3 numbers and require the gain on both backtests.

**R3. Similarity of the next book to the user's recent books (Tasks 2 / 4 lesson)**
- Binary cosine between the next book and the user's last 1 / 3 / 10 books (max and mean), and
  the EASE score of the next book over the whole history (`ease_similarity`, float32: 192k items
  is too many for full EASE, so restrict to the top ~30k books or skip EASE).
- Why it could help: in Task 4, similarity to the reads near t was the largest gain. A next book
  that is close to the last few (series, same author) suggests binge reading and a short gap;
  a distant one suggests a fresh start, possibly after a lapse. The co-read delay measures this
  only when someone read both books; cosine covers 93% of rows.

**R4. Cap at the end of the data**
- If R1 finds predictions beyond `to_end_d`, clip them to it. The trees see `to_end_d`, but
  bagging and the ×1.07 scale can push predictions past it. Hard rules have hurt before (the
  first-seen floor), so keep only if it wins on both backtests.

**R5. More seeds**
- n_bag 5 → 10. Variance only; expect ≤ 0.1%. Last, on the final configuration.

**Ship if** the combination wins on both backtests and is ≤ −0.4% on A (same seeds). Otherwise
stop: Task 6 is a spare (best three are Tasks 1, 2, 3 at 4.0), so the gain is insurance only.

**Round 5 results** (5 seeds, paired; reference = shipped: A 7,062,989, B 7,058,508)
| Variant | Δ% A | Δ% B | Note |
|---|---|---|---|
| R2 2 buckets (60 d) | −0.09 | −0.00 | nothing |
| R2 3 buckets (30 / 120 d) | −0.27 | **+0.07** | fails on B: per-bucket scales unstable (30–90 d: 1.17 on A, 1.05 on B) |
| R3 similarity, all 16 cols | −0.17 | −0.34 | |
| **R3 last-reads similarity only (13 cols)** | **−0.20** | **−0.41** | best; whole-history cosine adds noise |
| R4 cap at data end | −0.01 | −0.00 | caps 0.03% / 0.00% of rows: the trees already respect `to_end_d` (first run NaN: 27 users without in-range timestamps; fixed with `np.fmin`) |
| R5 all similarity, 10 seeds | −0.16 | −0.33 | = 5 seeds |
| R6 last-reads + cap | **−0.22** | **−0.42** | best overall |
| R6 last-reads, own scale (± cap) | −0.20 (−0.22) | −0.41 (−0.42) | best scale unchanged (A 1.075, B 1.05): no recalibration needed |
| R5 last-reads, 10 seeds | −0.20 | −0.40 | = 5 seeds |

**R1 findings:**
- By true gap, 90+ day gaps are 37% of rows and 80% of the error (90–365 d: 33%; 365+ d: 47%),
  predicted at ~45% / ~30% of the true gap (median log ratio −0.8 / −1.2). Gaps < 7 d are
  overpredicted ~7× (+1.97). This is the median predictor meeting unpredictable lapses.
- By predicted gap, calibration is good: median log ratio within ±0.16 in every bucket on both
  backtests. No stable structure left for post-hoc scaling (R2).

**Conclusion:** the remaining error is lapses the features cannot foresee. The best variant
(−0.20% A) is half the −0.4% target: expected public ≈ 8.515–8.52M, still tier 3.5.
**Stopped; nothing shipped.** The best variant (last-reads similarity + cap, −0.22% A /
−0.42% B) is consistent but would give ≈ 8.52M public, still 3.5. Placement counts only for the
best three tasks (1, 2, 3), so a better Task 6 rank earns nothing; not worth a code change and
upload. If ever needed: 13 `loc_*` columns from `LocalSimilarity(binary_cosine)` queried just
after t_last, fitted on the same history as the other features (`notebooks/task6_round5.ipynb`).

## How to evaluate
- `notebooks/task6_improve.ipynb`: E1 shipped → E2 features → E3 steps → E4 bagging →
  E5 best on both backtests → E6 coverage and permutation importance.
- Ship if the best variant wins on **both** backtests and gains ≥ 2% on A. Then `make check`,
  `make changes`, and upload.

## Results log
| Variant | Backtest A | Backtest B | Public | Note |
|---|---|---|---|---|
| shipped | 7,182,796 | 7,191,137 | 8.76M | |
| + item_gaps | 7,185,263 (+0.03%) | | | no signal; dropped |
| + coread | 7,115,770 (−0.93%) | | | co_q25 7th in permutation importance; delay defined on 39% of rows |
| + both | 7,115,651 (−0.93%) | | | = coread alone |
| both, n_steps 2 / 3 | 7,127,154 (−0.77%) / 7,145,328 (−0.52%) | | | more targets hurt; keep n_steps = 1 |
| both, n_steps 2, n_bag 5 | 7,111,673 (−0.99%) | 7,095,231 (−1.33%) | | bagging ≈ −0.2% |
| round 2: coread (n_steps 1) | 7,115,770 (−0.93%) | | | `notebooks/task6_coread.ipynb` |
| coread, bag 5 | 7,142,051 (−0.57%) | | | worse than one seed: seed noise ≈ 0.3% |
| coread k = 3 / 5 / 10 | −0.89% / −1.09% / −1.02% | | | coverage of the delay 57% / 63% / 67% (k=1: 39%); within noise |
| **shipped: coread k=5, bag 5** | **7,110,402 (−1.01%)** | **7,096,961 (−1.31%)** | **8,637,709 (−1.4%)** | wins on both; bagging kept to reduce seed variance |
| round 3: tuning (3-seed bags) | 7,116,635 → 7,095,367 with 127 leaves (−0.30%) | not run (BEST unset) | | noise ≈ 0.04%; other settings within ±0.1%; ×1.05 scale −0.31% |
| round 4: best scale by leaves | 31: 1.06 · 63: 1.07 · 127: 1.07 · 255: 1.08 | 31: 1.035 · 63–255: 1.045–1.05 | | best scale grows B → A, as gaps do |
| **shipped: + 127 leaves, × 1.07** | **−0.73% (7,064,960 at ×1.05)** | **−0.48%** | **8,542,010 (−1.1%)** | rank 107 → 76; 8.5M needs another −0.5% |

**Outcome (round 2):** the co-read delay is a real but small gain (~1% on both backtests, every variant),
short of the ~3% for 8.5M. `item_gaps` and `n_steps` were removed from the code. Public 8.76M → **8.64M**
(−1.4%, a bit more than validation). Still tier 3.5; 8.5M needs another −1.6%.
