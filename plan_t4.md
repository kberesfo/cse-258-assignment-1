# Task 4 plan: feature engineering toward the 4.0 tier (acc ≥ 0.95)

**Now:** public **0.9277** (tier 3.5; 4.0 needs 0.95, +0.022). Validation (GB on held-out half)
**0.9173**.
**Model:** GB pairwise combiner on `PairFeatures` (BPR, not-yet-seen, age, popularity, trend
20/50/100, cosine sum/max/mean) + seen-by-user rule.
**Calibration:** additive validation gains carried over to public almost 1:1 so far
(+0.013 val → +0.011 public).

## Why there is room
Task 4 holds out **random** interactions and gives their timestamp. The user's other reads on
both sides of t are in training, but the current features use the user's history as an
unordered set (BPR, cosine over all history) and look at time only from the item's side
(age, trend). Features that tie the candidate to *what the user was reading around t* use
signal the model does not see yet.

## Experiments (`notebooks/task4_features.ipynb`)
Each block is added on top of the baseline features as φ(item1) − φ(item2) and scored with the
same GB / held-out half as `task4.ipynb`, in both directions (train on A, test on B and the
reverse) to tell signal from noise. A block is kept if it gains ≥ +0.002 in both directions.

### F1. Calendar affinity (day of week, hour, month)
- log[ P(weekday(t) | item) / P(weekday(t)) ], smoothed toward the global distribution.
  Same for hour and month. Cheap; likely small (books aren't weekday-specific), but it is the
  requested DOW feature and costs nothing to test.

### F2. Time-local item similarity (expected largest gain)
- Take the user's 2n history interactions nearest in time to t (n before, n after).
- Cosine similarity of the candidate to those items: sum / max within ±1, 7, 30 days;
  mean / max of the k = 1, 3, 10 nearest in time; exp-decay weighted sum (τ = 7 days).
- Captures series and binge reading around t, which the global cosine features average away.

### F3. LightFM score (hybrid BPR / WARP)
- `LightFMModel` (now with `loss` and `user_features`). Variants:
  pure BPR (id features only), + item features (log-pop bucket, first-seen year, modal
  weekday), + user features (activity bucket, median read year), WARP loss.
- Small grid over components (32, 64) and epochs (20, 50) for the best variant.
- Expect overlap with implicit BPR; the item side features (year) may add a time prior.
  LightFM's Hogwild threading isn't deterministic; set `threads=1` before shipping if it
  goes in.

### F4. Combine and ablate
- All kept blocks together; drop-one ablation; then a light GB tuning pass
  (`max_iter`, `learning_rate`, `max_leaf_nodes`), because more features may need more trees.

## Shipping (after results)
- Move the kept features into `PairFeatures` as options (`calendar=`, `local_sim=`,
  `lightfm_params=`); defaults reproduce Tasks 2/5 unchanged.
- Turn them on in `task4_features()`; update the Task 4 docstring; `make check`,
  `make changes`, re-run `rules.ipynb`.

## Results (round 1)
| Block | A→B | B→A | Mean Δ | Keep? |
|---|---|---|---|---|
| baseline | 0.9173 | 0.9176 | — | |
| F1 weekday / hour / month | 0.9170 / 0.9170 / 0.9170 | 0.9176 / 0.9172 / 0.9177 | −0.0001 to −0.0003 | no |
| F2 local sim (n=10, τ=7) | 0.9218 | 0.9237 | **+0.0053** | **yes** |
| F2 n=5 / n=20 / τ=1 / τ=30 | 0.9218–0.9225 | 0.9233–0.9237 | +0.0052 to +0.0055 | (insensitive; keep defaults) |
| F3 LightFM best (WARP, 64, 30 ep) | 0.9180 | 0.9186 | +0.0009 | no (noise; not deterministic) |
| F3 BPR + item, grid | 0.9171–0.9176 | 0.9172–0.9183 | 0 to +0.0006 | no |
| F4 all | 0.9225 | 0.9233 | +0.0055 | — |
| F4 − local | 0.9172 | 0.9175 | −0.0001 | local is the whole gain |
| GB 300 it / lr 0.05 / 63 leaves | 0.9227 | 0.9244 | +0.0006 over default | no (below +0.002) |

**Shipped:** `LocalSimilarity` (`models/ranking/local.py`), turned on with
`PairFeatures(..., local_similarity=True)` in `task4_features()`. Expected public ≈ 0.9277 + 0.005
≈ 0.933 (still 3.5; more margin). The 4.0 tier (0.95) needs about +0.017 more.

## Round 2 ideas (local signal is the only thing that moved)
- Local similarity with other measures: Jaccard, BPR-factor cosine (dense, covers rare items).
- Direction: separate before-t / after-t summaries (series order).
- Candidate's reader timing: do other readers of the candidate read it close to the user's
  neighbouring items? (co-read *with time gap*, like Task 6's P3.)
