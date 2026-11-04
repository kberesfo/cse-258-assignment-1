# Assignment 1 plan: reach ~22 / 33 (comprehensive exam bar)

**Goal:** 65–70% on Assignment 1 ≈ **22 / 33**.
**Deadline:** start of week 8 (exact time on Gradescope).
**Rule:** Homework 3 is already passed (6/6 thresholds). Do not resubmit or change it. Everything below is for Assignment 1 only.
**Approach:** Phase 1 (squeeze basic models) is done. Phase 2 (feature-based pairwise combiners) has moved Tasks 2, 4 and 5 up a tier each. Remaining work: protect thin margins, push for the next tier on best-3 tasks, then the writeup.

---

## 1. How it's graded (as understood)

| Component | Marks | Source |
|---|---|---|
| Best 3 tasks × private leaderboard tiers | 3 × 4 = 12 | professor's post (1 of the 4 for any valid upload) |
| Best 3 tasks × public leaderboard tiers (private tiers ÷ 2) | 3 × 2 = 6 | professor's post ("likely similar but divided by 2") |
| Best 3 tasks × ranking | 3 × 4 = 12 | original spec; needed for the total to reach 33 (3 × (4 + 2 + 4) + 3) |
| Writeup | 3 | spec ("generally get full marks automatically") |

- [ ] **Ask the professor:** how are the 4 ranking marks per task computed (percentile bands, rank cutoffs)?
- [ ] **Ask the professor:** are features computed from test-file *inputs* (never labels) allowed, e.g. how often an item appears as a candidate, or similarity to the same user's other test candidates?

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

## 2. Where I stand (public scores)

**Overall leaderboard rank: 125 / ~215.**

| Task | Score | Tier | Task rank | Next cutoff | Distance | Top on board | History |
|---|---|---|---|---|---|---|---|
| 1 | 0.5917 | 2.5 | 125 | 0.58 | −0.012 | 0.5048 | 0.6185 → 0.5917 (drop 0s) |
| 2 | **0.9302** | **3.5** (+0.0002 margin) | **97** | 0.94 | +0.010 | 0.9591 | 0.9066 (BPR) → 0.9302 (LR combiner + cosine) |
| 3 | 0.7626 | 2.0 | 130 | 0.75 | −0.013 | 0.6644 | 0.7921 → 0.7626 (drop 0s) |
| 4 | **0.8984** | **3.0** | **122** | 0.92 | +0.022 | 0.9556 | 0.8706 (BPR + rule) → 0.8984 (GB combiner) |
| 5 | **0.8905** | **3.0** | **102** | 0.91 | +0.020 | 0.9349 | 0.8668 → 0.8703 (rules) → 0.8905 (GB combiner) |
| 6 | 11.21M | 2.0 | 120 | 11M | −0.21M | 7.99M | 13.01M → 11.39M (bound) → 11.21M (offset, λ=2) |

**Ranking-marks guess (unconfirmed):** if marks are linear in rank, 4 × (1 − rank / 215): Task 2 ≈ 2.2, Task 5 ≈ 2.1, Task 4 ≈ 1.7 → **≈ 6.0 / 12**. Ranks drift down as others upload.

**Best 3 now:** 3.5 + 3.0 + 3.0 (Tasks 2, 4, 5) → private 9.5 + public 4.75 = **14.25 / 18**, writeup **0 / 3**, ranking **? / 12**.
**If Task 2 slips to 3.0 on private:** 3.0 + 3.0 + 3.0 → 13.5 / 18.
**Each tier step on a best-3 task = +0.75** (+0.5 private, +0.25 public).
**Margin rule:** the hidden half decides. Aim about 0.005–0.01 past a cutoff, not right at it. Task 2's 3.5 is at +0.0002, so it is effectively a coin flip on private.

**Validation → public calibration** (use when predicting tiers):
| Task | Validation | Public | Note |
|---|---|---|---|
| 1 | 0.594 | 0.592 | agrees |
| 2 | 0.9371 (held-out half) | 0.9302 | public ~0.007 lower |
| 3 | 0.745 | 0.763 | public ~0.018 worse |
| 4 | 0.9041 (held-out half) | 0.8984 | public ~0.006 lower |
| 5 | 0.8932 (held-out half; + unseen rule on ~4% of test rows) | 0.8905 | public ~0.003 lower |
| 6 | 8.43M | 11.21M | public ≈ 1.33 × validation |

---

## 3. Next steps (in order)

### N1. Task 2: secure 3.5 (aim public ≥ 0.935, val ≥ 0.942)
Diagnostics (`notebooks/task2_diagnostics.ipynb`): no cold-start problem (72% of users have 50+ books; accuracy flat by history length), popularity carries no signal, model is well calibrated. Over 80% of errors are low-confidence pairs where every existing feature alone is at chance, so a fallback rule cannot help. New signals are needed.
- [ ] **Top-k cosine:** mean of the k highest similarities to the history (k = 5, 10). Targets diluted sums for heavy users.
- [ ] **Rating-weighted cosine:** weight history items by rating (or rating − user mean); 0-ratings as plain interactions.
- [ ] **Recency-weighted cosine:** only if `tasks_1_2_train.csv` has timestamps.
- [ ] Each is a reweighting/filter of the history matrix `H` before `H.multiply(S[i])`; test against 0.9371 on the held-out half.

### N2. Tasks 4 & 5: add cosine features (aim 3.5: 0.92 / 0.91 public)
Cosine gave Task 2 +0.023 and does not depend on time, so it is a new signal for both combiners.
- [ ] Add `cos_sum`, `cos_max`, `cos_mean` (and N1's winners) to `PairContext` behind a flag, as with `transitions`.
- [ ] Task 4 needs about +0.022 public; Task 5 about +0.020. Check on each held-out half first (current: 0.9041 / 0.8932).
- [ ] Re-run the Task 4 `np.allclose` check after any `pairwise.py` change.

### N3. Writeup (+3); the approaches are now stable enough to start
- [ ] `writeup.md`: one short paragraph per task, condensed from the docstrings.
- [ ] Explain the pairwise combiner as the course's pairwise logistic regression (σ(θ·[φ(u,i) − φ(u,j)])), with GB as a swap-in for the linear combiner on Tasks 4/5 (captures feature interactions).
- [ ] Mention Tasks 1/3 0-rating handling and that the HW3 submission kept them.
- [ ] Block an evening before the deadline.

### N4. Insurance: a fourth strong task (only after N1–N3)
- [ ] **Task 1 → 3.0 (≤ 0.58) or Task 3 → 2.5 (≤ 0.75):** bias model + residual regressor; key feature is item-kNN, Σ cos(i,j)(r_uj − b̂_uj) / Σ cos(i,j), reusing `fit_item_cosine`. Task 3 also has the time-bias model on branch `task3-time-bins` (val 0.7406).
- [ ] **Task 6 → 2.5 (≤ 11M):** `HistGradientBoostingRegressor(loss="absolute_error")` on user shrunk median, n_gaps, bound, item popularity, item age at t_last.

### N5. Optional: test-file structure
- [ ] Only if the professor allows test-input features. Check rows per user in test vs validation first; if test users have several rows each, similarity to the user's other test candidates is a likely strong signal.

---

## 4. Done ✔
- [x] Tasks 1 & 3: drop 0-ratings (train + validate on nonzero rows); bias re-tuned; SVD and blend checked (no gain). Public 0.5917 / 0.7626.
- [x] Task 6: item first-appearance lower bound (val 10.14M → 8.53M), +35-day median-delay offset, λ re-tuned to 2 → val 8.43M, public 11.21M. Upper bound tried: hurt (corrupt future timestamps), not used.
- [x] Task 5 Phase 1: recency (last m books) gave no gain; trend rule shipped (0.8703).
- [x] Task 3 Phase 2: time-binned item biases val 0.7406 vs 0.7449; year one-hot and user-day corrections small. Not shipped (no tier change).
- [x] Task 4: GB pairwise combiner (`models/ranking/pairwise.py`): BPR, not-yet-seen, item age, log popularity, trend share at 20/50/100 bins. Val 0.8803 → 0.9041; public 0.8706 → 0.8984.
- [x] Task 5: same combiner with query time = user's `t_last`, plus transition counts; unseen-item rule kept last (validation drops those rows). Val 0.8728 → 0.8932; public 0.8703 → 0.8905.
- [x] Task 2: LR pairwise combiner on BPR, log popularity, item-item cosine (sum/max/mean over history; `models/ranking/similarity.py`). Val 0.9146 → 0.9371; public 0.9066 → 0.9302.
- [x] Diagnostics: low-confidence fallback rule tested on Task 2, no gain (every feature alone ≈ chance on those rows).

## 5. Checklist before every upload
- [ ] `make check` (runs predict + bundle): all six prediction files report "matches".
- [ ] Spot-check `head -3 dist/task_N_predictions.csv` for each changed task.
- [ ] For tasks with rules after a combiner: confirm the rule rows (e.g. Task 5 unseen candidates) predict as expected.
- [ ] Record the new public scores in section 2; commit and tag the state.

## 6. Path to 22
Now: 14.25 / 18 on tiers (13.5 if Task 2 slips on private); task ranks 97 / 122 / 102 on the best 3.
- With the writeup: **17.25 / 21**, needing about **4.75 of 12** ranking marks.
- Under the linear-rank guess (≈ 6.0), that is **≈ 23 / 33**: just past the bar, with little room for a private-split slip.
- N1 secures Task 2 at 3.5. N2 lifting Task 4 or 5 to 3.5 adds +0.75 each → up to 15.75 / 18 (18.75 with the writeup), needing ~3.25 from ranking.
- Ranking marks are the remaining unknown; the professor's answer decides how much N2 and N4 are worth.