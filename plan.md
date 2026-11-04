# Assignment 1 plan: reach ~22 / 33 (comprehensive exam bar)

**Goal:** 65–70% on Assignment 1 ≈ **22 / 33**.
**Deadline:** start of week 8 (exact time on Gradescope).
**Rule:** Homework 3 is already passed (6/6 thresholds). Do not resubmit or change it. Everything below is for Assignment 1 only.
**Approach:** Phase 1 (squeeze basic models) and Phase 2 (pairwise combiners + test-design rules) are done; Tasks 2, 4 and 5 are now the best 3. Remaining work: protect thin margins, find the remaining test-design tells (Task 4), push Task 4 to 3.5, then the writeup.

---

## 1. How it's graded (as understood)

| Component | Marks | Source |
|---|---|---|
| Best 3 tasks × private leaderboard tiers | 3 × 4 = 12 | professor's post (1 of the 4 for any valid upload) |
| Best 3 tasks × public leaderboard tiers (private tiers ÷ 2) | 3 × 2 = 6 | professor's post ("likely similar but divided by 2") |
| Best 3 tasks × ranking | 3 × 4 = 12 | original spec; needed for the total to reach 33 (3 × (4 + 2 + 4) + 3) |
| Writeup | 3 | spec ("generally get full marks automatically") |

Confirmed on Piazza: Tasks 1/3 MSE, Tasks 2/4/5 accuracy, Task 6 MAE; the visible leaderboard is **not** the hidden test; 0-ratings are skipped in the leaderboard MSE (@117).

- [ ] **Ask the professor:** how are the 4 ranking marks per task computed (percentile bands, rank cutoffs)?
- [ ] **Ask the professor:** are features computed from test-file *inputs* (never labels) allowed, e.g. how often an item appears as a candidate, or similarity to the same user's other test candidates?
- [ ] **Ask a TA:** many Task 1 scores cluster at ~0.52 (and Task 3 at ~0.67–0.72), including from students near chance on other tasks. What approach with course tools gets there?

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

The leaderboard is sorted by Task 1, so the overall "rank" shown is the Task 1 position (129). Use per-task ranks.

| Task | Score | Tier | Task rank | Next cutoff | Distance | Top on board | History |
|---|---|---|---|---|---|---|---|
| 1 | 0.5917 | 2.5 | 129 | 0.58 | −0.012 | 0.5048 | 0.6185 → 0.5917 (drop 0s) |
| 2 | **0.9407** | **4.0** (+0.0007 margin) | ? (update) | — (top tier) | — | 0.9591 | 0.9066 (BPR) → 0.9302 (LR combiner + cosine) → 0.9407 (+ seen-by-user rule) |
| 3 | 0.7626 | 2.0 | 130 | 0.75 | −0.013 | 0.6644 | 0.7921 → 0.7626 (drop 0s) |
| 4 | **0.9172** | **3.0** | ? (update) | 0.92 | **+0.0028** | 0.9556 | 0.8706 (BPR + rule) → 0.8984 (GB combiner) → 0.9172 (+ seen-by-user rule) |
| 5 | **0.8945** | **3.0** | ? (update) | 0.91 | +0.016 | 0.9349 | 0.8668 → 0.8703 (rules) → 0.8905 (GB combiner) → 0.8945 (+ seen-by-user rule) |
| 6 | 11.21M | 2.0 | 120 | 11M | −0.21M | 7.99M | 13.01M → 11.39M (bound) → 11.21M (offset, λ=2) |

**Best 3 now:** 4.0 + 3.0 + 3.0 (Tasks 2, 4, 5) → private 10 + public 5 = **15 / 18**, writeup **0 / 3**, ranking **? / 12**.
**If Task 2 slips to 3.5 on private:** 3.5 + 3.0 + 3.0 → 14.25 / 18.
**Each tier step on a best-3 task = +0.75** (+0.5 private, +0.25 public).
**Margin rule:** the hidden half decides. Aim about 0.005–0.01 past a cutoff. Task 2's 4.0 is at +0.0007 (coin flip on private); Task 4 is 0.0028 short of 3.5.

**Leaderboard context:** many students sit at Task 4 ≈ 0.94–0.95 and Task 2 ≈ 0.95, so more test-design tells likely remain (see N1). A large cluster at Task 1 ≈ 0.52, including students near chance on Task 5, suggests a simple Task 1 approach not yet found (cross-file data ruled out: 0% user/item overlap between files).

**Validation → public calibration** (use when predicting tiers):
| Task | Validation | Public | Note |
|---|---|---|---|
| 1 | 0.5941 | 0.5917 | agrees |
| 2 | 0.9371 (held-out half) | 0.9302 | public ~0.007 lower (before rule) |
| 3 | 0.745 | 0.763 | public ~0.018 worse |
| 4 | 0.9041 (held-out half) | 0.8984 | public ~0.006 lower (before rule) |
| 5 | 0.8932 (held-out half) | 0.8905 | public ~0.003 lower (before rule) |
| 6 | 8.43M | 11.21M | public ≈ 1.33 × validation |

**Rule predictions (`notebooks/rules.ipynb`) vs leaderboard:** Task 4 +1.97 predicted / +1.88 observed; Task 2 ~0.9406 / 0.9407; Task 5 ~0.8948 / 0.8945. The notebook's coverage × disagreement estimate is reliable.

---

## 3. Next steps (in order)

### N1. Task 4: find the remaining test-design tells (TODO; the 0.94+ cluster)
Task 4's extra information is the row timestamp, so the likely tells involve time. Check each in `rules.ipynb`; any that covers more than a few percent of rows is worth a rule or feature.
- [ ] **Same-second batches:** share of test rows whose (user, t) matches an exact training timestamp for that user. If large, similarity to the other books in that batch should identify the real item.
- [ ] **Candidate activity near t:** exactly one candidate active (first ≤ t + w and last ≥ t − w) within ±7 days; then refine to the number of interactions within ±w of t.
- [ ] **Rows per user in the test:** if users have several rows, a per-user test signal exists (needs the professor's OK; see N6).
- [ ] **Error analysis on validation** (as in `task2_diagnostics.ipynb`): where do the combiner's remaining errors sit?
- [ ] Same checks for Task 2 / Task 5 where they apply (Task 2 has train timestamps but no test timestamps).

### N2. Tasks 4 & 5: add cosine features (Task 4 needs +0.003 for 3.5)
Cosine gave Task 2 +0.023 and does not depend on time, so it is a new signal for both combiners.
- [ ] Add `cos_sum`, `cos_max`, `cos_mean` to `PairContext` behind a flag, as with `transitions`.
- [ ] Task 4: also **time-local cosine** (similarity to the user's history items near t), since Task 4 knows when the interaction happened.
- [ ] Check on each held-out half first (current: 0.9041 / 0.8932). Re-run the Task 4 `np.allclose` check after any `pairwise.py` change.
- [ ] Optional: build validation like the test (decoys may come from the user's history) so the combiner learns the seen-by-user case and validation measures it.

### N3. Task 2: protect the 4.0 margin (+0.0007)
Diagnostics (`notebooks/task2_diagnostics.ipynb`): no cold-start problem, popularity carries no signal, model well calibrated; >80% of errors are low-confidence pairs where every feature alone is at chance, so new signals are needed.
- [ ] **Top-k cosine:** mean of the k highest similarities to the history (k = 5, 10).
- [ ] **Rating-weighted cosine:** weight history items by rating (or rating − user mean); 0-ratings as plain interactions.
- [ ] **Recency-weighted cosine:** train has timestamps; weight history items by recency within the user's history.

### N4. Writeup (+3); the approaches are stable enough to start
- [ ] `writeup.md`: one short paragraph per task, condensed from the sectioned docstrings (Test design / Model / Rules / Validation).
- [ ] Explain the pairwise combiner as the course's pairwise logistic regression (σ(θ·[φ(u,i) − φ(u,j)])), with GB as a swap-in on Tasks 4/5 (feature interactions; +0.0023 over LR on Task 4).
- [ ] Explain the test-design rules and how they were found (Piazza post + `rules.ipynb`; predicted vs observed gains).
- [ ] Mention Tasks 1/3 0-rating handling and that the HW3 submission kept them.
- [ ] Block an evening before the deadline.

### N5. Insurance: a fourth strong task (only after N1–N4)
- [ ] **Task 1:** E4 (factors frozen vs free) pending. E5 error breakdown (by user size, item popularity, user std, predicted and true rating) + isotonic calibration test. Then implicit feedback (zeros as interactions: residual features, SVD++ if they help). Note: Task 1 test has no timestamps, so time features are out.
- [ ] **Task 3 → 2.5 (≤ 0.75):** time-bias model on branch `task3-time-bins` (val 0.7406); item-kNN residual Σ cos(i,j)(r_uj − b̂_uj) / Σ cos(i,j); same-session neighbour rating (P(same rating) 0.55 within an hour vs 0.43 after a month).
- [ ] **Task 6 → 2.5 (≤ 11M):** `HistGradientBoostingRegressor(loss="absolute_error")` on user shrunk median, n_gaps, bound, item popularity, item age at t_last.

### N6. Optional: test-file structure
- [ ] Only if the professor allows test-input features. Do not cross-reference one task's test file to answer another's (Tasks 1 and 2 share IDs) without explicit approval.

---

## 4. Done ✔
- [x] Tasks 1 & 3: drop 0-ratings (train + validate on nonzero rows); bias re-tuned; SVD and blend checked (no gain). Public 0.5917 / 0.7626.
- [x] Task 6: item first-appearance lower bound (val 10.14M → 8.53M), +35-day median-delay offset, λ re-tuned to 2 → val 8.43M, public 11.21M. Upper bound tried: hurt (corrupt future timestamps), not used.
- [x] Task 5 Phase 1: recency (last m books) gave no gain; trend rule shipped (0.8703).
- [x] Task 3 Phase 2: time-binned item biases val 0.7406 vs 0.7449; year one-hot and user-day corrections small. Not shipped (no tier change).
- [x] Task 4: GB pairwise combiner (`models/ranking/pairwise.py`): BPR, not-yet-seen, item age, log popularity, trend share at 20/50/100 bins. Val 0.8803 → 0.9041; public 0.8706 → 0.8984.
- [x] Task 5: same combiner with query time = user's `t_last`, plus transition counts; unseen-item rule kept last. Val 0.8728 → 0.8932; public 0.8703 → 0.8905.
- [x] Task 2: LR pairwise combiner on BPR, log popularity, item-item cosine (`models/ranking/similarity.py`). Val 0.9146 → 0.9371; public 0.9066 → 0.9302.
- [x] **Seen-by-user rule** (Tasks 2/4/5): if exactly one candidate is in the user's training history, predict the other (test decoys don't exclude history; positives are held out). Found via a Piazza post; coverage 1.92% / 5.07% / 0.69%. Public: Task 2 0.9302 → 0.9407, Task 4 0.8984 → 0.9172, Task 5 0.8905 → 0.8945.
- [x] Unseen-item rule kept on Tasks 2/4 as a guard (fires on 0% of their tests).
- [x] Diagnostics: low-confidence fallback rule on Task 2, no gain. Task 1 data checks: no test timestamps, 0% test pairs in train, 1.48% zero rows, 1.3% single-rating users, cross-file overlap 0%.
- [x] Task 1 LFM: bias λ confirmed optimal (inside grid); SVD with one λ best 0.5989; separate bias/factor λ best 0.5980; all worse than bias 0.5941.
- [x] Notebooks cleaned (Task 1, Task 2): header with shipped model and scores, one validation split, experiments E1… with results in markdown; `rules.ipynb` and `task2_diagnostics.ipynb` added.
- [x] Docstrings in sectioned format (Test design / Model / Rules / Validation) for Tasks 2, 4, 5.

## 5. Checklist before every upload
- [ ] Copy `dist/` to `dist_prev/` first, so changed predictions can be counted.
- [ ] `make check` (runs predict + bundle): all six prediction files report "matches".
- [ ] Re-run `rules.ipynb`: every applied rule shows 0.00% disagreement with `dist/`.
- [ ] Record the new public scores and per-task ranks in section 2; commit and tag the state.

## 6. Path to 22
Now: **15 / 18** on tiers (14.25 if Task 2 slips to 3.5 on private).
- With the writeup: **18 / 21**, needing about **4 of 12** ranking marks.
- Task 4 → 3.5 (+0.0028 public) adds +0.75 → 15.75 / 18 (18.75 with the writeup), needing ~3.25 from ranking.
- Ranking marks are the remaining unknown; the professor's answer decides how much N1–N3 are worth beyond tier safety.