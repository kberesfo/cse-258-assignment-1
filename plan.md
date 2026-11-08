# Assignment 1 plan: reach ~22 / 33 (comprehensive exam bar)

**Goal:** 65–70% on Assignment 1 ≈ **22 / 33** (professor: a 4.0, a 3.5 and a 3.0 should pass).
**Deadline:** start of week 8 (exact time on Gradescope).
**Rule:** Homework 3 is already passed (6/6 thresholds). Do not resubmit or change it. Everything below is for Assignment 1 only.
**Status:** **model work is done.** Best 3 = Task 2 (4.0) + two of Tasks 1 / 3 / 4 (all 3.5), with Task 5 (3.0) and Task 6 (2.5) as further spares. Remaining: the writeup.

---

## 1. How it's graded (as understood)

| Component | Marks | Source |
|---|---|---|
| Best 3 tasks × private leaderboard tiers | 3 × 4 = 12 | professor's post (1 of the 4 for any valid upload) |
| Best 3 tasks × public leaderboard tiers (private tiers ÷ 2) | 3 × 2 = 6 | professor's post ("likely similar but divided by 2") |
| Best 3 tasks × ranking | 3 × 4 = 12 | original spec; needed for the total to reach 33 (3 × (4 + 2 + 4) + 3) |
| Writeup | 3 | spec ("generally get full marks automatically") |

Confirmed on Piazza: Tasks 1/3 MSE, Tasks 2/4/5 accuracy, Task 6 MAE; the visible leaderboard is **not** the hidden test; 0-ratings are skipped in the leaderboard MSE (@117).
Confirmed in lecture: exploiting how the test was constructed is allowed ("find my mistakes and exploit them").

- [ ] **Ask the professor (optional):** how are the 4 ranking marks per task computed? Does "4.0 / 3.5 / 3.0" mean hidden-leaderboard tiers, and is it a total (10.5) or that exact pattern?
- [x] ~~Test-input features allowed?~~ Answered in lecture: exploiting test construction is fair game.

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

## 2. Where I stand (public scores, final models)

The leaderboard is sorted by Task 1, so the "rank" shown (90) is the Task 1 position.

| Task | Score | Tier | Margin | History |
|---|---|---|---|---|
| 1 | **0.5443** | **3.5** | 0.016 under 0.56 (4.0 needs 0.54) | 0.6185 → 0.5917 (drop 0s) → 0.5501 (stack: bias + cosine kNN + stats) → 0.5453 (Jaccard + adjusted-cosine kNN) → 0.5443 (+ raw-rating kNN) |
| 2 | **0.9410** | **4.0** | +0.001 over 0.94 (thin) | 0.9066 (BPR) → 0.9302 (LR combiner + cosine) → 0.9407 (+ seen-by-user rule) → 0.9410 (+ time features, BPR K=10) |
| 3 | **0.7084** | **3.5** | 0.0016 under 0.71 (thin) | 0.7921 → 0.7626 (drop 0s) → 0.7229 (stack: time-bias + kNN + stats + session) → 0.7084 (Task 1's kNN sources) |
| 4 | **0.9277** | **3.5** | +0.008 over 0.92 | 0.8706 (BPR + rule) → 0.8984 (GB combiner) → 0.9172 (+ seen-by-user) → 0.9277 (+ cosine features) |
| 5 | 0.9030 | 3.0 | 3.5 needs 0.91 (+0.007) | 0.8668 → 0.8703 (rules) → 0.8905 (GB combiner) → 0.8945 (+ seen-by-user) → 0.9030 (+ cosine) |
| 6 | **10.65M** | **2.5** | 0.35M under 11M | 13.01M → 11.39M (bound) → 11.21M (offset, λ=2) → 10.65M (`GapRegressor`) |

**Best 3:** 4.0 + 3.5 + 3.5 → private 11 + public 5.5 = **16.5 / 18**, writeup **0 / 3**, ranking **? / 12**.
**Slack:** three tasks at 3.5 (1, 3, 4) for two 3.5 slots. If Task 2 slips to 3.5 on private: 3.5 + 3.5 + 3.5 = 10.5 tier points, still at the professor's bar.
**Thin margins:** Task 2's 4.0 (+0.001) and Task 3's 3.5 (0.0016). Task 4 (+0.008) and Task 1 (0.016) are the solid 3.5s.

**Validation → public calibration** (how estimates held up):
| Task | Validation change | Public change | Note |
|---|---|---|---|
| 1 | outer holdout 0.5514 → 0.5465 → 0.5454 | 0.5501 → 0.5453 → 0.5443 | relative change carries over almost exactly |
| 2 | 0.9371 → 0.9383 (+0.001, four splits) | 0.9407 → 0.9410 (+0.0003) | small gains shrink on public |
| 3 | outer holdout 0.7019 → 0.6908 (−1.6%) | 0.7229 → 0.7084 (−2.0%) | relative gain carries over |
| 4 | 0.9041 → 0.9173 (+0.013) | 0.9172 → 0.9277 (+0.011) | additive gain carries over |
| 5 | 0.8932 → 0.9029 (+0.010) | 0.8945 → 0.9030 (+0.009) | additive gain carries over |
| 6 | ~8.43M → 8.01M (−5%) | 11.21M → 10.65M (−5%) | relative gain carries over; public ≈ 1.33 × validation |

**Rule predictions (`notebooks/rules.ipynb`) vs leaderboard:** Task 4 +1.97 predicted / +1.88 observed; Task 2 ~0.9406 / 0.9407; Task 5 ~0.8948 / 0.8945.

---

## 3. Lecture hints (professor, week 6) and how they played out

- **Top solutions ensemble standard models** (meta-model on user/item features + several models' predictions; "a small delta") → this is `StackedRating` and the pairwise combiners. Adding SVD (K = 1, 5) as members: no gain (redundant with kNN). Adding a raw-rating kNN: −0.2% (Task 1). BPR K = 10 as a member: +0.001 on Task 2, nothing on Task 5.
- **MF and BPR with small K** → MF never beat the bias model on Task 1, alone or as a stack member.
- **Weighted-average / Pearson models "probably not winners"** → on this data item-kNN was the largest Task 1/3 gain (−4 to −6%).
- **Temporal / sequence tasks may gain little** → confirmed: time-bias base −0.6% on Task 3; recent-history similarity and recency windows gave nothing on Task 5 ("books aren't very sequential"). FPMC not attempted.
- **Task 6: MAE loss, time-series regression on the previous deltas** → `GapRegressor`, −5% (2.0 → 2.5).
- **Exploit test-construction mistakes** → seen-by-user and unseen-item rules.

---

## 4. Next steps

### N1. Writeup (+3) — the only remaining required item
- [ ] `writeup.md`: one short paragraph per task, condensed from the sectioned docstrings (Test design / Model / Rules / Validation).
- [ ] Frame the shared approach: ensembles of standard models (bias, time-bias, BPR, item-kNN, MF tested) combined by a small meta-model on user/item features, as in lecture; pairwise logistic regression σ(θ·[φ(u,i) − φ(u,j)]) for ranking, GB where it captures interactions.
- [ ] Test-design rules and how they were found (Piazza post + `rules.ipynb`; predicted vs observed gains).
- [ ] Negative results worth one line each: MF ≈ no signal beyond biases; co-reading similarity variants saturate; sequential signal weak on Task 5.
- [ ] 0-rating handling for Tasks 1/3 (HW3 kept them).
- [ ] Update Task 1's docstring to the sectioned format (last one not yet done).
- [ ] `make submit` copies `writeup.md` → `dist/writeup.txt`.

### N2. Before the deadline
- [ ] Commit and tag the final state (`a1-final`); confirm the last upload matches `dist/`.
- [ ] Re-run `make check` once more on a clean checkout.

### Not pursuing (documented as future work)
- FPMC for Task 5 (factorized transitions; no library, uncertain gain on weakly sequential data).
- Task 1 → 4.0 (all directions plateau ~0.545 on the outer holdout; needs ~0.541).
- Further Task 2 margin (remaining ideas give < 0.001).

---

## 5. Done ✔
- [x] Tasks 1 & 3: drop 0-ratings (train + validate on nonzero rows).
- [x] Task 1 LFM: bias λ optimal; SVD best ~0.597 even with separate λ and full convergence; factors don't help alone.
- [x] Task 1 error analysis: 1–2 star ratings = 7% of rows, 36% of error → user-specific dislikes.
- [x] Task 1 stack: bias + item-kNN on residuals + `RatingStats`, GB on residuals. Similarity sweep: Jaccard + adjusted cosine best; GB tuning no gain; SVD members no gain; raw-rating kNN −0.2% (outer holdout 0.5454; public 0.5443).
- [x] Task 3 stack: time-bias base + Task 1's kNN sources + stats + session residual; outer holdout 0.7369 → 0.6908; public 0.7084. Test drawn within users' spans (94.7%).
- [x] Task 2: LR pairwise combiner on BPR, popularity, cosine sum/max/mean, time features relative to t_last, BPR K = 10 (val 0.9383, gain consistent over four splits). Jaccard, top-k cosine, span share: no gain.
- [x] Task 4: GB combiner + seen-by-user rule + cosine features (val 0.9041 → 0.9173).
- [x] Task 5: GB combiner (transitions, t_last query time) + rules + cosine (val 0.9029). Recent-history similarity (m = 1–10) and BPR K = 10: no gain.
- [x] Task 6: `GapRegressor` (GB, MAE on log1p gaps; lags, medians, GapModel prediction, bound) + bound + 35 days; val 8.01M; public 10.65M.
- [x] Seen-by-user rule (Tasks 2/4/5) and unseen-item rule (Task 5; guard on 2/4).
- [x] Class refactor: `RatingModel` / `FeatureSource` / `SurpriseModel` (`BiasModel`, `SVDModel`), `TimeBiasModel`, `ItemKNN` (residual / rating target), `RatingStats`, `SessionResidual`, `ModelPrediction`, `StackedRating`; `BPRModel`, `TransitionModel`, `PairFeatures`; `GapModel`, `GapRegressor`, `split_last`; shared `models/similarity.py`.
- [x] Makefile: automatic `dist_history/` snapshots; `make changes`.
- [x] Notebooks rebuilt on the classes (Tasks 1–6), plus `rules.ipynb`, `task2_diagnostics.ipynb`.
- [x] Sectioned docstrings for Tasks 2–6.

## 6. Checklist before every upload
- [ ] `make check`: all six prediction files report "matches".
- [ ] `make changes`: only the tasks you meant to change show changed rows.
- [ ] Re-run `rules.ipynb` if a ranking task changed.
- [ ] Record new public scores in section 2; commit and tag.

## 7. Path to 22
- Tiers: **16.5 / 18**. With the writeup: **19.5 / 21** → about **2.5 of 12** ranking marks needed.
- One half-tier slip on private: 15.75 / 18 → 18.75 with the writeup → ~3.25 from ranking.
- The writeup is the remaining item that moves the total.