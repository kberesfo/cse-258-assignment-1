# Task 2 plan: widen the 4.0 margin

**Now:** public **0.9410** (4.0 cutoff 0.94, margin +0.001), validation 0.9383.
**Why new signal is needed:** the remaining tweaks in `plan.md` each give < 0.001, and small
validation gains have shrunk by about two-thirds on public. Aim for **≥ +0.003 on validation**
before uploading.

## Facts behind the plan (checked 2026-11-10)
- The test is a random ~1% holdout of interactions (20,000 rows). A user's test-row count
  correlates 0.84 with their history length. This is the same design as
  `make_pair_validation` (random holdout), so validation should keep predicting public.
- Users average 1.94 test rows each (max 60). Each row holds one real item, so the user's other
  test rows are evidence of what they read. No feature uses this yet.
- Validation holds out 10% of interactions, but test features come from models refit on all of
  train. Validation users have ~10× more held-out rows, and history-sum features (`cos_sum`)
  sit on a different scale.

## Steps (priority order)

### P1. Similarity to the user's other test candidates: no gain, dropped
- [x] `models/ranking/context.py`: `ctx_sum / ctx_max / ctx_mean` = binary cosine of each
      candidate to both candidates of the user's other rows (pairwise difference).
- [x] Density matching: in validation, each other row is kept with probability
      `keep = (test rows / train rows) / (val rows / tr rows)` ≈ 0.1. The test uses keep = 1.
- [x] Wired into `task2_pair_features(ctx_keep=...)` behind `TASK2_TEST_CONTEXT` (off for now).
- [x] Ran `notebooks/task2_context.ipynb`. Density matching worked: 66% of validation rows
      have context vs 69% of test rows; mean other rows 2.76 vs 3.00.
- [x] **Result: no gain.** Matched context +0.0001 (seed 0); seeds 1–3: +0.0001 / −0.0001 /
      −0.0001. Even unthinned (~28 other rows) gives only +0.0002 with LR and +0.0007 with GB.
      Rows with and without context gain the same, i.e. nothing.
- **Why:** the user's other real items are statistically just more history items, and the
  history already holds ~60 of them, which `cos_sum / max / mean` already use. A few extra noisy
  items add no new information. Code removed from the package (not in the submission);
  `notebooks/task2_context.ipynb` keeps the saved outputs.

### P2. Fix the scale mismatch between validation and test features
- Use a smaller holdout (`val_frac` ≈ 0.02) over several seeds, and fit the combiner on all the
  folds' pairs combined.
- Cheap alternative: normalise history-sum features (e.g. `cos_sum / log(1 + |history|)`).

### P3. Stronger base models as extra features (one column each)
- RP3beta (random-walk item-to-item score with a popularity penalty): cheap, often beats cosine.
- EASE (closed-form item-to-item) on the top ~15–20k items. All 51k would need a ~10 GB dense
  matrix.
- ALS from `implicit`.
- BPR: `learning_rate` and `iterations` never swept; try averaging 3 seeds.

### P4. Better features from the user's own history
- Similarity to the last N history items, or weighted by recency.
- Mean of the top-k similarities (`ItemSimilarity(top_k=...)` already supports it).
- User-to-user kNN: how many of the user's nearest neighbours read the candidate.
- Similarity to items the user rated highly versus poorly.

### P5. Combiner (low priority)
- Once new features are in, retry gradient boosting with history length and popularity as
  interaction features. The errors are concentrated in low-confidence pairs.

## How to evaluate
- Use the held-out half of validation pairs, on ≥ 3 seeds. Keep a change only if it wins on
  every seed.
- Do the experiments in notebooks. Move only the winners into `task2_pair_features`, then
  `make check` → `make changes` → upload.

## Results log
| Step | Validation | Public | Note |
|---|---|---|---|
| shipped | 0.9383 | 0.9410 | BPR K=100 + K=10, cosine, time features, rules |
| P1 test context | 0.9384 (+0.0001; seeds 1–3 ±0.0001) | not uploaded | redundant with history cosine |
