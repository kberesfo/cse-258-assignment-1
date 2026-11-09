CSE 258 Assignment 1 — Writeup
Code: assignment1.py (one section per task; run `python assignment1.py <data_dir> <out_dir>`)

**1. Approach.** Every task uses a stacked ensemble: standard course models (bias / time-bias,
BPR, item- and user-kNN, transition counts, a shrunk median gap) feed a small learned combiner,
either logistic regression or gradient-boosted trees, which weighs their outputs together with
user/item features (activity, popularity, time). For ratings the combiner predicts the base
model's residual, so it only corrects the base. Validation sets are built the way the test sets
were (held-out interactions, popularity-matched decoys), so a validation gain predicts a
leaderboard gain; uniform decoys would make popularity look predictive when it isn't. The
combiner never sees base outputs on rows the base models were fit on: for ratings it is trained
out-of-fold over five folds (so it learns from every row), for rankings on a held-out split; the
base models are then refit on all of train for the test. Where the test's construction makes an
answer certain (e.g. a decoy already in the user's history), a rule overrides the combiner
(Tasks 2, 4 and 5; Section 3).

**2. Tasks.**

**Task 1 (rating prediction).** The base model is α + β_u + β_i (ALS); latent factors γ_u·γ_i
never helped, since the data is too sparse to fit them (best SVD 0.597 vs 0.594 on validation).
The bias model's error is concentrated in 1–2 star ratings (7% of rows, 36% of error), i.e.
user-specific dislikes, so the combiner's inputs describe how this user rated similar books and
how similar users rated this book: item-kNN on the bias residuals (Jaccard and adjusted cosine,
plus a similarity-weighted average rating), the same Jaccard kNN over only the user's k closest
books (a whole-history average dilutes the few close ones, e.g. a series), user-kNN over the
100 most similar readers, and user/item rating profiles (mean, spread, share of 1–2 and 5-star
ratings). Outer-holdout MSE: first stack 0.5430 → 0.5299 (−2.4%); the closest-books kNN gave
the largest single gain (−0.006). Public: bias only 0.5917 → first stack 0.5443 → 0.5296.

**Task 2 (interaction prediction).** A pairwise logistic regression (the course's pairwise
model) combines the differences between the two candidates' features: two BPR models
(K = 100 and K = 10), popularity, cosine similarity to the user's history, and time features
relative to the user's last interaction. Decoys are popularity-matched, so popularity alone is
useless; what separates the real item from a decoy is how similar it is to the books the user
already read (the closest such book carries most of the weight). Validation: BPR alone 0.9146
→ combiner 0.9383; public 0.9066 → 0.9410 (including the seen-by-user rule).

**Task 3 (rating prediction with timestamps).** Task 1's stack with a time-aware base,
α + β_u + β_i + β_{i,bin(t)}, and features for the user's residuals near the row's time: the
mean residual on the same day / week, on the user's nearest ratings before and after, and the
closest-books kNN weighted by time apart (14 / 90 / 365 days). Time in the base alone added
little (−0.6%), as the professor expected for this data; the user's recent residuals helped
more. Every source beat noise when removed from the full model. Outer-holdout MSE: first stack
0.6931 → 0.6760 (−0.016 on each of four splits); public: bias only 0.7626 → first stack 0.7084
→ 0.6881.

**Task 4 (interaction prediction with timestamps).** A gradient-boosted pairwise combiner takes
the BPR score plus time features at the row's timestamp t (whether the book had appeared yet,
its age, and its share of interactions in t's time bin at 20/50/100 bins) and cosine
similarity to the user's history. Decoys ignore time, so a candidate that first appears after
t is almost always the decoy (97% on about a quarter of rows); boosted trees learn how much to
trust that signal relative to the others, which beat logistic regression (0.9041 vs 0.9018).
Validation: BPR + a hard not-yet-seen rule 0.8803 → 0.9173; public 0.8706 → 0.9277
(including the seen-by-user rule).

**Task 5 (next-item prediction).** Task 4's combiner, with two changes: test rows have no
timestamp, so the query time is the user's last training interaction (the next book follows
it), and a first-order transition count from the user's last book is added, along with cosine
similarity to the history. The real next book is more often a recent release than a decoy is,
and sometimes follows the last book directly (e.g. a series); similarity to the user's most
recent books added nothing beyond whole-history similarity, consistent with books being only
weakly sequential. Validation: BPR + hard rules 0.8728 → combiner 0.9029; public 0.8703 →
0.9030 (including the unseen-item and seen-by-user rules).

**Task 6 (time-delta prediction, MAE).** Following the lecture's hint, this is treated as
regression on the user's previous gaps: a gradient-boosted regressor with MAE loss on log gaps,
trained by stepping back one interaction per user. Inputs are the user's last eight gaps and gap
statistics, a shrunk per-user median gap, and, since the test gives the next book, when other
readers read that book relative to the user's last interaction (it can't be read before it first
appears, and its later readers show when it was being read). MAE is minimized by the median, and
the median is unchanged by the log transform, so the model still predicts a median gap while the
log scale tames the skewed gaps (median 13 days, mean 71). Validation MAE: per-user median
12.05M → 7.18M; public 13.01M (first submission) → 8.76M.

**3. Rules from the test design.** Positives are held out of each user's training history, but
decoys were sampled without excluding it, so a candidate already in the user's history is the
decoy (1.9% of Task 2 rows, 5.1% of Task 4, 0.7% of Task 5); in Task 5, a candidate that never
appears in training is the real item (4.0% of rows). Both came from a Piazza discussion and were
checked in validation before use: on Task 4 the seen-by-user rule predicted +1.97 points and gave
+1.88 on the leaderboard. Task 6's reader-time features rely on the same kind of fact: the test
gives the next book.

**4. What didn't work.** Latent factors (MF / SVD) never beat the bias model on ratings and
added nothing as extra stack members; further similarity variants (Jaccard, top-k co-reading)
saturated on Task 2; similarity to the user's most recent books and recency windows added
nothing on Task 5; tuned boosting parameters alone moved MSE by under 0.001. Not tried: FPMC for
Task 5 and user drift (α_u·dev_u(t), Koren) for Task 3.