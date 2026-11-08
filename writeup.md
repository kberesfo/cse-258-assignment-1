CSE 258 Assignment 1 — Writeup
Code: assignment1.py (one section per task; run `python assignment1.py <data_dir> <out_dir>`)

Every task uses a stacked ensemble: standard course models (bias / time-bias, BPR, item-kNN,
transition counts, a shrunk median gap) feed a small learned combiner, either logistic
regression or gradient-boosted trees, which weighs their outputs together with user/item
features (activity, popularity, time). For ratings the combiner predicts the base model's
residual, so it only corrects the base. Validation sets are built the way the test sets were
(held-out interactions, popularity-matched decoys), so a validation gain predicts a leaderboard
gain; uniform decoys would make popularity look predictive when it isn't. Base models are fit
on a reduced training set and the combiner on held-out rows, so it sees base outputs as they
look on unseen data; the base models are then refit on all of train for the test. Where the
test's construction makes an answer certain (e.g. a decoy already in the user's history), a
rule overrides the combiner (Tasks 2, 4 and 5).

**Task 1 (rating prediction).** The base model is a bias-only latent factor model,
α + β_u + β_i (ALS); adding latent factors γ_u·γ_i never helped, since the data is too sparse
to fit them (best SVD 0.597 vs 0.594 on validation). The bias model's error was concentrated
in 1–2 star ratings (7% of rows, 36% of error), i.e. user-specific dislikes, so I added three
item-kNN features that read how the user rated similar books (Jaccard and adjusted-cosine
similarity on the bias model's residuals, plus a similarity-weighted average rating), and a
gradient-boosted regressor learns the correction to the bias model from them and user/item
statistics. MSE on an outer holdout: 0.5876 → 0.5454; public 0.5917 → 0.5443.

**Task 2 (interaction prediction).** A pairwise logistic regression (the course's pairwise
model) combines the differences between the two candidates' features: two BPR models
(K = 100 and K = 10), popularity, cosine similarity to the user's history, and time features
relative to the user's last interaction. Decoys are popularity-matched, so popularity alone is
useless; what separates the real item from a decoy is how similar it is to the books the user
already read (the closest such book carries most of the weight). Validation: BPR alone 0.9146
→ combiner 0.9383; public 0.9066 → 0.9410 (including the seen-by-user rule, Section 3).

**Task 3 (rating prediction with timestamps).** Task 1's stack, with a time-aware base model
α + β_u + β_i + β_{i,bin(t)} and an extra feature for the user's mean residual on the same
day / week. Item-kNN carried most of the gain; time added little (the time-bias base alone
improved MSE by 0.6%), as the professor expected for this data. MSE on an outer holdout:
0.7369 → 0.6908 (−6%); public 0.7626 → 0.7084.

**Task 4 (interaction prediction with timestamps).** Task 4 used a gradient-boosted pairwise combiner that takes
the BPR score plus time features at the row's timestamp t (whether the book had appeared yet,
its age, and its share of interactions in t's time bin at 20/50/100 bins) and cosine
similarity to the user's history. Decoys ignore time, so a candidate that first appears after
t is almost always the decoy (97% on about a quarter of rows); boosted trees learn how much to
trust that signal relative to the others, which beat logistic regression (0.9041 vs 0.9018).
Validation: BPR + a hard not-yet-seen rule 0.8803 → 0.9173; public 0.8706 → 0.9277
(including the seen-by-user rule, Section 3).


**Task 5 (next-item prediction).** Task 4's combiner, with two changes: test rows have no
timestamp, so the query time is the user's last training interaction (the next book follows
it), and a first-order transition count from the user's last book is added, along with cosine
similarity to the history. The real next book is more often a recent release than a decoy is,
and sometimes follows the last book directly (e.g. a series); similarity to the user's most
recent books added nothing beyond whole-history similarity, consistent with books being only
weakly sequential. Validation: BPR + hard rules 0.8728 → combiner 0.9029; public 0.8703 →
0.9030 (including the unseen-item and seen-by-user rules, Section 3).


**Task 6 (time-delta prediction, MAE).** Following the lecture's hint, this is treated as
regression on the user's previous gaps: a gradient-boosted regressor with MAE loss on log gaps,
using the last five gaps, gap statistics, and a shrunk per-user median gap as inputs, trained
by stepping back one interaction per user. MAE is minimized by the median, and the median is
unchanged by the log transform, so the model still predicts a median gap while the log scale
tames the heavily skewed gaps (median 13 days, mean 71); the next book also can't be read
before it first appears, so predictions are raised to that date plus 35 days (the median delay
after a book's first appearance). Validation MAE: per-user median 12.05M → 8.01M; public
13.01M (first submission) → 10.65M.