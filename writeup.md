CSE 258 Assignment 1 — Writeup
Code: assignment1.py (one section per task; run `python assignment1.py <data_dir> <out_dir>`)

**1. Approach.** Every task uses a stacked ensemble: standard course models (bias / time-bias,
BPR, item- and user-kNN, EASE, transition counts, a shrunk median gap) feed a small learned
combiner, gradient-boosted trees, which weighs their outputs together with user/item features
(activity, popularity, time). Logistic regression, the course's pairwise model, was the first
ranking combiner; trees replaced it because they can use interactions between signals. For
ratings the combiner predicts the base model's residual, so it only corrects the base.
Validation sets are built the way the test sets were (held-out interactions, popularity-matched
decoys), so a validation gain predicts a leaderboard gain; uniform decoys would make popularity
look predictive when it isn't. The combiner never sees base outputs on rows the base models were
fit on: for ratings it is trained out-of-fold over five folds (so it learns from every row), for
rankings on a held-out split; the base models are then refit on all of train for the test. Where
the test's construction makes an answer certain (e.g. a decoy already in the user's history), a
rule overrides the combiner (Tasks 2, 4 and 5; Section 3).

**2. Tasks.**

**Task 1 (rating prediction).** The base model is α + β_u + β_i (ALS); latent factors γ_u·γ_i
never helped, since the data is too sparse to fit them (best SVD 0.597 vs 0.594 on validation).
The bias model's error is concentrated in 1–2 star ratings (7% of rows, 36% of error), i.e.
user-specific dislikes, so the combiner's inputs describe how this user rated similar books and
how similar users rated this book: item-kNN on the bias residuals (Jaccard and adjusted cosine,
plus a similarity-weighted average rating), the same Jaccard kNN over only the user's k closest
books (a whole-history average dilutes the few close ones, e.g. a series), user-kNN over the 100
most similar readers, and user/item rating profiles (mean, spread, share of 1–2 and 5-star
ratings). Outer-holdout MSE: first stack 0.5430 → 0.5299 (−2.4%); the closest-books kNN gave the
largest single gain (−0.006). Public: bias only 0.5917 → first stack 0.5443 → 0.5296.

**Task 2 (interaction prediction).** A gradient-boosted pairwise combiner takes the differences
between the two candidates' features: two BPR models (K = 100 and K = 10), popularity, cosine
similarity to the user's history, time features relative to the user's last interaction, and two
features carried over from Task 4. The first is EASE (Steck 2019), a closed-form item-item
model, B = −P / diag(P) with P = (XᵀX + λI)⁻¹ (λ = 25, all 51k items, in float32 to fit in
memory), summed over the user's history. The second is similarity to what the user was reading
when the candidate was being read: rows have no timestamp, so the query time is the candidate's
median read time, and the feature is cosine similarity to the user's 20 reads nearest it. Decoys
are popularity-matched, so popularity alone is useless; what separates the real item from a
decoy is how similar it is to the books the user already read. Pairwise logistic regression (the
course's model) was the combiner until EASE was added: with it, the same features gained
+0.0015, against +0.0027 for boosted trees, which can use the EASE score in interaction with the
other features. Validation: BPR alone 0.9146 → LR combiner 0.9383 → +0.0027 with EASE, local
similarity and trees (mean of four splits, 0.9378 → 0.9404); public 0.9066 → 0.9410 → 0.9454
(including the seen-by-user rule).

**Task 3 (rating prediction with timestamps).** Task 1's stack with a time-aware base, α + β_u +
β_i + β_{i,bin(t)}, and features for the user's residuals near the row's time: the mean residual
on the same day / week, on the user's nearest ratings before and after, and the closest-books
kNN weighted by time apart (14 / 90 / 365 days). Time in the base alone added little (−0.6%), as
the professor expected for this data; the user's recent residuals helped more. Every source beat
noise when removed from the full model. Outer-holdout MSE: first stack 0.6931 → 0.6760 (−0.016
on each of four splits); public: bias only 0.7626 → first stack 0.7084 → 0.6881.

**Task 4 (interaction prediction with timestamps).** A gradient-boosted pairwise combiner takes
the BPR score plus time features at the row's timestamp t (whether the book had appeared yet,
its age, and its share of interactions in t's time bin at 20/50/100 bins), cosine similarity to
the user's history, EASE (λ = 25), and time-local similarity. Decoys ignore time, so a candidate
that first appears after t is almost always the decoy (97% on about a quarter of rows); boosted
trees learn how much to trust that signal relative to the others, which beat logistic regression
(0.9041 vs 0.9018). Test rows are random held-out reads, so the user's other reads on both sides
of t are in training: time-local similarity is the cosine similarity of the candidate to the
user's 20 reads nearest t, summarized by time window (±1 / 7 / 30 days), by the 1 / 3 / 10
nearest, and with exponential decay. It catches series and books read back to back, which an
average over the whole history dilutes, and was the largest gain since the combiner (+0.005, the
same on both validation halves and for every neighbourhood size tried). Validation: BPR + a hard
not-yet-seen rule 0.8803 → 0.9173 → 0.9218 (time-local) → 0.9242 (EASE); public 0.8706 → 0.9277
→ 0.9345 (including the seen-by-user rule).

**Task 5 (next-item prediction).** Task 4's combiner, with two changes: test rows have no
timestamp, so the query time is the user's last training interaction (the next book follows it),
and a first-order transition count from the user's last book is added, along with cosine
similarity to the history. The real next book is more often a recent release than a decoy is,
and sometimes follows the last book directly (e.g. a series); similarity to the user's most
recent books added nothing beyond whole-history similarity, consistent with books being only
weakly sequential. Validation: BPR + hard rules 0.8728 → combiner 0.9029; public 0.8703 → 0.9030
(including the unseen-item and seen-by-user rules).

**Task 6 (time-delta prediction, MAE).** Following the lecture's hint, this is treated as
regression on the user's previous gaps: a gradient-boosted regressor with MAE loss on log gaps,
trained by stepping back one interaction per user. Inputs are the user's last eight gaps and gap
statistics, a shrunk per-user median gap, and, since the test gives the next book, when other
readers read that book relative to the user's last interaction (it can't be read before it first
appears, and its later readers show when it was being read). A co-reading feature adds
book-to-book timing: for readers of one of the user's last five books who also read the next
book, their delay between the two, shifted to the user's own reading times, implies when this
user reaches it (series and books read back to back). Five seeds are averaged. MAE is minimized
by the median, and the median is unchanged by the log transform, so the model still predicts a
median gap while the log scale tames the skewed gaps (median 13 days, mean 71). One correction
comes from how it is trained: the model learns from the gap one step before the one it predicts,
and gaps lengthen toward the end of a user's history (held-out median 44 → 49 days from one step
to the next), so its predictions run short; they are multiplied by 1.07, the best scale on
validation (the best scale itself rose from 1.05 to 1.07 one step later). Validation MAE:
per-user median 12.05M → 7.18M → 7.11M with co-reading (−1.0%) → −0.7% more with deeper trees
(127 leaves) and the scale; public 13.01M (first submission) → 8.76M → 8.64M → 8.54M.

**3. Rules from the test design.** Positives are held out of each user's training history, but
decoys were sampled without excluding it, so a candidate already in the user's history is the
decoy (1.9% of Task 2 rows, 5.1% of Task 4, 0.7% of Task 5); in Task 5, a candidate that never
appears in training is the real item (4.0% of rows). Both came from a Piazza discussion and were
checked in validation before use: on Task 4 the seen-by-user rule predicted +1.97 points and
gave +1.88 on the leaderboard. Task 6's reader-time features rely on the same kind of fact: the
test gives the next book.

**4. What didn't work.** Latent factors (MF / SVD) never beat the bias model on ratings and
added nothing as extra stack members; further similarity variants (Jaccard, top-k co-reading)
saturated on Task 2, as did similarity to the candidates in the user's other test rows (each
holds one real book, but they are just a few more history items for users who already have ~60);
similarity to the user's most recent books and recency windows added nothing on Task 5; tuned
boosting parameters alone moved MSE by under 0.001. On Task 6, how quickly other readers reached
the next book added nothing, and training on several held-out gaps per user made the model
worse; seed-to-seed noise (~0.3%) is as large as the differences between co-reading variants. A
last round found where the rest of the error is: gaps over 90 days are 37% of rows but 80% of
the error, predicted at a third to a half of their length, i.e. breaks in reading the features
cannot foresee; by predicted gap the model is already well calibrated, so per-range scales did
not transfer between backtests, and similarity of the next book to the user's last reads gained
only −0.2% (not shipped). On Task 4, none of these cleared the bar of +0.002 on both validation
halves: how much more often a book is read on the row's weekday, hour or month (−0.0001 to
−0.0004); LightFM (BPR or WARP, with or without item and user side features such as popularity,
release year and activity; at most +0.0009); and EASE over only the reads near t (it already
weighs which history items matter, so localizing added +0.0001 over EASE on the whole history).
Not tried: FPMC for Task 5 and user drift (α_u·dev_u(t), Koren) for Task 3.
