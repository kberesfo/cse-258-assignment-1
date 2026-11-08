CSE 258 Assignment 1 — Writeup
Code: assignment1.py (one section per task; run `python assignment1.py <data_dir> <out_dir>`)

Every task combines standard course models (bias / time-bias, BPR, item-kNN, transition counts, the gap median) with a small learned combiner / ensemble of standard models. Validation sets are built the way the test was built (held-out interactions, popularity-matched decoys), so that a validation gain predicts a leaderboard gain. Optional half-sentence: uniform decoys would make popularity look predictive when it isn’t.
The combiner: logistic regression or gradient-boosted trees takes the base models’ outputs plus user/item features (activity, popularity, time) and learns how to weigh them. For ratings it predicts the base model’s residual, so it only corrects the base.
Two-stage fit: base models are fit on a reduced training set, the combiner on held-out rows (so it sees base outputs as they look on unseen data), and then the base models are refit on all of train for the test.
Rules: where the test’s construction makes an answer certain (for example, a decoy already in the user’s history), a hard rule overrides the combiner (see Section 3). 

Task 1 (rating prediction):


Task 2 (interaction prediction):
TODO

Task 3 (rating prediction with timestamps):
Latent factor model (Surprise SVD: global offset + user/item biases + K-dim user/item factors, SGD on observed ratings). Hyperparameters chosen on a fixed 90/10 validation split; final model retrained on all training data. TODO: final settings + validation MSE vs baseline.

Task 4 (interaction prediction with timestamps):
TODO

Task 5 (next-item prediction):
TODO

Task 6 (time delta prediction):
I used Shrunk Per-User Median instead of mean, so 
