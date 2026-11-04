CSE 258 Assignment 1 — Writeup
Code: assignment1.py (one section per task; run `python assignment1.py <data_dir> <out_dir>`)

Task 1 (rating prediction):
Latent factor model (Surprise SVD: global offset + user/item biases + K-dim user/item factors, SGD on observed ratings). Hyperparameters chosen on a fixed 75/25 validation split; final model retrained on all training data. TODO: final settings + validation MSE vs baseline.

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
