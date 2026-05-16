# dlfa_gru_amazon_review_rating_predictor
Predicts Amazon product ratings (1–5) from review text using GRU-based deep learning models.

Implements and compares two architectures:
- nn.GRUCell — manual recurrence loop with full per-step control
- nn.GRU     — PyTorch's optimised fused CUDA kernel implementation

Both models are evaluated using Spearman's Rank Order Correlation Coefficient (SROCC) on a held-out test set.

Tech stack: Python · PyTorch · torchtext · NLTK · NumPy · pandas · matplotlib · scikit-learn · SciPy

Dataset: Amazon Product Reviews (Kaggle) — uses the 'Text' column as input and 'Score' (1–5) as target.
