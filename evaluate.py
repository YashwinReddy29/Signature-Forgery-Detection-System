import numpy as np

test = np.load('/work/data/processed/test.npz')

# Handle key names safely
X1 = test['X_1_test'] if 'X_1_test' in test.files else test['X_1']
X2 = test['X_2_test'] if 'X_2_test' in test.files else test['X_2']
X3 = test['X_3_test'] if 'X_3_test' in test.files else test['X_3']

# Ensure float32
X1 = X1.astype(np.float32)
X2 = X2.astype(np.float32)
X3 = X3.astype(np.float32)

# Make sure arrays are 4D (N,H,W,C)
if X1.ndim == 3: X1 = X1[..., None]
if X2.ndim == 3: X2 = X2[..., None]
if X3.ndim == 3: X3 = X3[..., None]

# Manual L2 distance per sample: sqrt(sum((A-B)^2))
def batch_l2(a, b):
    diff = a - b
    return np.sqrt(np.sum(diff * diff, axis=(1, 2, 3)))

d_pos = batch_l2(X1, X2)   # anchor vs positive
d_neg = batch_l2(X1, X3)   # anchor vs negative

acc = float(np.mean(d_pos < d_neg))

print("Samples:", len(d_pos))
print("Triplet verification accuracy (d_pos < d_neg):", acc)
print("Mean d_pos:", float(np.mean(d_pos)), "Mean d_neg:", float(np.mean(d_neg)))
