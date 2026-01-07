import numpy as np

# Optional sklearn metrics (nice to have). If unavailable, we still run.
try:
    from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, accuracy_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False

test = np.load('/work/data/processed/test.npz')

# Load triplets (support multiple key styles)
def get_key(d, keys):
    for k in keys:
        if k in d.files:
            return d[k]
    raise KeyError("None of these keys exist: " + str(keys))

X1 = get_key(test, ['X_1_test', 'X_1'])
X2 = get_key(test, ['X_2_test', 'X_2'])
X3 = get_key(test, ['X_3_test', 'X_3'])

X1 = X1.astype(np.float32)
X2 = X2.astype(np.float32)
X3 = X3.astype(np.float32)

# Ensure 4D (N,H,W,1)
if X1.ndim == 3: X1 = X1[..., None]
if X2.ndim == 3: X2 = X2[..., None]
if X3.ndim == 3: X3 = X3[..., None]

def batch_l2(a, b):
    diff = a - b
    return np.sqrt(np.sum(diff * diff, axis=(1, 2, 3)))

# Distances
d_pos = batch_l2(X1, X2)  # genuine
d_neg = batch_l2(X1, X3)  # forged

# Build binary classification set:
# smaller distance => more likely genuine
dist = np.concatenate([d_pos, d_neg], axis=0)
y    = np.concatenate([np.ones_like(d_pos, dtype=np.int32),
                       np.zeros_like(d_neg, dtype=np.int32)], axis=0)

# Score: higher score => more genuine (invert distance)
score = -dist

print("=== Threshold evaluation (genuine vs forged) ===")
print("Genuine samples:", len(d_pos), "Forged samples:", len(d_neg))
print("Mean distance genuine:", float(np.mean(d_pos)), "forged:", float(np.mean(d_neg)))

# ROC-AUC (if sklearn present)
if HAVE_SK:
    auc = roc_auc_score(y, score)
    print("ROC-AUC:", float(auc))
else:
    print("ROC-AUC: sklearn not available (skipping)")

# Find best threshold on DISTANCE (simple + robust)
# Predict genuine if dist <= threshold
cands = np.unique(dist)
# To reduce time if large, subsample thresholds
if len(cands) > 2000:
    idx = np.linspace(0, len(cands)-1, 2000).astype(int)
    cands = cands[idx]

best = None
best_thr = None

for thr in cands:
    pred = (dist <= thr).astype(np.int32)
    acc = float(np.mean(pred == y))
    if (best is None) or (acc > best):
        best = acc
        best_thr = float(thr)

print("Best threshold (distance):", best_thr)
print("Best accuracy:", best)

# Final metrics at best threshold
pred = (dist <= best_thr).astype(np.int32)

if HAVE_SK:
    cm = confusion_matrix(y, pred)  # rows: true [0,1], cols: pred [0,1]
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average='binary')
    acc = accuracy_score(y, pred)
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(cm)
    print("Accuracy:", float(acc))
    print("Precision:", float(prec), "Recall:", float(rec), "F1:", float(f1))
else:
    # Manual confusion matrix
    TN = int(np.sum((y==0) & (pred==0)))
    FP = int(np.sum((y==0) & (pred==1)))
    FN = int(np.sum((y==1) & (pred==0)))
    TP = int(np.sum((y==1) & (pred==1)))
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print([[TN, FP],[FN, TP]])
    acc = float(np.mean(pred == y))
    prec = float(TP / (TP + FP)) if (TP + FP) else 0.0
    rec  = float(TP / (TP + FN)) if (TP + FN) else 0.0
    f1   = float(2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
    print("Accuracy:", acc)
    print("Precision:", prec, "Recall:", rec, "F1:", f1)

# Also report EER-ish threshold (approx where FAR ~= FRR)
# FAR: forged predicted genuine; FRR: genuine predicted forged
fars = []
frrs = []
ths  = []

for thr in cands:
    pred_genuine = (dist <= thr).astype(np.int32)
    FAR = float(np.mean(pred_genuine[y==0] == 1))  # false accept rate
    FRR = float(np.mean(pred_genuine[y==1] == 0))  # false reject rate
    fars.append(FAR); frrs.append(FRR); ths.append(float(thr))

fars = np.array(fars); frrs = np.array(frrs); ths = np.array(ths)
eer_idx = int(np.argmin(np.abs(fars - frrs)))
print("Approx EER threshold (distance):", float(ths[eer_idx]))
print("FAR:", float(fars[eer_idx]), "FRR:", float(frrs[eer_idx]), "EER~:", float((fars[eer_idx]+frrs[eer_idx])/2.0))
