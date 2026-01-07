import os
import numpy as np
from .utils import process

# interim folders created by clean_pattern_b.py
BASE = os.path.join(os.path.dirname(__file__), "../../data/interim")
REAL_DIR = os.path.join(BASE, "real")
FORG_DIR = os.path.join(BASE, "forged")

OUT_DIR = os.path.join(os.path.dirname(__file__), "../../data/processed")

# (H, W) expected by many legacy signature repos
IMAGE_SHAPE = (155, 220)

def list_files(d):
    return sorted([os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])

def writer_id_from_name(path):
    # our adapter names: 049_train_0.png etc -> writer id is first token
    base = os.path.basename(path)
    return base.split("_")[0]

def build_index(files):
    idx = {}
    for p in files:
        wid = writer_id_from_name(p)
        idx.setdefault(wid, []).append(p)
    return idx

def make_triplets(real_idx, forg_idx, max_triplets_per_writer=50, seed=42):
    rng = np.random.RandomState(seed)
    X1, X2, X3 = [], [], []

    writers = sorted(set(real_idx.keys()) & set(forg_idx.keys()))
    for wid in writers:
        reals = real_idx.get(wid, [])
        forgs = forg_idx.get(wid, [])
        if len(reals) < 2 or len(forgs) < 1:
            continue

        # create triplets: (anchor real, positive real, negative forged)
        # sample to avoid exploding size
        count = min(max_triplets_per_writer, len(reals) * 2)
        for _ in range(count):
            a, p = rng.choice(reals, size=2, replace=False)
            n = forgs[rng.randint(0, len(forgs))]

            try:
                ia = process(a, image_shape=IMAGE_SHAPE)
                ip = process(p, image_shape=IMAGE_SHAPE)
                ineg = process(n, image_shape=IMAGE_SHAPE)
            except Exception:
                continue

            X1.append(ia); X2.append(ip); X3.append(ineg)

    X1 = np.array(X1, dtype=np.float32)
    X2 = np.array(X2, dtype=np.float32)
    X3 = np.array(X3, dtype=np.float32)
    return X1, X2, X3

def save_npz(path, X1, X2, X3):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        X_1_train=X1, X_2_train=X2, X_3_train=X3,
        X_1=X1, X_2=X2, X_3=X3,
        X1=X1, X2=X2, X3=X3
    )
    print("Wrote:", path, "triplets:", len(X1))

if __name__ == "__main__":
    real_files = list_files(REAL_DIR)
    forg_files = list_files(FORG_DIR)

    if len(real_files) == 0 or len(forg_files) == 0:
        raise RuntimeError("Interim folders empty: real=%d forged=%d" % (len(real_files), len(forg_files)))

    real_idx = build_index(real_files)
    forg_idx = build_index(forg_files)

    X1, X2, X3 = make_triplets(real_idx, forg_idx)

    if len(X1) == 0:
        raise RuntimeError("No triplets created. Check writer ids + images readability.")

    # shuffle + split 80/10/10
    n = len(X1)
    idx = np.arange(n)
    np.random.shuffle(idx)
    X1, X2, X3 = X1[idx], X2[idx], X3[idx]

    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)

    X1_tr, X2_tr, X3_tr = X1[:n_train], X2[:n_train], X3[:n_train]
    X1_va, X2_va, X3_va = X1[n_train:n_train+n_valid], X2[n_train:n_train+n_valid], X3[n_train:n_train+n_valid]
    X1_te, X2_te, X3_te = X1[n_train+n_valid:], X2[n_train+n_valid:], X3[n_train+n_valid:]

    # ensure non-empty valid/test
    if len(X1_va) == 0:
        X1_va, X2_va, X3_va = X1_tr[:1], X2_tr[:1], X3_tr[:1]
    if len(X1_te) == 0:
        X1_te, X2_te, X3_te = X1_va[:1], X2_va[:1], X3_va[:1]

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, "train.npz"), X_1_train=X1_tr, X_2_train=X2_tr, X_3_train=X3_tr)
    np.savez(os.path.join(OUT_DIR, "valid.npz"), X_1_valid=X1_va, X_2_valid=X2_va, X_3_valid=X3_va)
    np.savez(os.path.join(OUT_DIR, "test.npz"),  X_1_test=X1_te,  X_2_test=X2_te,  X_3_test=X3_te)

    print("Done. train/valid/test written to:", OUT_DIR)
    print("train:", len(X1_tr), "valid:", len(X1_va), "test:", len(X1_te))
