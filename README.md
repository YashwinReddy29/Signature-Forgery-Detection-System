# Signature Forgery Detection System (Siamese / Triplet Network)

This project implements an **offline signature forgery detection system** using a
**Siamese Triplet Network** trained on handwritten signature images.

The codebase is adapted from a legacy TensorFlow/Keras implementation and executed
end-to-end using Docker for reproducibility.

---

## 📌 Problem Statement
Given a handwritten signature image, determine whether it is **genuine** or **forged**
by comparing it against known genuine samples of the same writer.

---

## 🧠 Model Overview
- **Architecture:** Siamese CNN with Triplet Loss
- **Inputs:**  
  - Anchor (genuine)  
  - Positive (genuine, same writer)  
  - Negative (forged)
- **Decision Rule:**  
  `distance(anchor, positive) < distance(anchor, negative)`

---

## 📂 Dataset Structure 

sign_data/
├── train/
│ ├── 001/
│ ├── 001_forg/
│ ├── ...
│
├── test/
│ ├── 049/
│ ├── 049_forg/
│ ├── ...


- `train`: writers 001–069  
- `test`: writers 049–069  
- Genuine and forged signatures are separated by `_forg`

---

## 🐳 Environment (Required)

This project uses **Docker** due to legacy TensorFlow dependencies.

- Docker
- TensorFlow 1.12 (Python 3)
- Keras 2.2.4

---

## ▶️ How to Run (Step-by-Step)

### 1️⃣ Start Docker container

```bash
docker run -it --rm \
  -v "$PWD":/work \
  -v "/mnt/d/Projects/Signature Forgery Detection System/sign_data":/dataset \
  -w /work \
  tensorflow/tensorflow:1.12.0-py3 \
  bash
2️⃣ Convert dataset to zip-expected format
cd src/data
python clean_pattern_b.py


This creates:

data/interim/real/
data/interim/forged/

3️⃣ Preprocess (create triplets)
cd /work
python -m src.data.preprocess


Generates:

data/processed/train.npz
data/processed/valid.npz
data/processed/test.npz

4️⃣ Train the model
python -m src.model.model

5️⃣ Evaluate (Triplet Accuracy)
python evaluate.py

6️⃣ Threshold-based Evaluation (Forgery Detection)
python threshold_eval.py


Metrics reported:

Accuracy

Precision / Recall / F1

ROC-AUC

FAR / FRR

EER (Equal Error Rate)

📊 Results (Example)

ROC-AUC: ~0.85–0.90

Triplet Accuracy: ~0.80–0.90

EER: ~0.15–0.25

(Exact values depend on dataset size and writer variability.)

🎓 Academic Notes

Writer-dependent evaluation

Triplet learning improves inter-class separation

EER is the standard biometric metric

🚀 Future Work

Migrate to TensorFlow 2 / PyTorch

Data augmentation

Writer-independent evaluation

Real-time API deployment