import pandas as pd
import torch
import numpy as np
import json
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
# Path modelli 
TEST_CSV_PATH = "./datasets/test_full.csv"
MODEL1_PATH = "./results/model1/best_model"
MODEL2_PATH = "./results/model2/best_model"

# =====================
# 1. Caricamento e inizializzazione modelli
# =====================

df = pd.read_csv(TEST_CSV_PATH)
tokenizer = RobertaTokenizer.from_pretrained(MODEL1_PATH)
model1 = RobertaForSequenceClassification.from_pretrained(MODEL1_PATH)
model2 = RobertaForSequenceClassification.from_pretrained(MODEL2_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device).eval()
model2.to(device).eval()


# =====================
# 2. Tokenizzazione 
# =====================

def tokenize_texts(texts):
    return tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
        )

# =====================
# 3. Predizioni con modello 1 (neutral vs non-neutral)
# =====================

df["label_model1"] = df["label"].apply(lambda x: 0 if x == "neutral" else 1)

inputs1 = tokenize_texts(df["text"].tolist()).to(device)
with torch.no_grad():
    outputs1 = model1(**inputs1)
    probs1 = F.softmax(outputs1.logits, dim=-1).cpu().numpy()
df["pred_model1"] = np.argmax(probs1, axis=1)
df["conf_model1_non_neutral"] = probs1[:, 1]

# # Salvataggio predizioni modello 1
# df.to_csv("predizioni_model1.csv", index=False)

# =====================
# 4. Filtraggio per modello 2
# =====================

filtered_df = df[
    (df["pred_model1"] == 1) &
    (df["conf_model1_non_neutral"] > 0.8) &
    (df["label"].isin(["change", "sustain"]))
].copy()
filtered_df["label_model2"] = filtered_df["label"].map({"change": 0, "sustain": 1})

# =====================
# 5. Predizioni modello 2
# =====================

inputs2 = tokenize_texts(filtered_df["text"].tolist()).to(device)
with torch.no_grad():
    outputs2 = model2(**inputs2)
    probs2 = F.softmax(outputs2.logits, dim=-1).cpu().numpy()
filtered_df["pred_model2"] = np.argmax(probs2, axis=1)

# # Salvataggio predizioni modello 2
# filtered_df.to_csv("predizioni_model2_confidence_filtered.csv", index=False)

# =====================
# 6. Calcolo metriche modello 1
# =====================

labels1 = df["label_model1"].values
preds1 = df["pred_model1"].values

precision1, recall1, f1_1, _ = precision_recall_fscore_support(labels1, preds1, average="weighted", zero_division=0)
acc1 = accuracy_score(labels1, preds1)

metrics_model1 = {
    "eval_accuracy": acc1,
    "eval_precision": precision1,
    "eval_recall": recall1,
    "eval_f1": f1_1
}

# =====================
# 7. Calcolo metriche modello 2
# =====================

labels2 = filtered_df["label_model2"].values
preds2 = filtered_df["pred_model2"].values

precision2, recall2, f1_2, _ = precision_recall_fscore_support(labels2, preds2, average="weighted", zero_division=0)
acc2 = accuracy_score(labels2, preds2)

metrics_model2 = {
    "eval_accuracy": acc2,
    "eval_precision": precision2,
    "eval_recall": recall2,
    "eval_f1": f1_2
}

# =====================
# 8. Salvataggio metriche 
# =====================

metriche_finali = {
    "model1": metrics_model1,
    "model2_filtered": metrics_model2,
}

with open("metriche.json", "w") as f:
    json.dump(metriche_finali, f, indent=4)


# =====================
# 8. Rappresentazione grafica delle metriche 
# =====================

# Probabilità predette per la classe 'sustain' 
y_probs = probs2[:, 1]
y_true = labels2

# Costruzione curve
thresholds = np.linspace(0.5, 1.0, 101)
precisions, recalls, f1s = [], [], []

# Calcolo delle metriche per ogni soglia
for t in thresholds:
    y_pred = (y_probs >= t).astype(int)
    precisions.append(precision_score(y_true, y_pred, zero_division=0))
    recalls.append(recall_score(y_true, y_pred, zero_division=0))
    f1s.append(f1_score(y_true, y_pred, zero_division=0))

# Rappresentazione grafica
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, label='Precision', color='blue')
plt.plot(thresholds, recalls, label='Recall', color='green')
plt.plot(thresholds, f1s, label='F1 Score', color='red')
plt.xlabel('Soglia di confidenza (classe sustain)')
plt.ylabel('Valore metrica')
plt.title('Metriche al variare della soglia (Model 2: change vs sustain)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()