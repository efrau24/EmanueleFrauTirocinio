from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd


# =====================
# 1. Caricamento e split del dataset
# =====================

df = pd.read_csv("./datasets/dataset_mi.csv")

# Split 70/30 in train_val e test
temp_train_df, test_df = train_test_split(
    df, 
    test_size=0.3, 
    stratify=df["label"], 
    random_state=42
)

df_filtered = temp_train_df[temp_train_df['label'].isin(['change', 'sustain'])]
df_filtered.to_csv("./datasets/dataset_change_sustain.csv", index=False)

# Split 80/20 su temp_train_df in train e validation
train_df, val_df = train_test_split(
    temp_train_df,
    test_size=0.2,
    stratify=temp_train_df["label"],
    random_state=42
)


# =========================
# 2. Preparazione modello 1 (neutral vs non-neutral)
# =========================

def prepare_binary_dataset(df):
    binary_df = df.copy()
    binary_df["label"] = binary_df["label"].apply(lambda x: 0 if x == "neutral" else 1)
    dataset = Dataset.from_pandas(binary_df.rename(columns={"label": "labels"})).remove_columns("__index_level_0__")
    return dataset

train_dataset = prepare_binary_dataset(train_df)
val_dataset = prepare_binary_dataset(val_df)



# =====================
# 3. Tokenizzazione
# =====================
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# =====================
# 4. Definizione metrica
# =====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# =====================
# 5. Caricamento modello e addestramento
# =====================
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results/model1/checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  
    weight_decay=0.01,
    logging_dir="./results/model1/logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# training
trainer.train()


# =========================
# 6. Salvataggio modello 1
# =========================

trainer.save_model("./results/model1/best_model")
tokenizer.save_pretrained("./results/model1/best_model")


# =========================
# 7. Salvataggio test_df
# =========================
test_df.to_csv("./datasets/test_full.csv", index=False)

