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
import csv

# =====================
# 1. Caricamento e bilanciamento dataset
# =====================
df = pd.read_csv("dataset_mi_neutral.csv")
neutral_df = df[df['label'] == 'neutral']
non_neutral_df = df[df['label'] == 'non_neutral']

# Oversampling di non_neutral
non_neutral_oversampled = non_neutral_df.sample(len(neutral_df), replace=True, random_state=42)
balanced_df = pd.concat([neutral_df, non_neutral_oversampled])

# Mapping delle etichette 
label_map = {'neutral': 0, 'non_neutral': 1}
balanced_df['label'] = balanced_df['label'].map(label_map)

# =====================
# 2. Split in training e validation
# =====================

# Primo split: 60% train
train_df, temp_df = train_test_split(
    balanced_df,
    test_size=0.4,
    stratify=balanced_df["label"],
    random_state=42
)

# Secondo split: 30% test, 10% validation
test_df, val_df = train_test_split(
    temp_df, 
    test_size=0.25, 
    stratify=temp_df["label"], 
    random_state=42
)

# Conversione in Dataset Hugging Face
train_dataset = Dataset.from_pandas(train_df).remove_columns("__index_level_0__").rename_column("label", "labels")
val_dataset = Dataset.from_pandas(val_df).remove_columns("__index_level_0__").rename_column("label", "labels")
test_dataset = Dataset.from_pandas(test_df).remove_columns("__index_level_0__").rename_column("label", "labels")


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
test_dataset = test_dataset.map(tokenize_function, batched=True)


test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
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
# 5. Caricamento modello
# =====================
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# =====================
# 6. Training 
# =====================
training_args = TrainingArguments(
    output_dir="./results/neutral_non_neutral/checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  
    weight_decay=0.01,
    logging_dir="./results/neutral_non_neutral/logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

# Inizializzazione Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Training
trainer.train()

# =====================
# 7. Valutazione su test set
# =====================

metrics = trainer.evaluate(eval_dataset=test_dataset)
print(metrics)

with open("./results/neutral_non_neutral/metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(metrics.keys())
    writer.writerow(metrics.values())


# =====================
# 8. Salvataggio
# =====================

trainer.save_model("./results/neutral_non_neutral/best_model")    
tokenizer.save_pretrained("./results/neutral_non_neutral/best_model")  

 