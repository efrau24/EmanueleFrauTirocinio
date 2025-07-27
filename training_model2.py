from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# =====================
# 1. Caricamento e bilanciamento dataset
# =====================

df = pd.read_csv("./datasets/dataset_change_sustain.csv")

change_df = df[df['label'] == 'change']
sustain_df = df[df['label'] == 'sustain']

# Oversampling di sustain
sustain_oversampled = sustain_df.sample(len(change_df), replace=True, random_state=42)
balanced_df = pd.concat([change_df, sustain_oversampled])

# Mapping delle etichette 
label_map = {'change': 0, 'sustain': 1}
balanced_df['label'] = balanced_df['label'].map(label_map)



# Split manuale in train/val (80/20)
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(
    balanced_df, 
    test_size=0.2, 
    stratify=balanced_df["label"], 
    random_state=42
)

# Conversione in Dataset Hugging Face
train_dataset = Dataset.from_pandas(train_df).remove_columns("__index_level_0__").rename_column("label", "labels")
val_dataset = Dataset.from_pandas(val_df).remove_columns("__index_level_0__").rename_column("label", "labels")

# =====================
# 2. Tokenizzazione
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
# 3. Definizione metrica
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
# 4. Caricamento modello
# =====================
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# =====================
# 5. Training 
# =====================
training_args = TrainingArguments(
    output_dir="./results/model2/checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  
    weight_decay=0.01,
    logging_dir="./results/model2/logs",
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

trainer.train()

# =====================
# 6. Salvataggio
# =====================

trainer.save_model("./results/model2/best_model")
tokenizer.save_pretrained("./results/model2/best_model")
