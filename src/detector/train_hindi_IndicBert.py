import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv("data/processed/hindi_dataset.csv")
df.columns = df.columns.str.strip()

true_samples = df[df["label"] == 1].sample(1500, random_state=42)
fake_samples = df[df["label"] == 0].sample(1500, random_state=42)
df = pd.concat([true_samples, fake_samples]).reset_index(drop=True)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

model_id = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True).remove_columns(['text', '__index_level_0__'])
test_dataset = test_dataset.map(tokenize, batched=True).remove_columns(['text', '__index_level_0__'])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions)
    }

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="models/indicbert_hindi",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
print(trainer.evaluate())