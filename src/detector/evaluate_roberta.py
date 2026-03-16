import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from datasets import Dataset

df = pd.read_csv("data/processed/indian_news_dataset.csv")

test_df = pd.read_csv("data/processed/english_test.csv")
test_dataset = Dataset.from_pandas(test_df)


model_path = "models/roberta_english/checkpoint-200"
model = AutoModelForSequenceClassification.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")


def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)


test_dataset = test_dataset.map(tokenize, batched=True)


test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)


model.eval()

predictions = []
true_labels = []

with torch.no_grad():
    for item in test_dataset:
        input_ids = item["input_ids"].unsqueeze(0)
        attention_mask = item["attention_mask"].unsqueeze(0)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

        predictions.append(pred)
        true_labels.append(item["label"])

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nClassification Report:\n")
print(classification_report(true_labels, predictions))