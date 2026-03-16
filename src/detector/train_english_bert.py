import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/indian_news_dataset.csv")
true_samples = df[df["label"] == 1].sample(2000, random_state= 42)
fake_samples = df[df["label"] == 0].sample(2000, random_state= 42)
df = pd.concat([true_samples,fake_samples]).reset_index(drop=True)

print(df.shape)
print(df["label"].value_counts())
train_df, test_df = train_test_split(df, test_size=0.2, random_state= 42, stratify= df["label"])

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

model_name = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], trucation = True, padding = "max_length")
train_dataset = train_dataset.map(tokenize, batched= True)
test_dataset = test_dataset.map(tokenize, batched= True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels = 2
)
training_args = TrainingArguments(
    output_dir="models/roberta_english",
    num_train_epochs= 1,
    per_device_train_batch_size= 16,
    per_device_eval_batch_size= 16,
    eval_strategy= "epoch"
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset= train_dataset,
    eval_dataset= test_dataset
)

trainer.train()
trainer.evaluate()