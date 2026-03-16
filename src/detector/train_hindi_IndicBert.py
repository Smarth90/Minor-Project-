import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

df = pd.read_csv("data/processed/hindi_dataset.csv")
true_samples = df[df["label"] == 1].sample(1500,random_state= 42)
fake_samples = df[df["label"] == 0].sample(1500,random_state= 42)
df = pd.concat([true_samples,fake_samples]).reset_index(drop = True)

print(df.shape)
print(df["label"].value_counts())

train_df, test_df = train_test_split(df,test_size = 0.2, random_state= 42, stratify=df["label"])

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

model_name = "ai4bharat/indic-bert"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


training_args = TrainingArguments(
    output_dir="models/indicbert_hindi",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_steps=50,
    dataloader_pin_memory=False
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)


trainer.train()

trainer.evaluate()