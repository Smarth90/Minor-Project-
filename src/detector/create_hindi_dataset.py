from transformers import MarianMTModel, MarianTokenizer
import pandas as pd

df = pd.read_csv("data/processed/indian_news_dataset.csv")

true_samples = df[df["label"] == 1].sample(3333, random_state=42)
fake_samples = df[df["label"] == 0].sample(1667, random_state=42)

df = pd.concat([true_samples, fake_samples]).reset_index(drop=True)

model_name = "Helsinki-NLP/opus-mt-en-hi"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

translated = []

for text in df["text"]:
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    output = model.generate(**tokens)
    translated.append(tokenizer.decode(output[0], skip_special_tokens=True))

df["text"] = translated

df.to_csv("data/processed/hindi_dataset.csv", index=False)

print("Hindi dataset created:", df.shape)