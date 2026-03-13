import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/processed/hindi_dataset.csv")
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https\S+", " ", text)
    text = re.sub(r"\n"," ", text)
    text = re.sub(r"[^\w\s]", " ",text)
    return text

df["text"] = df["text"].apply(clean_text)
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 42, stratify = y)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = LinearSVC(class_weight="balanced")

model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n")
print(classification_report(y_test, preds))

pickle.dump(model, open("models/hindi_svm_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/hindi_tfidf_vectorizer.pkl", "wb"))

