import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import re


DATA_PATH = r"data\processed\indian_news_dataset.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)          # remove urls
    text = re.sub(r"[^a-zA-Z\s]", "", text)      # keep only alphabets
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data():
    df = pd.read_csv(DATA_PATH)

    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise ValueError("Check column names. Expected 'text' and 'label'.")

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

    return df


def train():
    df = load_data()

    X = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    print("\n=== RESULTS ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model, vectorizer



if __name__ == "__main__":
    train()