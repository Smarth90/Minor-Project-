import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/processed/indian_news_dataset.csv")

X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LinearSVC()

model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))


pickle.dump(model, open("models/svm_model.pkl","wb"))
pickle.dump(vectorizer, open("models/tfidf_vectorizer.pkl","wb"))