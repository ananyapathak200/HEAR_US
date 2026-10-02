import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/complaints.csv")
X, y = df["text"], df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def make_vec():
    return TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))

candidates = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "NaiveBayes": MultinomialNB(),
}

best_name, best_score = None, -1
for name, clf in candidates.items():
    pipe = make_pipeline(make_vec(), clf)
    cv = cross_val_score(pipe, X, y, cv=5).mean()   # 5-fold cross-validation
    print(f"{name}: 5-fold CV accuracy = {cv:.3f}")
    if cv > best_score:
        best_name, best_score = name, cv

print(f"\nBest model: {best_name}\n")

# Evaluate best model on the held-out test set
vec = make_vec()
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)
model = candidates[best_name].fit(Xtr, y_train)
pred = model.predict(Xte)

print(classification_report(y_test, pred, zero_division=0))
print("Confusion matrix (rows = true, cols = predicted):")
print(model.classes_)
print(confusion_matrix(y_test, pred, labels=model.classes_))

# Retrain on ALL data before saving the final model
vec = make_vec()
Xall = vec.fit_transform(X)
model = candidates[best_name].fit(Xall, y)
joblib.dump(model, "models/category_model.pkl")
joblib.dump(vec, "models/category_vectorizer.pkl")
print("\nModel saved successfully!")