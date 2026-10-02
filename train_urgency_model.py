import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from urgency_rules import find_critical_keywords

LABELS = ["Low", "Medium", "High", "Critical"]

df = pd.read_csv("data/complaints.csv")
X, y = df["text"], df["urgency"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def make_vec():
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2))

def make_model():
    # class_weight="balanced" makes the rare Critical class count more
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=10)

vec = make_vec()
model = make_model().fit(vec.fit_transform(X_train), y_train)
ml_pred = model.predict(vec.transform(X_test))

# Hybrid: keyword override on top of the ML prediction
hybrid_pred = [
    "Critical" if find_critical_keywords(t) else p
    for t, p in zip(X_test, ml_pred)
]

def critical_recall(true, pred):
    return recall_score(true, pred, labels=["Critical"], average=None, zero_division=0)[0]

print("=== ML MODEL ONLY ===")
print(classification_report(y_test, ml_pred, labels=LABELS, zero_division=0))
print(confusion_matrix(y_test, ml_pred, labels=LABELS))
print(f"Critical recall (ML only): {critical_recall(y_test, ml_pred):.2f}\n")

print("=== HYBRID (ML + KEYWORD OVERRIDE) ===")
print(classification_report(y_test, hybrid_pred, labels=LABELS, zero_division=0))
print("Rows = true, cols = predicted, order:", LABELS)
print(confusion_matrix(y_test, hybrid_pred, labels=LABELS))
print(f"Critical recall (hybrid): {critical_recall(y_test, hybrid_pred):.2f}\n")

# Check the override on the FULL dataset (rules are not trained, so this is fair)
all_flag = df["text"].apply(lambda t: bool(find_critical_keywords(t)))
true_crit = df["urgency"] == "Critical"
print("Override on full dataset:")
print(f"  Critical caught by keywords: {(all_flag & true_crit).sum()} of {true_crit.sum()}")
print(f"  Non-Critical wrongly forced to Critical: {(all_flag & ~true_crit).sum()} of {(~true_crit).sum()}")
print("  Critical complaints the keywords MISS:")
for t in df.loc[true_crit & ~all_flag, "text"]:
    print("   -", t)

# Retrain on ALL data before saving
vec = make_vec()
model = make_model().fit(vec.fit_transform(X), y)
joblib.dump(model, "models/urgency_model.pkl")
joblib.dump(vec, "models/urgency_vectorizer.pkl")
print("\nUrgency model saved successfully!")