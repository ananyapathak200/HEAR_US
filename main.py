from typing import List, Union
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from profanity_filter import check_profanity
from urgency_rules import find_critical_keywords
import duplicate_detector

# Step 1: Create the FastAPI app
app = FastAPI(title="HearUs ML Service")

# Step 2: Load the trained models once, when the server starts
category_model = joblib.load("models/category_model.pkl")
category_vectorizer = joblib.load("models/category_vectorizer.pkl")
urgency_model = joblib.load("models/urgency_model.pkl")
urgency_vectorizer = joblib.load("models/urgency_vectorizer.pkl")

# Step 3: Request shape (shared by all endpoints)
class ComplaintText(BaseModel):
    text: str

class Candidate(BaseModel):
    id: Union[int, str]
    text: str

class DuplicateRequest(BaseModel):
    text: str
    candidates: List[Candidate]      # recent complaints (same category) from the backend
    similar_threshold: float = 0.6       # optional: score at/above -> similar
    duplicate_threshold: float = 0.8     # optional: score at/above -> duplicate

# Step 4: Endpoints
@app.get("/")
def health_check():
    return {"status": "ok", "service": "HearUs ML Service"}

@app.post("/check/profanity")
def profanity_endpoint(payload: ComplaintText):
    return check_profanity(payload.text)

@app.post("/predict/category")
def predict_category(payload: ComplaintText):
    X = category_vectorizer.transform([payload.text])
    probs = category_model.predict_proba(X)[0]
    best = probs.argmax()
    return {
        "category": str(category_model.classes_[best]),
        "confidence": round(float(probs[best]), 3),
    }

@app.post("/predict/urgency")
def predict_urgency(payload: ComplaintText):
    # 1) ML prediction
    X = urgency_vectorizer.transform([payload.text])
    probs = urgency_model.predict_proba(X)[0]
    best = probs.argmax()
    ml_label = str(urgency_model.classes_[best])
    ml_conf = round(float(probs[best]), 3)

    # 2) Keyword override: force Critical if any high-severity term is present
    keywords = find_critical_keywords(payload.text)
    if keywords:
        return {
            "urgency": "Critical",
            "confidence": 1.0,
            "ml_prediction": ml_label,
            "override_applied": True,
            "matched_keywords": keywords,
        }
    return {
        "urgency": ml_label,
        "confidence": ml_conf,
        "ml_prediction": ml_label,
        "override_applied": False,
        "matched_keywords": [],
    }

@app.post("/check/duplicates")
def check_duplicates(payload: DuplicateRequest):
    try:
        matches = duplicate_detector.find_duplicates(
            payload.text,
            [c.model_dump() for c in payload.candidates],
            similar_threshold=payload.similar_threshold,
            duplicate_threshold=payload.duplicate_threshold,
        )
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="sentence-transformers is not installed. Run: pip install sentence-transformers",
        )
    return {
        "is_duplicate": any(x["level"] == "duplicate" for x in matches),
        "has_similar": len(matches) > 0,
        "matches": matches,
    }