# HearUs ML Service

Python (FastAPI) service that adds AI features to the HearUs complaint system.
The Spring Boot backend calls it over HTTP. It does not store any data.

| Feature | Endpoint | What it does |
|---|---|---|
| Health | `GET /` | Checks the service is running |
| Profanity | `POST /check/profanity` | Detects abusive words |
| Category | `POST /predict/category` | Hostel / Academics / Infrastructure / Harassment / Administration |
| Urgency | `POST /predict/urgency` | Low / Medium / High / Critical (ML + keyword override) |
| Duplicates | `POST /check/duplicates` | Finds similar existing complaints |

---

## 1. Setup (one time)

Requires **Python 3.12 or newer** (developed on 3.14) and internet access for the first install.

```bash
git clone <repo-url>
cd <repo-folder>

python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

`sentence-transformers` is a large install (it includes PyTorch). The duplicate-detection model (~90 MB) downloads automatically the **first time** `/check/duplicates` is called, so the first call is slow and needs internet.

## 2. Run

From inside the project folder (where `main.py` is) with the venv active:

```bash
uvicorn main:app --port 8000
```

Add `--reload` during development. Then open:

- http://localhost:8000/docs - interactive test page (Swagger)
- http://localhost:8000/ - health check

Spring Boot usually runs on 8080, so there is no port clash.

---

## 3. API contract

All POST endpoints take JSON and return JSON. **Response field names are snake_case.**

### POST /check/profanity
```json
// request
{"text": "complaint text"}
// response
{"flagged": true, "matched_terms": ["shit"]}
```

### POST /predict/category
```json
// request
{"text": "No water in my hostel room since yesterday"}
// response
{"category": "Hostel", "confidence": 0.7}
```
Categories: `Hostel`, `Academics`, `Infrastructure`, `Harassment`, `Administration`.

### POST /predict/urgency
```json
// request
{"text": "A senior assaulted me and threatened me"}
// response
{
  "urgency": "Critical",
  "confidence": 1.0,
  "ml_prediction": "High",
  "override_applied": true,
  "matched_keywords": ["assaulted", "threatened"]
}
```
Levels: `Low`, `Medium`, `High`, `Critical`.
`override_applied = true` means a safety keyword forced Critical regardless of the ML model.

### POST /check/duplicates
The backend sends the new complaint plus **recent complaints of the same category** (for example the last 30 days, at most 100-200 for speed). The service stores nothing.
```json
// request
{
  "text": "Water is not coming in my room",
  "candidates": [
    {"id": 101, "text": "No water supply in my hostel room since yesterday"},
    {"id": "C-102", "text": "The mess food is stale"}
  ],
  "similar_threshold": 0.6,
  "duplicate_threshold": 0.8
}
// response
{
  "is_duplicate": false,
  "has_similar": true,
  "matches": [{"id": 101, "similarity": 0.66, "level": "similar"}]
}
```
- `similar_threshold` and `duplicate_threshold` are optional (defaults 0.6 and 0.8).
- `id` can be a number or a string and is returned unchanged.
- `level = "duplicate"` (score >= 0.8): almost certainly the same complaint.
- `level = "similar"` (0.6 to 0.8): probably related. Ask the student to confirm.
- No match: the complaint is new.

### Errors
- `422` - invalid request body (for example the `text` field is missing).
- `503` - from `/check/duplicates` only, if `sentence-transformers` is not installed.

---

## 4. Suggested flow in the backend

```
Student submits complaint
  1. /check/profanity   -> if flagged: mask the words or send for review
  2. /predict/category  -> route to the right department
  3. /predict/urgency   -> if Critical: escalate (Dean / ICC)
  4. /check/duplicates  -> send recent same-category complaints from the DB
  5. Save the complaint with category, urgency and duplicate links
```

Suggestions (tune as needed):
- **Do not auto-reject flagged complaints.** A student reporting harassment may quote the abuse. Mask it or flag for review instead.
- If the category `confidence` is low, mark the complaint for manual review.
- If the ML service is down or times out, still save the complaint (for example as "Pending review") instead of failing the submission. Set a connect/read timeout on the HTTP client.

## 5. Spring Boot notes

Python returns `snake_case` (`matched_terms`, `override_applied`, `is_duplicate`...). Jackson expects camelCase by default, so either add this to `application.properties`:

```properties
spring.jackson.property-naming-strategy=SNAKE_CASE
```

or annotate fields with `@JsonProperty("override_applied")`.

Sketch (not compiled; adapt to your project):

```java
public record CategoryResponse(String category, double confidence) {}

public record UrgencyResponse(
    String urgency,
    double confidence,
    @JsonProperty("ml_prediction") String mlPrediction,
    @JsonProperty("override_applied") boolean overrideApplied,
    @JsonProperty("matched_keywords") List<String> matchedKeywords) {}

@Service
public class MlClient {
    private final RestTemplate rest = new RestTemplate();
    private static final String BASE = "http://localhost:8000";

    public CategoryResponse predictCategory(String text) {
        return rest.postForObject(BASE + "/predict/category",
                Map.of("text", text), CategoryResponse.class);
    }

    public UrgencyResponse predictUrgency(String text) {
        return rest.postForObject(BASE + "/predict/urgency",
                Map.of("text", text), UrgencyResponse.class);
    }
}
```

Put the base URL in `application.properties` so it can change when the service is deployed elsewhere.

---

## 6. Project structure

```
main.py                  API endpoints
profanity_filter.py      blocklist + matching
urgency_rules.py         keyword override for Critical
duplicate_detector.py    Sentence-BERT similarity
train_category_model.py  trains the category model
train_urgency_model.py   trains the urgency model
data/complaints.csv      training data (text, category, urgency)
models/*.pkl             saved trained models
requirements.txt
```

## 7. Retraining (ML developer only)

After adding rows to `data/complaints.csv`:

```bash
python train_category_model.py
python train_urgency_model.py
```
Then commit the updated `models/*.pkl` and **restart the service** so it loads the new models.

The `.pkl` files must be loaded with the same scikit-learn version used to train them (pinned in `requirements.txt`).

## 8. Known limitations

- Training data is synthetic and small (200 rows). Category accuracy is about 80% on the held-out test set. Urgency from the ML model alone is weak (about 35-46%), which is why the keyword override exists for Critical cases.
- Duplicate thresholds (0.6 / 0.8) were chosen from manual tests and may need retuning.
- Every duplicate check re-embeds all candidates, so keep the candidate list reasonably small.