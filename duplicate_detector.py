import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"   # small (~90 MB), fast on CPU, good for sentence similarity
_model = None


def _get_model():
    """Load the Sentence-BERT model once (first call takes a few seconds)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # imported here so other
        _model = SentenceTransformer(MODEL_NAME)                # endpoints work without it
    return _model


def _encode(texts):
    """Turn a list of texts into normalized embedding vectors."""
    return _get_model().encode(texts, normalize_embeddings=True)


def find_duplicates(text, candidates, similar_threshold=0.6, duplicate_threshold=0.8, top_k=5):
    """
    text                : the new complaint (string)
    candidates          : list of {"id": ..., "text": ...} - recent complaints to compare against
    similar_threshold   : score at/above this -> level "similar"   (student should confirm)
    duplicate_threshold : score at/above this -> level "duplicate" (almost certainly the same)
    Returns             : matches, most similar first, each with a "level"
    """
    candidates = [c for c in candidates if c["text"].strip()]
    if not text.strip() or not candidates:
        return []

    vectors = _encode([text] + [c["text"] for c in candidates])
    new_vec, cand_vecs = vectors[0], vectors[1:]

    # vectors are normalized, so dot product == cosine similarity
    scores = cand_vecs @ new_vec

    matches = []
    for c, s in zip(candidates, scores):
        if s >= similar_threshold:
            level = "duplicate" if s >= duplicate_threshold else "similar"
            matches.append({"id": c["id"], "similarity": round(float(s), 3), "level": level})
    matches.sort(key=lambda m: -m["similarity"])
    return matches[:top_k]