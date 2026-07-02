from fastapi import FastAPI
from pydantic import BaseModel

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import load_sentiment_model, predict_sentiment
from src.database import init_db, save_prediction, get_all_predictions

# ============================================================
# Load model & artifacts (sekali saja saat API pertama kali start)
# ============================================================
model, tokenizer, MAX_LEN = load_sentiment_model(
    model_path='../models/bilstm_sentiment_model_default.keras',
    tokenizer_path='../models/tokenizer.pkl',
    max_len_path='../models/max_len.pkl'
)

print(f'✅ Model, tokenizer, dan MAX_LEN ({MAX_LEN}) berhasil dimuat.')

# Inisialisasi database (bikin tabel kalau belum ada)
init_db()
print('✅ Database siap.')

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="Gojek Sentiment Analysis API")


class ReviewInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    original_text: str
    clean_text: str
    sentiment: str
    confidence: float


@app.get("/")
def root():
    return {"message": "Gojek Sentiment Analysis API is running"}


@app.post("/predict", response_model=PredictionOutput)
def predict(review: ReviewInput):
    result = predict_sentiment(review.text, model, tokenizer, MAX_LEN)

    # Simpan hasil prediksi ke database
    save_prediction(
        original_text=result["original_text"],
        clean_text=result["clean_text"],
        sentiment=result["sentiment"],
        confidence=result["confidence"],
    )

    return PredictionOutput(**result)


@app.get("/history")
def history(limit: int = 20):
    records = get_all_predictions(limit=limit)
    return [
        {
            "id": r.id,
            "original_text": r.original_text,
            "sentiment": r.sentiment,
            "confidence": r.confidence,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]