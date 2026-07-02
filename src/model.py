"""
Model loading utilities untuk Gojek Sentiment Analysis.
Dipakai bersama oleh API (FastAPI) dan UI (Streamlit) supaya
logic loading model tidak perlu ditulis ulang di kedua tempat.

Cara pakai:
    from src.model import load_sentiment_model, predict_sentiment

    model, tokenizer, max_len = load_sentiment_model(
        model_path='models/bilstm_sentiment_model_default.keras',
        tokenizer_path='models/tokenizer.pkl',
        max_len_path='models/max_len.pkl'
    )

    hasil = predict_sentiment("aplikasi bagus banget", model, tokenizer, max_len)
"""

import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.preprocessing import clean_text


def load_sentiment_model(model_path: str, tokenizer_path: str, max_len_path: str):
    """
    Load model BiLSTM, tokenizer, dan MAX_LEN dari file.

    Returns:
        model: model Keras yang sudah di-load
        tokenizer: Tokenizer yang sudah di-fit saat training
        max_len: panjang maksimal sequence (int)
    """
    model = load_model(model_path)

    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    with open(max_len_path, 'rb') as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


def predict_sentiment(text: str, model, tokenizer, max_len: int) -> dict:
    """
    Prediksi sentiment untuk satu teks review.

    Pipeline: clean_text -> tokenize -> pad -> predict
    (urutan yang sama persis dengan training)

    Returns:
        dict berisi original_text, clean_text, sentiment, confidence
    """
    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

    prob = float(model.predict(padded, verbose=0)[0][0])
    sentiment = "Positif" if prob > 0.5 else "Negatif"
    confidence = prob if sentiment == "Positif" else 1 - prob

    return {
        "original_text": text,
        "clean_text": cleaned,
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
    }