"""
Database utilities untuk Gojek Sentiment Analysis.
Menyimpan history setiap prediksi sentiment ke PostgreSQL,
supaya bisa dipakai untuk analisis trend (time series) secara
real-time di kemudian hari.

Cara pakai:
    from src.database import init_db, save_prediction, get_all_predictions

    init_db()  # panggil sekali saat API start, bikin tabel kalau belum ada
    save_prediction("aplikasi bagus", "aplikasi bagus", "Positif", 0.98)
"""

import os
from pathlib import Path
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/gojek_sentiment"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class PredictionHistory(Base):
    """Tabel untuk menyimpan history setiap prediksi sentiment."""
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    original_text = Column(String, nullable=False)
    clean_text = Column(String, nullable=False)
    sentiment = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Buat tabel di database kalau belum ada. Aman dipanggil berkali-kali."""
    Base.metadata.create_all(bind=engine)


def save_prediction(original_text: str, clean_text: str, sentiment: str, confidence: float):
    """Simpan satu hasil prediksi ke database."""
    session = SessionLocal()
    try:
        record = PredictionHistory(
            original_text=original_text,
            clean_text=clean_text,
            sentiment=sentiment,
            confidence=confidence,
        )
        session.add(record)
        session.commit()
    finally:
        session.close()


def get_all_predictions(limit: int = 100):
    """Ambil history prediksi terbaru, urut dari yang paling baru."""
    session = SessionLocal()
    try:
        results = (
            session.query(PredictionHistory)
            .order_by(PredictionHistory.created_at.desc())
            .limit(limit)
            .all()
        )
        return results
    finally:
        session.close()