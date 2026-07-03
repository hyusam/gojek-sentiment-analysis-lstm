import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_endpoint():
    """API harus merespons di endpoint root."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_positive_review():
    """Review jelas positif harus diklasifikasi Positif."""
    response = client.post("/predict", json={"text": "aplikasi sangat bagus dan membantu"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "Positif"
    assert data["confidence"] > 0.5


def test_predict_negative_review():
    """Review jelas negatif harus diklasifikasi Negatif."""
    response = client.post("/predict", json={"text": "aplikasi error terus tidak bisa dipakai"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "Negatif"
    assert data["confidence"] > 0.5


def test_predict_empty_text():
    """Teks kosong tetap harus direspons tanpa crash."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 200


def test_predict_missing_field():
    """Request tanpa field 'text' harus mengembalikan error validasi."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_history_endpoint():
    """Endpoint history harus mengembalikan list."""
    response = client.get("/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)