# GoSentiment — Gojek App Review Sentiment Analysis

Sistem klasifikasi sentiment ulasan aplikasi Gojek menggunakan Deep Learning (BiLSTM), lengkap dengan pipeline end-to-end dari data understanding, modeling, hingga deployment sebagai REST API dan dashboard interaktif.

## Latar Belakang

Gojek sebagai super-app terbesar di Indonesia menerima ribuan ulasan pengguna setiap harinya di Google Play Store. Perubahan kebijakan, gangguan layanan, maupun pembaruan fitur selalu memicu lonjakan ulasan yang tidak mungkin dimonitor secara manual dalam skala besar.

Proyek ini membangun sistem klasifikasi sentiment otomatis berbasis LSTM untuk mendeteksi pergeseran opini pengguna secara real-time, membantu tim produk merespons lebih cepat sebelum sentimen negatif berdampak pada reputasi brand.

## Dataset

- **Sumber:** [Gojek App Reviews Bahasa Indonesia](https://www.kaggle.com/datasets/ucupsedaya/gojek-app-reviews-bahasa-indonesia) (Kaggle, oleh ucupsedaya)
- **Jumlah:** 225.002 ulasan, periode November 2021 – Februari 2024
- **Kolom:** `userName`, `content`, `score` (1–5), `at`, `appVersion`
- **Label:** Binary sentiment — score 4–5 → Positif, score 1–2 → Negatif (score 3/netral di-drop)
- **Setelah preprocessing:** 86.158 baris, balanced (43.079 Positif / 43.079 Negatif)

## Arsitektur & Hasil Model

Dua model Deep Learning dibangun dan dibandingkan:

| Model | Test Accuracy | Test Loss |
|---|---|---|
| DNN (Baseline) | 91.75% | 0.2438 |
| **BiLSTM (Main)** ✅ | **92.04%** | **0.2384** |
| BiLSTM (Tuned) | 91.99% | 0.2389 |

**BiLSTM (Main)** dipilih sebagai model final — akurasi tertinggi, loss terendah, dan trade-off precision-recall paling seimbang antar kedua kelas sentiment.

Detail preprocessing (cleaning, slang normalization, stemming Sastrawi, negation-preserving stopword removal), EDA, dan evaluasi lengkap ada di folder `notebooks/`.

## Fitur Tambahan: Time Series Sentiment Trend

Selain klasifikasi statis, project ini juga menganalisis tren sentiment dari waktu ke waktu (agregasi mingguan). Beberapa lonjakan sentiment negatif berhasil divalidasi bertepatan dengan insiden nyata (misalnya gangguan aplikasi Gojek pada April 2023) — membuktikan model mampu menangkap sinyal dari peristiwa dunia nyata, bukan sekadar noise statistik.

## Arsitektur Sistem

```
┌─────────────┐      ┌──────────────┐      ┌───────────────┐
│  Streamlit  │ ───> │   FastAPI    │ ───> │  BiLSTM Model │
│  (Frontend) │ <─── │  (/predict)  │ <─── │  + Tokenizer  │
└─────────────┘      └──────┬───────┘      └───────────────┘
                             │
                             ▼
                      ┌─────────────┐
                      │ PostgreSQL  │
                      │  (Docker)   │
                      └─────────────┘
```

- **Model:** BiLSTM (TensorFlow/Keras)
- **API:** FastAPI — endpoint `/predict` dan `/history`
- **Database:** PostgreSQL (Docker) — menyimpan history setiap prediksi
- **Frontend:** Streamlit — input teks interaktif + dashboard history
- **CI/CD:** GitHub Actions — automated testing pada setiap push

> **Catatan:** Streamlit UI pada project ini berfungsi sebagai demo interaktif dan dashboard sederhana untuk melihat history prediksi — bukan merepresentasikan alur kerja production sesungguhnya. Pada implementasi production, review baru akan diambil otomatis oleh scraper terjadwal dan dikirim ke API secara otomatis, dengan dashboard berperan sebagai interface monitoring bagi tim produk, bukan sebagai entry point data manual.

## Struktur Project

```
gojek-sentiment-analysis-lstm/
├── api/                    # FastAPI backend
│   └── main.py
├── app/                    # Streamlit frontend
│   └── streamlit_app.py
├── src/                    # Reusable modules
│   ├── preprocessing.py    # Text cleaning pipeline
│   ├── model.py            # Model loading & prediction
│   └── database.py         # Database ORM & queries
├── notebooks/               
│   ├── 01_eda.ipynb        # EDA & preprocessing
│   └── 02_modeling.ipynb   # Modeling, tuning, evaluasi, time series
├── models/                 # Saved model, tokenizer (gitignored)
├── data/                   # Dataset & processed data (gitignored)
├── tests/                  # Unit tests
├── assets/                 # Visualisasi EDA & evaluasi
├── docker-compose.yml      # PostgreSQL container
└── requirements.txt
```

## Cara Menjalankan

### 1. Setup environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Jalankan database
```bash
docker-compose up -d
```

### 3. Jalankan API
```bash
cd api
fastapi dev main.py
```
API tersedia di `http://127.0.0.1:8000`, dokumentasi interaktif di `http://127.0.0.1:8000/docs`.

### 4. Jalankan Streamlit UI
```bash
cd app
streamlit run streamlit_app.py
```
Dashboard tersedia di `http://localhost:8501`.

### 5. Jalankan tests
```bash
pytest tests/ -v
```

## Tech Stack

- **Deep Learning:** TensorFlow / Keras
- **NLP:** Sastrawi (stemming & stopword Bahasa Indonesia)
- **Hyperparameter Tuning:** Keras Tuner (Random Search)
- **Backend:** FastAPI, SQLAlchemy
- **Database:** PostgreSQL (Docker)
- **Frontend:** Streamlit
- **CI/CD:** GitHub Actions
- **Data Analysis:** Pandas, Seaborn, Matplotlib

## Insight Bisnis

- **Pain Points (Negatif):** Masalah aplikasi (lag/error), akun (login/suspend), tarif & promo
- **Value Proposition (Positif):** Driver ramah & membantu, kemudahan transaksi GoPay, kebermanfaatan layanan sehari-hari

## Author

Wahyu Iqsam
