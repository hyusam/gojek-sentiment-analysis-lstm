import streamlit as st
import requests
import pandas as pd

# ============================================================
# Konfigurasi
# ============================================================
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Gojek Sentiment Analysis",
    page_icon="🏍️",
    layout="centered"
)

# ============================================================
# Header
# ============================================================
st.title("Gojek Sentiment Analysis")
st.markdown("Klasifikasi sentiment ulasan Gojek secara real-time menggunakan **BiLSTM**.")

st.divider()

# ============================================================
# Section 1: Prediksi Sentiment
# ============================================================
st.subheader("Analisis Review")

user_input = st.text_area(
    "Masukkan teks review:",
    placeholder="Contoh: aplikasinya bagus banget, drivernya ramah dan cepat",
    height=100
)

if st.button("Analisis Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Silakan masukkan teks review terlebih dahulu.")
    else:
        with st.spinner("Menganalisis..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": user_input},
                    timeout=10
                )
                response.raise_for_status()
                result = response.json()

                sentiment = result["sentiment"]
                confidence = result["confidence"]

                col1, col2 = st.columns(2)
                with col1:
                    if sentiment == "Positif":
                        st.success(f"### 😊 {sentiment}")
                    else:
                        st.error(f"### 😞 {sentiment}")
                with col2:
                    st.metric("Confidence", f"{confidence * 100:.2f}%")

                with st.expander("Detail preprocessing"):
                    st.write("**Teks asli:**", result["original_text"])
                    st.write("**Teks setelah cleaning:**", result["clean_text"])

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Tidak bisa terhubung ke API. "
                    "Pastikan API sudah jalan (`fastapi dev main.py` di folder `api/`)."
                )
            except Exception as e:
                st.error(f"Terjadi error: {e}")

st.divider()

# ============================================================
# Section 2: History Prediksi
# ============================================================
st.subheader("History Prediksi")

if st.button("Refresh History"):
    st.rerun()

try:
    history_response = requests.get(f"{API_URL}/history", params={"limit": 50}, timeout=10)
    history_response.raise_for_status()
    history_data = history_response.json()

    if history_data:
        df_history = pd.DataFrame(history_data)
        df_history["created_at"] = pd.to_datetime(df_history["created_at"])
        df_history = df_history.sort_values("created_at", ascending=False)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Prediksi", len(df_history))
        col2.metric("Positif", (df_history["sentiment"] == "Positif").sum())
        col3.metric("Negatif", (df_history["sentiment"] == "Negatif").sum())

        st.dataframe(
            df_history[["created_at", "original_text", "sentiment", "confidence"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Belum ada history prediksi. Coba analisis review di atas.")

except requests.exceptions.ConnectionError:
    st.warning("Tidak bisa mengambil history — pastikan API sudah jalan.")
except Exception as e:
    st.warning(f"Gagal mengambil history: {e}")