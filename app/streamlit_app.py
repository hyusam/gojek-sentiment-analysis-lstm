import streamlit as st
import requests
import pandas as pd
import altair as alt
from streamlit_autorefresh import st_autorefresh

# ============================================================
# Konfigurasi
# ============================================================
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Gojek Sentiment Analysis",
    page_icon="🏍️",
    layout="centered"
)
# Auto-refresh setiap 60 detik
st_autorefresh(interval=60_000, key="auto_refresh")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("TENTANG PROJECT")
    st.markdown("""
    **GoSentiment**  
    Sentiment Analysis pada ulasan aplikasi Gojek menggunakan Deep Learning (BiLSTM).
    """)

    st.divider()

    st.markdown("**Model Performance**")
    st.metric("Test Accuracy", "92.04%")
    st.metric("Test Loss", "0.2384")

    st.divider()

    st.markdown("**Arsitektur**")
    st.markdown("""
    - Model: BiLSTM (TensorFlow/Keras)
    - API: FastAPI
    - Database: PostgreSQL
    - Frontend: Streamlit
    """)

    st.divider()

    st.markdown("**Links**")
    st.markdown("[📂 GitHub Repository](https://github.com/hyusam/gojek-sentiment-analysis-lstm)")

    st.divider()
    st.caption("🔄 Auto-refresh setiap 60 detik")

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

st.divider()

# ============================================================
# Section 3: Live Sentiment Trend (dari database)
# ============================================================
st.subheader("Live Sentiment Trend")
st.caption("Trend sentiment berdasarkan seluruh data prediksi yang tersimpan di database.")

try:
    trend_response = requests.get(f"{API_URL}/history", params={"limit": 1000}, timeout=10)
    trend_response.raise_for_status()
    trend_data = trend_response.json()

    if len(trend_data) >= 2:
        df_trend = pd.DataFrame(trend_data)
        df_trend["created_at"] = pd.to_datetime(df_trend["created_at"])

        # Agregasi per hari (karena data live biasanya belum sebanyak data historis)
        df_trend["date"] = df_trend["created_at"].dt.date
        daily_trend = df_trend.groupby(["date", "sentiment"]).size().unstack(fill_value=0)

        if "Positif" not in daily_trend.columns:
            daily_trend["Positif"] = 0
        if "Negatif" not in daily_trend.columns:
            daily_trend["Negatif"] = 0

        daily_trend["Total"] = daily_trend["Positif"] + daily_trend["Negatif"]
        daily_trend["Negative_Ratio"] = daily_trend["Negatif"] / daily_trend["Total"]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Jumlah Prediksi per Hari**")

            chart_data = daily_trend[["Positif", "Negatif"]].reset_index()
            chart_data_melted = chart_data.melt(id_vars="date", var_name="Sentiment", value_name="Jumlah")

            chart1 = alt.Chart(chart_data_melted).mark_line(point=True).encode(
                x=alt.X("date:T", title="Tanggal"),
                y=alt.Y("Jumlah:Q", title="Jumlah Prediksi"),
                color=alt.Color(
                    "Sentiment:N",
                    scale=alt.Scale(domain=["Positif", "Negatif"], range=["#3498db", "#e74c3c"])
                )
            ).properties(height=300)

            st.altair_chart(chart1, use_container_width=True)

        with col2:
            st.markdown("**Rasio Sentiment Negatif**")

            ratio_data = daily_trend[["Negative_Ratio"]].reset_index()

            chart2 = alt.Chart(ratio_data).mark_line(point=True, color="#e74c3c").encode(
                x=alt.X("date:T", title="Tanggal"),
                y=alt.Y("Negative_Ratio:Q", title="Rasio Negatif")
            ).properties(height=300)

            rule = alt.Chart(pd.DataFrame({'y': [0.5]})).mark_rule(
                color="gray", strokeDash=[4, 4]
            ).encode(y="y:Q")

            st.altair_chart(chart2 + rule, use_container_width=True)

        latest_ratio = daily_trend["Negative_Ratio"].iloc[-1]
        if latest_ratio > 0.5:
            st.warning(f"⚠️ Rasio sentiment negatif hari terakhir: **{latest_ratio*100:.1f}%** — di atas 50%, perlu diperhatikan.")
        else:
            st.success(f"✅ Rasio sentiment negatif hari terakhir: **{latest_ratio*100:.1f}%** — masih terkendali.")

    else:
        st.info("Belum cukup data untuk menampilkan trend. Coba analisis beberapa review terlebih dahulu.")

except requests.exceptions.ConnectionError:
    st.warning("Tidak bisa mengambil data trend — pastikan API sudah jalan.")
except Exception as e:
    st.warning(f"Gagal menampilkan trend: {e}")