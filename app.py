import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Sentimen Lapas Cibinong",
    page_icon="📊",
    layout="wide"
)

# Judul Dashboard
st.title("📊 Analisis Sentimen & Aspek Temuan Hunian Mewah Lapas Cibinong")
st.caption("Studi Komentar TikTok @jawaposofficial terkait Sidak Ombudsman RI Menggunakan Machine Learning")

# Load Data Hasil Klasifikasi
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Hasil_Aspek_Klasifikasi.csv")
        return df
    except Exception:
        return None

df = load_data()

# Load Model & Vectorizer
@st.cache_resource
def load_models():
    try:
        vec = joblib.load("vectorizer.pkl")
        model = joblib.load("model_svm_sentimen.pkl")
        return vec, model
    except Exception:
        return None, None

vectorizer, model_svm = load_models()

# --- BAGIAN 1: METRIK RINGKASAN (KPI CARDS) ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Komentar Mentah", "397")
col2.metric("Komentar Bersih", "322", "81.1% valid")
col3.metric("Model Sentimen Terbaik", "81.5%", "SVM (Linear)")
col4.metric("Model Aspek Terbaik", "81.8%", "SVM & Logistic Regression")

st.markdown("---")

# --- BAGIAN 2: GRAFIK SENTIMEN & TREN WAKTU ---
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Distribusi Sikap Publik")
    sentiment_data = pd.DataFrame({
        "Sentimen": ["Netral", "Marah", "Dukung"],
        "Jumlah": [196, 111, 15]
    })
    fig_sent = px.pie(
        sentiment_data,
        names="Sentimen",
        values="Jumlah",
        hole=0.55,
        color="Sentimen",
        color_discrete_map={"Netral": "#94a3b8", "Marah": "#ef4444", "Dukung": "#10b981"}
    )
    fig_sent.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
    st.plotly_chart(fig_sent, use_container_width=True)

with col_right:
    st.subheader("5 Hari Komentar Teramai")
    timeline_data = pd.DataFrame({
        "Tanggal": ["24 Sep 2026", "25 Sep 2026", "30 Sep 2026", "01 Okt 2026", "02 Okt 2026"],
        "Komentar": [131, 41, 21, 48, 27]
    })
    fig_time = px.bar(
        timeline_data,
        x="Tanggal",
        y="Komentar",
        text="Komentar",
        color_discrete_sequence=["#3b82f6"]
    )
    fig_time.update_traces(textposition="outside")
    fig_time.update_layout(margin=dict(t=20, b=10, l=10, r=10), height=320, yaxis_title="Jumlah Komentar")
    st.plotly_chart(fig_time, use_container_width=True)

st.markdown("---")

# --- BAGIAN 3: DISTRIBUSI SENTIMEN PER ASPEK ---
st.subheader("Distribusi Sentimen per Kategori Aspek")

aspect_df = pd.DataFrame({
    "Aspek": [
        "Fasilitas Mewah (FML)", "Fasilitas Mewah (FML)", "Fasilitas Mewah (FML)",
        "Integritas Petugas (IPL)", "Integritas Petugas (IPL)", "Integritas Petugas (IPL)",
        "Penegakan Hukum (PHK)", "Penegakan Hukum (PHK)", "Penegakan Hukum (PHK)",
        "Lainnya", "Lainnya", "Lainnya"
    ],
    "Sentimen": ["Marah", "Netral", "Dukung"] * 4,
    "Jumlah": [22, 74, 1, 27, 17, 8, 43, 21, 4, 19, 84, 2]
})

fig_aspect = px.bar(
    aspect_df,
    x="Aspek",
    y="Jumlah",
    color="Sentimen",
    barmode="stack",
    color_discrete_map={"Netral": "#94a3b8", "Marah": "#ef4444", "Dukung": "#10b981"}
)
fig_aspect.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
st.plotly_chart(fig_aspect, use_container_width=True)

st.markdown("---")

# --- BAGIAN 4: EVALUASI 7 ALGORITMA ML ---
st.subheader("Evaluasi 7 Algoritma Machine Learning")
tipe_eval = st.radio("Pilih Evaluasi Klasifikasi:", ["Sentimen (80:20)", "Aspek (80:20)"], horizontal=True)

if "Sentimen" in tipe_eval:
    model_res = pd.DataFrame({
        "Model": ["SVM", "Logistic Regression", "Random Forest", "Neural Network", "KNN", "Naive Bayes", "Decision Tree"],
        "Akurasi (%)": [81.5, 80.0, 78.5, 78.5, 73.8, 73.8, 73.8]
    }).sort_values(by="Akurasi (%)", ascending=True)
    bar_color = "#6366f1"
else:
    model_res = pd.DataFrame({
        "Model": ["SVM", "Logistic Regression", "Random Forest", "Neural Network", "Decision Tree", "Naive Bayes", "KNN"],
        "Akurasi (%)": [81.8, 81.8, 79.5, 77.3, 75.0, 72.7, 59.1]
    }).sort_values(by="Akurasi (%)", ascending=True)
    bar_color = "#10b981"

fig_models = px.bar(
    model_res,
    x="Akurasi (%)",
    y="Model",
    orientation="h",
    text="Akurasi (%)",
    color_discrete_sequence=[bar_color]
)
fig_models.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_models.update_layout(xaxis=dict(range=[50, 95]), height=320, margin=dict(t=10, b=10, l=10, r=10))
st.plotly_chart(fig_models, use_container_width=True)

st.markdown("---")

# --- BAGIAN 5: LIVE TESTER PREDIKSI KOMENTAR ---
st.subheader("🧪 Uji Coba Prediksi Komentar Baru")

user_comment = st.text_area(
    "Masukkan contoh komentar TikTok:",
    placeholder="Contoh: 'pecat petugas yang menghalangi ombudsman!' atau 'kamarnya lebih mewah dari rumah saya'"
)

if st.button("Prediksi Sentimen", type="primary"):
    if user_comment.strip():
        if vectorizer is not None and model_svm is not None:
            # Transformasi teks & Prediksi SVM
            vec_input = vectorizer.transform([user_comment])
            prediksi = model_svm.predict(vec_input)[0]

            if prediksi == "Marah":
                st.error(f"Hasil Prediksi Model: **{prediksi}** 😡")
            elif prediksi == "Dukung":
                st.success(f"Hasil Prediksi Model: **{prediksi}** 👍")
            else:
                st.info(f"Hasil Prediksi Model: **{prediksi}** 😐")
        else:
            st.warning("Model belum dimuat. Pastikan file 'vectorizer.pkl' dan 'model_svm_sentimen.pkl' berada di direktori yang sama.")
    else:
        st.warning("Silakan ketik komentar terlebih dahulu.")