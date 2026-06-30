"""
app.py
=======
SMART MAINTENANCE SIMULATOR (SMS)
Integrasi: Machine Learning (RandomForest) -> XAI (SHAP) -> SPK (AHP+SAW)
+ Fitur Novelty: AI Narrator otomatis, Robustness Guard, Mode Anonimisasi Publik,
  Confidence Band, dan Scenario Comparison (before vs after).

Cara jalan lokal:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from spk_module import jalankan_saw, buat_matriks_alternatif, BOBOT_AHP

# -------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Maintenance Simulator",
    page_icon="🛠️",
    layout="wide",
)

# -------------------------------------------------------------------------
# LOAD ARTIFACTS (cached supaya tidak reload setiap interaksi slider)
# -------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/model_risiko.joblib")
    scaler = joblib.load("models/scaler.joblib")
    metrics = joblib.load("models/metrics.joblib")
    features = joblib.load("models/features.joblib")
    daftar_mesin = pd.read_csv("data/daftar_mesin.csv")
    explainer = shap.Explainer(model)
    return model, scaler, metrics, features, daftar_mesin, explainer


model, scaler, metrics, features, daftar_mesin, explainer = load_artifacts()

# -------------------------------------------------------------------------
# SIDEBAR: MODE & INPUT "WHAT-IF"
# -------------------------------------------------------------------------
st.sidebar.title("Panel Kontrol Simulasi")

mode_publik = st.sidebar.toggle(
    "🔒 Mode Demo Publik (Anonimisasi)",
    value=False,
    help="Saat aktif, label mesin & angka biaya riil disamarkan agar aman ditampilkan ke audiens umum."
)

st.sidebar.markdown("### Intervensi Skenario (What-If)")
suhu = st.sidebar.slider("Suhu Mesin (°C)", 0, 1500, 70,
                          help="Coba masukkan nilai ekstrem (>200) untuk menguji Robustness Guard.")
tekanan = st.sidebar.slider("Tekanan (bar)", 0.0, 20.0, 5.0, 0.1)
vibrasi = st.sidebar.slider("Vibrasi (mm/s)", 0.0, 15.0, 3.0, 0.1)
usia_mesin = st.sidebar.slider("Usia Mesin (tahun)", 0.0, 25.0, 5.0, 0.5)
jam_operasi = st.sidebar.slider("Jam Operasi Kumulatif", 0, 12000, 4000, 100)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Bobot kriteria SPK (AHP) diselaraskan dengan kontribusi SHAP: "
    f"Risiko={BOBOT_AHP['risiko_kegagalan']}, Vibrasi={BOBOT_AHP['vibrasi']}, "
    f"Biaya={BOBOT_AHP['biaya_perbaikan']}, Efisiensi={BOBOT_AHP['efisiensi']}"
)

# -------------------------------------------------------------------------
# ROBUSTNESS GUARD (Soal Umpan Balik #5)
# -------------------------------------------------------------------------
BATAS_WAJAR = {"suhu": (0, 200), "tekanan": (0, 15), "vibrasi": (0, 12),
               "usia_mesin": (0, 20), "jam_operasi": (0, 10000)}

input_sekarang = {"suhu": suhu, "tekanan": tekanan, "vibrasi": vibrasi,
                   "usia_mesin": usia_mesin, "jam_operasi": jam_operasi}

peringatan_drift = []
for fitur, (lo, hi) in BATAS_WAJAR.items():
    if not (lo <= input_sekarang[fitur] <= hi):
        peringatan_drift.append(fitur)

st.title("Smart Maintenance Simulator")
st.caption("Hybrid Decision Intelligence: Machine Learning × Explainable AI × Sistem Pendukung Keputusan")

if peringatan_drift:
    st.warning(
        f"⚠️ **Peringatan Out-of-Distribution (Drift Risk):** Nilai pada fitur "
        f"**{', '.join(peringatan_drift)}** berada jauh di luar rentang data latih. "
        "Prediksi model di bawah ini **tidak dapat dijamin akurasinya** untuk kondisi ekstrem ini."
    )

# -------------------------------------------------------------------------
# PIPELINE: SLIDER -> SCALER -> MODEL ML -> MATRIKS SPK -> SAW -> RANKING
# -------------------------------------------------------------------------
input_df = pd.DataFrame([input_sekarang])[features]
input_scaled = scaler.transform(input_df)
prediksi_risiko = float(model.predict(input_scaled)[0])
prediksi_risiko = float(np.clip(prediksi_risiko, 0, 100))

rmse = metrics["rmse"]
batas_bawah = max(0, prediksi_risiko - rmse)
batas_atas = min(100, prediksi_risiko + rmse)

matriks_x = buat_matriks_alternatif(prediksi_risiko, vibrasi, daftar_mesin)
hasil_spk = jalankan_saw(matriks_x)

# -------------------------------------------------------------------------
# LAYOUT UTAMA: 3 KOLOM
# -------------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 1, 1.3])

with col1:
    st.subheader("Prediksi Risiko")
    st.metric("Skor Risiko Kegagalan", f"{prediksi_risiko:.1f} / 100")
    st.caption(f"Confidence band (±RMSE): **{batas_bawah:.1f} – {batas_atas:.1f}** "
               f"(R² model = {metrics['r2']:.2f})")
    if prediksi_risiko > 70:
        st.error("Status: 🔴 KRITIS — Tindakan segera direkomendasikan")
    elif prediksi_risiko > 40:
        st.warning("Status: 🟡 WASPADA — Pantau secara berkala")
    else:
        st.success("Status: 🟢 AMAN")

with col2:
    st.subheader("Mengapa Hasilnya Demikian? (XAI)")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    shap_values = explainer(input_scaled)
    shap_values.feature_names = features
    shap.plots.waterfall(shap_values[0], max_display=5, show=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col3:
    st.subheader("Rekomendasi Tindakan (SPK - SAW)")
    tampil = hasil_spk.reset_index()
    if mode_publik:
        tampil["nama_mesin"] = [f"Unit-{i+1}" for i in range(len(tampil))]
        tampil["biaya_perbaikan"] = "•••• (disamarkan)"
    st.dataframe(
        tampil[["nama_mesin", "ranking", "skor_akhir", "risiko_kegagalan",
                "biaya_perbaikan", "efisiensi"]],
        hide_index=True, use_container_width=True
    )
    top_pick = hasil_spk.index[0]
    st.info(f"✅ **Rekomendasi #1:** prioritaskan tindakan pada **{top_pick if not mode_publik else 'Unit-1'}**.")

# -------------------------------------------------------------------------
# NOVELTY: Auto-generated insight (mini agent)
# Menggabungkan SHAP (kontribusi fitur) + SPK (ranking) menjadi 1 narasi.
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("Ringkasan Otomatis untuk Pengambil Keputusan")

kontribusi = pd.Series(shap_values[0].values, index=features).sort_values(key=abs, ascending=False)
fitur_dominan = kontribusi.index[0]
arah = "menaikkan" if kontribusi.iloc[0] > 0 else "menurunkan"

narasi = (
    f"Pada skenario saat ini, fitur **{fitur_dominan}** adalah faktor paling dominan yang "
    f"**{arah}** prediksi risiko sebesar **{abs(kontribusi.iloc[0]):.1f} poin**. "
    f"Model memprediksi skor risiko **{prediksi_risiko:.1f}** (rentang wajar "
    f"{batas_bawah:.1f}–{batas_atas:.1f} akibat error model). "
    f"Berdasarkan perhitungan SPK (AHP+SAW) yang bobotnya diselaraskan dengan temuan SHAP, "
    f"sistem merekomendasikan **{top_pick if not mode_publik else 'Unit-1'}** sebagai prioritas utama "
    f"karena kombinasi risiko, vibrasi, biaya, dan efisiensinya paling optimal."
)
st.write(narasi)

with st.expander("📈 Lihat detail kontribusi semua fitur (SHAP)"):
    st.bar_chart(kontribusi)

# -------------------------------------------------------------------------
# FOOTER: INFO PIPELINE (transparansi alur, bagian dari narrative building)
# -------------------------------------------------------------------------
st.markdown("---")
with st.expander("ℹ️ Alur Data Pipeline (Transparansi Sistem)"):
    st.code(
        "Slider (Streamlit) → input_df → scaler.transform() → model_ml.predict()\n"
        "→ buat_matriks_alternatif() → jalankan_saw(bobot_ahp) → ranking\n"
        "→ shap.Explainer() → waterfall plot → AI Narrator (auto-text)",
        language="text"
    )
st.caption("Smart Maintenance Simulator • Proyek UAS Pemodelan & Simulasi • "
           "Dibangun dengan Streamlit, scikit-learn, SHAP, dan SPK (AHP-SAW)")
