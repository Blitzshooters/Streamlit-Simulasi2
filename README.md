# 🛠️ Smart Maintenance Simulator (SMS)
### Proyek UAS — Praktikum Pemodelan & Simulasi (Minggu 16: Integrasi Akhir)

Simulator hybrid yang mengintegrasikan **Machine Learning (prediksi risiko kegagalan mesin)**,
**Explainable AI / SHAP (transparansi)**, dan **Sistem Pendukung Keputusan AHP+SAW (rekomendasi
tindakan)** ke dalam satu dashboard Streamlit interaktif.

## ✨ Unsur Novelty
| Fitur | Deskripsi |
|---|---|
| 🤖 **AI Narrator** | Menghasilkan narasi insight otomatis dari gabungan SHAP + ranking SPK — tidak perlu menulis interpretasi manual setiap demo. |
| 🔬 **Sensitivity Sweep** | (di notebook) Menyapu nilai sensor secara otomatis untuk menemukan *tipping point* kapan ranking SPK berubah. |
| 🛡️ **Robustness Guard** | Mendeteksi input out-of-distribution (mis. suhu 1000°C) dan memberi peringatan drift secara eksplisit, bukan diam-diam memberi prediksi salah. |
| 🔒 **Mode Demo Publik** | Toggle anonimisasi nama mesin & biaya agar aman ditampilkan di hadapan audiens umum (etika & privasi data). |
| 📈 **Confidence Band** | Skor risiko ditampilkan dengan rentang ±RMSE, bukan angka tunggal yang menyesatkan presisi. |

## 📂 Struktur Proyek
```
project/
├── app.py                     # Dashboard utama Streamlit
├── spk_module.py              # Modul SPK (AHP bobot + SAW ranking)
├── train_model.py             # Generate dataset sintetis + training model ML
├── runtime.txt                # Mengunci versi Python untuk Streamlit Cloud (lihat Troubleshooting)
├── requirements.txt           # Daftar dependency dengan versi spesifik (Replayability)
├── README.md
├── models/
│   ├── model_risiko.joblib    # Model RandomForestRegressor terlatih
│   ├── scaler.joblib          # StandardScaler
│   ├── metrics.joblib         # RMSE & R2 (untuk confidence band)
│   └── features.joblib        # Daftar nama fitur
├── data/
│   ├── sensor_data.csv        # Dataset sintetis sensor mesin
│   └── daftar_mesin.csv       # Data statis 3 mesin alternatif (untuk SPK)
└── notebook/
    └── Langkah1_2_3_Integrasi_dan_WhatIf.ipynb   # Notebook lengkap Langkah 1 (Sinkronisasi ML↔SPK),
                                                   # Langkah 2 (Integrasi XAI/SHAP), dan Langkah 3 (What-If Demo Prep)
```

## 🚀 Menjalankan Secara Lokal
```bash
# 1. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate      # Windows: venv\\Scripts\\activate

# 2. Install dependency dengan versi yang terkunci
pip install -r requirements.txt

# 3. (Hanya sekali / jika model belum ada) Generate dataset & latih model
python train_model.py

# 4. Jalankan dashboard
streamlit run app.py
```

## 🐛 Troubleshooting: Build Gagal di Streamlit Cloud (Pillow/zlib error)

Jika log deploy menunjukkan error seperti:
```
Failed to download and build `pillow==10.4.0`
RequiredDependencyException: zlib
```

**Penyebab**: Streamlit Cloud kadang menjalankan Python versi sangat baru (mis. 3.14) yang
belum punya *wheel* (binary) precompiled untuk `pillow`/`pandas` versi lama yang dikunci di
`requirements.txt`. Akibatnya pip mencoba build dari source dan gagal karena header sistem
(`zlib`) tidak tersedia di environment cloud.

**Solusi**: file `runtime.txt` di root project ini sudah mengunci Python ke versi **3.11**
(versi stabil yang seluruh dependency-nya tersedia dalam bentuk wheel siap pakai). Pastikan
file ini ikut ter-commit ke repository, lalu lakukan **Reboot app** / redeploy dari menu
Streamlit Cloud agar environment dibangun ulang dengan Python 3.11.

## ☁️ Deploy ke Streamlit Community Cloud
1. Push seluruh folder `project/` ke repository GitHub (publik atau private).
2. Pastikan `models/*.joblib` dan `data/*.csv` **ikut di-commit** (jangan di-`.gitignore`),
   karena `app.py` memuatnya langsung tanpa melatih ulang di cloud.
3. Buka [share.streamlit.io](https://share.streamlit.io), hubungkan repo, lalu set:
   - **Main file path**: `app.py`
   - Python akan otomatis membaca `requirements.txt`.
4. Klik **Deploy**. Tunggu beberapa menit hingga build selesai.

> 💡 Tip Replayability: versi library di `requirements.txt` dikunci spesifik (`==`) agar
> tidak ada fungsi *deprecated* yang tiba-tiba berubah perilaku saat dijalankan ulang
> di lingkungan cloud yang berbeda dari laptop pengembang.

## 🎤 Skenario Demo yang Disarankan (Live Interaction)
1. **Context**: "Mesin pabrik kami sering gagal tanpa peringatan — bisakah kita memprediksi & memprioritaskan perawatan?"
2. Geser slider **Vibrasi** dan **Suhu** ke nilai tinggi → tunjukkan skor risiko naik real-time.
3. Buka panel **SHAP waterfall** → jelaskan fitur mana yang paling bertanggung jawab.
4. Tunjukkan tabel **SPK Ranking** bergeser dari Mesin A ke Mesin B/C.
5. Baca paragraf **AI Narrator** sebagai kesimpulan otomatis — efektif untuk closing statement ke penguji.
6. (Opsional, uji Robustness) Naikkan slider Suhu ke >1000 → tunjukkan banner peringatan drift muncul.

## 🧠 Logic Consistency (Kesesuaian SHAP ↔ Bobot AHP)
Bobot kriteria di `spk_module.py` (`BOBOT_AHP`) sengaja diberi nilai tertinggi pada
**Risiko Kegagalan** dan **Vibrasi** — dua fitur yang juga terbukti paling dominan
pada visualisasi SHAP. Ini membuktikan prinsip *Logic Consistency* pada Bab XII:
fitur yang penting menurut model prediktif juga harus penting menurut bobot SPK.
