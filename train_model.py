"""
train_model.py
================
Membangun dataset sintetis sensor mesin pabrik, melatih model Machine Learning
untuk memprediksi RISIKO KEGAGALAN MESIN, lalu menyimpan model + scaler.

Proyek: Smart Maintenance Simulator (SMS) - Hybrid ML + XAI + SPK
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

np.random.seed(42)

# -----------------------------------------------------------------
# 1. GENERATE SYNTHETIC DATA
# -----------------------------------------------------------------
N = 1200

suhu       = np.random.normal(70, 15, N).clip(20, 130)      # derajat C
tekanan    = np.random.normal(5, 1.5, N).clip(0.5, 12)       # bar
vibrasi    = np.random.normal(3, 1.2, N).clip(0.1, 9)         # mm/s
usia_mesin = np.random.uniform(0, 15, N)                      # tahun
jam_operasi= np.random.uniform(500, 9000, N)                  # jam

# Risiko kegagalan (skor 0-100) -> dipengaruhi non-linear oleh fitur di atas
risiko = (
    0.35 * (suhu - 60) +
    4.5  * (vibrasi) +
    3.0  * (tekanan - 4) +
    1.8  * usia_mesin +
    0.004* jam_operasi +
    np.random.normal(0, 5, N)
)
risiko = (risiko - risiko.min()) / (risiko.max() - risiko.min()) * 100
risiko = risiko.clip(0, 100)

df = pd.DataFrame({
    "suhu": suhu,
    "tekanan": tekanan,
    "vibrasi": vibrasi,
    "usia_mesin": usia_mesin,
    "jam_operasi": jam_operasi,
    "risiko_kegagalan": risiko
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/sensor_data.csv", index=False)
print("Dataset sintetis tersimpan di data/sensor_data.csv")

# -----------------------------------------------------------------
# 2. PREPROCESSING
# -----------------------------------------------------------------
features = ["suhu", "tekanan", "vibrasi", "usia_mesin", "jam_operasi"]
X = df[features]
y = df["risiko_kegagalan"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------
# 3. TRAIN MODEL
# -----------------------------------------------------------------
model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)
print(f"RMSE: {rmse:.2f} | R2: {r2:.3f}")

# Simpan informasi error untuk ditampilkan di dashboard (Confidence Interval)
metrics = {"rmse": rmse, "r2": r2}

# -----------------------------------------------------------------
# 4. SAVE ARTIFACTS
# -----------------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model_risiko.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(metrics, "models/metrics.joblib")
joblib.dump(features, "models/features.joblib")

print("Model, scaler, metrics, dan daftar fitur berhasil disimpan di folder models/")
