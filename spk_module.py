"""
spk_module.py
==============
Modul Sistem Pendukung Keputusan (SPK):
- AHP sederhana (precomputed pairwise) -> menghasilkan bobot kriteria
- SAW (Simple Additive Weighting) -> menghasilkan ranking alternatif mesin

Kriteria yang dipakai (HARUS selaras dengan fitur penting di SHAP -> Logic Consistency):
1. Risiko Kegagalan (dari Model ML)      -> COST  (semakin kecil semakin baik)
2. Biaya Perbaikan (statis per mesin)    -> COST
3. Efisiensi Operasional (statis)        -> BENEFIT
4. Vibrasi (dari sensor real-time)       -> COST  (selaras dgn temuan SHAP)
"""

import numpy as np
import pandas as pd

# Bobot AHP (hasil precomputed dari pairwise comparison pakar)
# >>> Catatan Logic Consistency: bobot "Risiko" & "Vibrasi" dibuat tinggi
# karena SHAP menunjukkan kedua fitur ini paling berkontribusi pada prediksi.
BOBOT_AHP = {
    "risiko_kegagalan": 0.40,   # paling penting -> selaras dgn SHAP
    "vibrasi":           0.25,  # penting kedua  -> selaras dgn SHAP
    "biaya_perbaikan":   0.20,
    "efisiensi":         0.15,
}

KRITERIA_TIPE = {
    "risiko_kegagalan": "cost",
    "vibrasi": "cost",
    "biaya_perbaikan": "cost",
    "efisiensi": "benefit",
}


def normalisasi_saw(matriks: pd.DataFrame, tipe: dict) -> pd.DataFrame:
    """Normalisasi matriks keputusan sesuai tipe kriteria (cost/benefit)."""
    hasil = matriks.copy().astype(float)
    for kolom in matriks.columns:
        if tipe[kolom] == "benefit":
            hasil[kolom] = matriks[kolom] / matriks[kolom].max()
        else:  # cost
            hasil[kolom] = matriks[kolom].min() / matriks[kolom]
    return hasil


def jalankan_saw(matriks: pd.DataFrame, bobot: dict = BOBOT_AHP,
                  tipe: dict = KRITERIA_TIPE) -> pd.DataFrame:
    """
    Menjalankan algoritma SAW penuh:
    1. Normalisasi matriks
    2. Kalikan dengan bobot AHP
    3. Sum -> skor akhir -> ranking
    """
    matriks_norm = normalisasi_saw(matriks, tipe)
    skor = sum(matriks_norm[k] * bobot[k] for k in bobot)
    hasil = matriks.copy()
    hasil["skor_akhir"] = skor
    hasil["ranking"] = hasil["skor_akhir"].rank(ascending=False).astype(int)
    return hasil.sort_values("ranking")


def buat_matriks_alternatif(prediksi_risiko: float, vibrasi_sekarang: float,
                             daftar_mesin: pd.DataFrame) -> pd.DataFrame:
    """
    Menggabungkan hasil prediksi ML (dinamis, dari slider) dengan
    data statis tiap mesin (biaya & efisiensi) menjadi satu matriks keputusan.

    daftar_mesin harus punya kolom: nama_mesin, biaya_perbaikan, efisiensi
    """
    matriks = daftar_mesin.copy()
    # Setiap alternatif diasumsikan terkena dampak yang sama dari skenario
    # what-if global (suhu/usia dll), tapi punya variasi kecil per mesin
    matriks["risiko_kegagalan"] = prediksi_risiko + matriks["variasi_risiko"]
    matriks["vibrasi"] = vibrasi_sekarang + matriks["variasi_vibrasi"]
    matriks = matriks.drop(columns=["variasi_risiko", "variasi_vibrasi"])
    matriks = matriks.set_index("nama_mesin")
    return matriks
