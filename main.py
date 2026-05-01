"""
processing.py  ─  Pipeline Preprocessing & Feature Engineering
Toko Plastik | XGBoost & TabNet Forecasting App
"""

import pandas as pd
import numpy as np
import streamlit as st

# ─────────────────────────────────────────────────────────────
# KOLOM FITUR & TARGET
# ─────────────────────────────────────────────────────────────
FEATURES = [
    "ID_Barang",
    "Bulan", "Tahun", "Minggu_dalam_Tahun", "Minggu_dalam_Bulan",
    "Lag_1_Minggu", "Lag_2_Minggu", "Lag_3_Minggu", "Lag_4_Minggu",
    "Avg_4_Minggu", "Avg_8_Minggu", "Avg_12_Minggu",
    "Std_4_Minggu", "Std_8_Minggu", "Std_12_Minggu",
    "Max_4_Minggu", "Max_8_Minggu", "Max_12_Minggu",
    "Min_4_Minggu", "Min_8_Minggu", "Min_12_Minggu",
    "Median_4_Minggu", "Median_8_Minggu", "Median_12_Minggu",
]
TARGET = "Qty"


# ─────────────────────────────────────────────────────────────
# STEP 1: CLEANING RAW DATA
# ─────────────────────────────────────────────────────────────
def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop kolom tidak perlu jika ada
    drop_cols = [c for c in ["Id", "Id Transaksi", "Diskon", "SubTotal"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Pastikan kolom wajib ada
    required = {"Nama Barang", "Qty", "Satuan", "Harga", "Total Harga", "Tanggal Transaksi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan di dataset: {missing}")

    # Fix satuan
    df["Satuan"] = df["Satuan"].replace(
        {"ikat": "Ikat", "Ball": "Bal", "Gr": "Gram", "M": "Meter",
         "PACK": "Pack", "pack": "Pack"}
    )

    # Fix satuan kosong untuk produk tertentu
    satuan_fixes = {
        "NUTRIJELL RANDOM": "Sachet",
        "HD KRESEK ECER": "Pcs",
        "JOLLY FACIAL SOFTPACK 250S": "Pack",
        "SARUNG TANGAN KHARISMA": "Pack",
        "Sarung Tangan Kharisma": "Pack",
        "GARPU KUE/BUAH THREE STAR": "Pack",
        "KLIR SQ 120 ML": "Pack",
        "WIPES SANITIZER - SANITER": "Pack",
        "PASEO SMART FACIAL 540 PLY": "Pack",
        "SEAL CUP TOS FRUIT": "Roll",
    }
    for nama, satuan in satuan_fixes.items():
        mask = (df["Nama Barang"] == nama) & (df["Satuan"].isnull())
        df.loc[mask, "Satuan"] = satuan

    # Bersihkan kolom Harga dan Total Harga
    for col in ["Harga", "Total Harga"]:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .str.replace("Rp", "", regex=False)
                .str.replace(".", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Parse tanggal — robust multi-format
    tgl_raw = df["Tanggal Transaksi"].astype(str).str.strip()

    def _parse_tgl(series):
        for fmt in ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y",
                    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
                    "%d-%m-%Y %H:%M:%S", "%d-%m-%Y"]:
            try:
                parsed = pd.to_datetime(series, format=fmt, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    return parsed
            except Exception:
                continue
        return pd.to_datetime(series, dayfirst=True, errors="coerce")

    df["Tanggal Transaksi"] = _parse_tgl(tgl_raw)

    n_nat = df["Tanggal Transaksi"].isna().sum()
    if n_nat > 0:
        import warnings
        warnings.warn(f"{n_nat} baris memiliki format tanggal tidak dikenali dan dibuang.")

    df = df.dropna(subset=["Tanggal Transaksi"])

    if df.empty:
        raise ValueError(
            "Tidak ada baris valid setelah parsing tanggal. "
            "Pastikan kolom Tanggal Transaksi berformat dd/mm/yyyy atau yyyy-mm-dd."
        )

    df["Tanggal Transaksi"] = df["Tanggal Transaksi"].dt.normalize()  # buang jam

    # Parse Qty
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0).astype(float)

    # Penyesuaian unit khusus
    # SAGU MUTIARA: ×0.25
    df.loc[df["Nama Barang"] == "SAGU MUTIARA", "Qty"] *= 0.25

    # BESEK - 1 KODI (satuan kodi) → /18, rename ke BESEK
    mask_kodi = (df["Nama Barang"] == "BESEK - 1 KODI") & (df["Satuan"].str.lower() == "kodi")
    df.loc[mask_kodi, "Qty"] /= 18
    df.loc[df["Nama Barang"] == "BESEK - 1 KODI", "Satuan"] = "Bal"
    df.loc[df["Nama Barang"] == "BESEK - 1 KODI", "Nama Barang"] = "BESEK"

    # CREAMER NDC DONGXIAO
    df.loc[df["Nama Barang"] == "CREAMER NDC DONGXIAO 250 GR", "Qty"] *= 0.25
    df.loc[df["Nama Barang"] == "CREAMER NDC DONGXIAO 500 GR", "Qty"] *= 0.5
    for variant in ["CREAMER NDC DONGXIAO 250 GR", "CREAMER NDC DONGXIAO 500 GR"]:
        df.loc[df["Nama Barang"] == variant, "Satuan"] = "Kg"
    df.loc[
        df["Nama Barang"].isin([
            "CREAMER NDC DONGXIAO 250 GR",
            "CREAMER NDC DONGXIAO 500 GR",
            "CREAMER NDC DONGXIAO 1 KG",
        ]),
        "Nama Barang",
    ] = "CREAMER NDC DONGXIAO"

    # TOPLES TABUNG 1000 ML - Tebal (harga 224500) → ×60, satuan Pcs
    mask_t1 = (df["Nama Barang"] == "TOPLES TABUNG 1000 ML - Tebal") & (df["Harga"] == 224500)
    df.loc[mask_t1, "Qty"] *= 60
    df.loc[mask_t1, "Satuan"] = "Pcs"

    return df


# ─────────────────────────────────────────────────────────────
# STEP 2: ANALISIS ABC
# ─────────────────────────────────────────────────────────────
def analisis_abc(df: pd.DataFrame):
    """
    Menghitung klasifikasi ABC berdasarkan Total Harga per produk.
    Returns: (df_abc, grup_a, grup_b, grup_c, ringkasan)
    """
    df_produk = (
        df.groupby("Nama Barang")["Total Harga"]
        .sum()
        .reset_index()
        .rename(columns={"Total Harga": "total_harga"})
        .sort_values("total_harga", ascending=False)
        .reset_index(drop=True)
    )

    total = df_produk["total_harga"].sum()
    df_produk["persen"] = df_produk["total_harga"] / total * 100
    df_produk["kumulatif"] = df_produk["persen"].cumsum()

    def assign_abc(row):
        if row["kumulatif"] <= 80:
            return "A"
        elif row["kumulatif"] <= 95:
            return "B"
        else:
            return "C"

    df_produk["Grup"] = df_produk.apply(assign_abc, axis=1)

    grup_a = df_produk[df_produk["Grup"] == "A"][["Nama Barang", "total_harga"]].reset_index(drop=True)
    grup_b = df_produk[df_produk["Grup"] == "B"][["Nama Barang", "total_harga"]].reset_index(drop=True)
    grup_c = df_produk[df_produk["Grup"] == "C"][["Nama Barang", "total_harga"]].reset_index(drop=True)

    n_total = len(df_produk)
    ringkasan = {
        "n_produk": n_total,
        "n_a": len(grup_a),
        "n_b": len(grup_b),
        "n_c": len(grup_c),
        "persen_produk": {
            "A": len(grup_a) / n_total * 100,
            "B": len(grup_b) / n_total * 100,
            "C": len(grup_c) / n_total * 100,
        },
        "persen_pendapatan": {
            "A": grup_a["total_harga"].sum() / total * 100,
            "B": grup_b["total_harga"].sum() / total * 100,
            "C": grup_c["total_harga"].sum() / total * 100,
        },
    }

    return df_produk, grup_a, grup_b, grup_c, ringkasan


# ─────────────────────────────────────────────────────────────
# STEP 3: BUILD DAILY → WEEKLY TIME SERIES (SEMUA PRODUK)
# ─────────────────────────────────────────────────────────────
def build_weekly_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dari data transaksi mentah (sudah di-clean), bangun time series mingguan
    dengan full cross-join produk × minggu (zero-filled).
    """
    df_daily = (
        df.groupby(["Nama Barang", "Satuan", "Tanggal Transaksi"], as_index=False)
        .agg({"Qty": "sum"})
    )
    df_daily["Tanggal Transaksi"] = pd.to_datetime(df_daily["Tanggal Transaksi"])

    tgl_range = pd.date_range(
        start=df_daily["Tanggal Transaksi"].min(),
        end=df_daily["Tanggal Transaksi"].max(),
        freq="D",
    )

    produk_list = df_daily[["Nama Barang", "Satuan"]].drop_duplicates()
    full_index = (
        produk_list.assign(key=1)
        .merge(pd.DataFrame({"Tanggal Transaksi": tgl_range, "key": 1}), on="key")
        .drop("key", axis=1)
    )

    df_daily_full = full_index.merge(
        df_daily, on=["Nama Barang", "Satuan", "Tanggal Transaksi"], how="left"
    )
    df_daily_full["Qty"] = df_daily_full["Qty"].fillna(0)
    df_daily_full = df_daily_full.sort_values(["Tanggal Transaksi", "Nama Barang"]).reset_index(drop=True)

    # Resample ke mingguan (W = akhir minggu Minggu)
    df_weekly = (
        df_daily_full[["Nama Barang", "Satuan", "Tanggal Transaksi", "Qty"]]
        .set_index("Tanggal Transaksi")
        .groupby(["Nama Barang", "Satuan"])["Qty"]
        .resample("W")
        .sum()
        .reset_index()
    )

    return df_weekly


# ─────────────────────────────────────────────────────────────
# STEP 4: ANALISIS ZERO PERCENTAGE
# ─────────────────────────────────────────────────────────────
def hitung_zero_percentage(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung persentase minggu dengan Qty == 0 untuk setiap produk.
    Returns DataFrame dengan kolom: Nama Barang, Total_Weeks, Zero_Count, Zero_Percentage
    """
    zero_count = (
        df_weekly[df_weekly["Qty"] == 0]
        .groupby("Nama Barang")
        .size()
        .reset_index(name="Zero_Count")
    )
    total_count = (
        df_weekly.groupby("Nama Barang")
        .size()
        .reset_index(name="Total_Weeks")
    )
    df_zero = total_count.merge(zero_count, on="Nama Barang", how="left")
    df_zero["Zero_Count"] = df_zero["Zero_Count"].fillna(0)
    df_zero["Zero_Percentage"] = (df_zero["Zero_Count"] / df_zero["Total_Weeks"]) * 100
    df_zero = df_zero.sort_values("Zero_Percentage", ascending=True).reset_index(drop=True)
    return df_zero


# ─────────────────────────────────────────────────────────────
# STEP 5: FILTER PRODUK (ABC-A + threshold zero%)
# ─────────────────────────────────────────────────────────────
def filter_produk_peramalan(
    df_weekly: pd.DataFrame,
    produk_grup_a: list,
    zero_threshold: float,
    df_zero: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter df_weekly hanya untuk produk Grup A yang memenuhi threshold zero%.
    zero_threshold: nilai max zero percentage yang diperbolehkan (misal 8 berarti ≤8%)
    Jika 100, semua produk A diramal.
    """
    # Produk dengan zero% ≤ threshold
    if zero_threshold >= 100:
        valid = set(produk_grup_a)
    else:
        valid_by_zero = set(
            df_zero[df_zero["Zero_Percentage"] <= zero_threshold]["Nama Barang"].tolist()
        )
        valid = set(produk_grup_a) & valid_by_zero

    df_filtered = df_weekly[df_weekly["Nama Barang"].isin(valid)].copy()
    return df_filtered


# ─────────────────────────────────────────────────────────────
# STEP 6: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def buat_fitur(df_weekly_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Buat fitur temporal dan lag/rolling dari data mingguan yang sudah difilter.
    """
    df = df_weekly_filtered.copy()
    df["Tanggal Transaksi"] = pd.to_datetime(df["Tanggal Transaksi"])
    df = df.sort_values(["Nama Barang", "Tanggal Transaksi"]).reset_index(drop=True)

    # Encode produk
    produk_unik = sorted(df["Nama Barang"].unique())
    produk_map = {p: i for i, p in enumerate(produk_unik)}
    df["ID_Barang"] = df["Nama Barang"].map(produk_map)

    # Fitur waktu
    df["Bulan"] = df["Tanggal Transaksi"].dt.month
    df["Tahun"] = df["Tanggal Transaksi"].dt.year
    df["Minggu_dalam_Tahun"] = df["Tanggal Transaksi"].dt.isocalendar().week.astype(int)
    df["Minggu_dalam_Bulan"] = (df["Tanggal Transaksi"].dt.day - 1) // 7 + 1

    # Lag & Rolling per produk
    def rolling_per_produk(grp):
        grp = grp.sort_values("Tanggal Transaksi").copy()
        qty = grp["Qty"]

        # Lags
        grp["Lag_1_Minggu"] = qty.shift(1)
        grp["Lag_2_Minggu"] = qty.shift(2)
        grp["Lag_3_Minggu"] = qty.shift(3)
        grp["Lag_4_Minggu"] = qty.shift(4)

        # Rolling windows (min_periods=1)
        for w in [4, 8, 12]:
            shifted = qty.shift(1)
            grp[f"Avg_{w}_Minggu"]    = shifted.rolling(w, min_periods=1).mean()
            grp[f"Std_{w}_Minggu"]    = shifted.rolling(w, min_periods=1).std().fillna(0)
            grp[f"Max_{w}_Minggu"]    = shifted.rolling(w, min_periods=1).max()
            grp[f"Min_{w}_Minggu"]    = shifted.rolling(w, min_periods=1).min()
            grp[f"Median_{w}_Minggu"] = shifted.rolling(w, min_periods=1).median()

        return grp

    df = df.groupby("Nama Barang", group_keys=False).apply(rolling_per_produk)

    # Fill NaN lag awal dengan 0
    lag_cols = [c for c in df.columns if c.startswith(("Lag_", "Avg_", "Std_", "Max_", "Min_", "Median_"))]
    df[lag_cols] = df[lag_cols].fillna(0)

    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────
# STEP 7: TRAIN/VAL SPLIT
# ─────────────────────────────────────────────────────────────
def split_train_val(df: pd.DataFrame, test_size: float = 0.20):
    """
    Time-based split. Urutan baris diasumsikan sudah kronologis.
    """
    df = df.sort_values(["Tanggal Transaksi", "Nama Barang"]).reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - test_size))

    train = df.iloc[:split_idx]
    val   = df.iloc[split_idx:]

    available_features = [f for f in FEATURES if f in df.columns]
    X_train = train[available_features]
    y_train = train[TARGET]
    X_val   = val[available_features]
    y_val   = val[TARGET]

    return X_train, y_train, X_val, y_val


def hitung_batas_train(df: pd.DataFrame, test_size: float = 0.20) -> pd.Timestamp:
    df = df.sort_values(["Tanggal Transaksi", "Nama Barang"]).reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - test_size))
    return pd.to_datetime(df.iloc[split_idx - 1]["Tanggal Transaksi"])


# ─────────────────────────────────────────────────────────────
# STEP 8: HITUNG MINGGU PREDIKSI
# ─────────────────────────────────────────────────────────────
def hitung_minggu_prediksi(df: pd.DataFrame, n_minggu: int = 4):
    """
    Hitung rentang (Senin–Minggu) untuk n_minggu minggu ke depan setelah data terakhir.
    Returns list of (senin, minggu) tuple.

    Robust terhadap:
    - Kolom Tanggal Transaksi yang belum datetime
    - Nilai NaT / kosong setelah parsing
    - DataFrame kosong
    """
    if df is None or df.empty:
        raise ValueError("DataFrame kosong — tidak bisa menentukan rentang prediksi.")

    # Pastikan kolom ada
    if "Tanggal Transaksi" not in df.columns:
        raise ValueError("Kolom 'Tanggal Transaksi' tidak ditemukan di DataFrame.")

    # Konversi ke datetime, paksa error → NaT
    tgl_series = pd.to_datetime(df["Tanggal Transaksi"], errors="coerce")

    # Buang NaT sebelum ambil max
    tgl_valid = tgl_series.dropna()
    if tgl_valid.empty:
        raise ValueError(
            "Kolom 'Tanggal Transaksi' tidak memiliki nilai valid (semua NaT). "
            "Periksa format tanggal di data Anda."
        )

    tgl_max = tgl_valid.max()

    # Logika Senin pertama setelah tgl_max
    # weekday(): 0=Senin, 6=Minggu
    wd = tgl_max.weekday()
    if wd == 6:
        # Sudah hari Minggu → Senin berikutnya adalah +1 hari
        days_ahead = 1
    else:
        # Maju ke Senin pertama di minggu berikutnya
        days_ahead = 7 - wd

    senin_pertama = tgl_max + pd.Timedelta(days=days_ahead)
    # Normalisasi ke tengah malam (hilangkan komponen jam jika ada)
    senin_pertama = senin_pertama.normalize()

    rentang = []
    for i in range(n_minggu):
        s = senin_pertama + pd.Timedelta(weeks=i)
        e = s + pd.Timedelta(days=6)
        rentang.append((s, e))

    return rentang


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run_processing(raw_df: pd.DataFrame):
    """
    Jalankan full pipeline: clean → ABC → weekly.
    Returns: (df_weekly_all, grup_a, grup_b, grup_c, ringkasan, df_zero_all)
    """
    try:
        # 1. Clean
        df_clean = clean_raw(raw_df)

        # 2. ABC Analysis
        _, grup_a, grup_b, grup_c, ringkasan = analisis_abc(df_clean)

        # 3. Build weekly (SEMUA produk, belum difilter)
        df_weekly_all = build_weekly_full(df_clean)

        # 4. Zero percentage (dari semua produk)
        df_zero_all = hitung_zero_percentage(df_weekly_all)

        return df_weekly_all, grup_a, grup_b, grup_c, ringkasan, df_zero_all

    except Exception as e:
        st.error(f"❌ Error saat preprocessing: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None, None, None