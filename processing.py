"""
processing.py  ─  Pipeline Preprocessing & Feature Engineering
Toko Plastik | XGBoost & TabNet Forecasting App

Alur sesuai notebook prepro.ipynb:
  Cell 2-17  → clean_raw()           : fix satuan null, drop kolom, parse angka/tanggal, koreksi qty manual
  Cell 18-24 → analisis_abc()        : groupby Nama Barang+Satuan, ABC 20/30/50
  Cell 25    → filter ke Grup A      : di run_processing()
  Cell 26-28 → konversi_satuan()     : merge master_konversi, Qty×Multiplier, update satuan
  Cell 30-33 → build_weekly_full()   : daily full-index → resample W
  Cell 34-36 → hitung_zero_pct()     : collapse satuan, n_minggu_unik, zero%
  Cell 37    → filter_produk()       : filter zero% ≤ threshold
  Cell 39-49 → buat_fitur()          : ID_Barang, temporal, lag, rolling, fillna(0)
  Cell 53-54 → FEATURES, split_train_val()
"""

import pandas as pd
import numpy as np
import streamlit as st

# ─────────────────────────────────────────────────────────────
# FEATURES & TARGET  (Cell 53)
# ─────────────────────────────────────────────────────────────
FEATURES = [
    "Tahun",
    "Bulan",
    "Minggu_dalam_Bulan",
    "Minggu_dalam_Tahun",

    "Lag_1_Minggu",
    "Lag_2_Minggu",
    "Lag_3_Minggu",
    "Lag_4_Minggu",
    "Lag_8_Minggu",
    "Lag_12_Minggu",

    "Avg_4_Minggu",
    "Avg_8_Minggu",
    "Avg_12_Minggu",

    "Std_4_Minggu",
    "Std_8_Minggu",
    "Std_12_Minggu",

    "Max_4_Minggu",
    "Max_8_Minggu",
    "Max_12_Minggu",

    "Min_4_Minggu",
    "Min_8_Minggu",
    "Min_12_Minggu",

    "Median_4_Minggu",
    "Median_8_Minggu",
    "Median_12_Minggu",
]
TARGET = "Qty"


# ─────────────────────────────────────────────────────────────
# HELPER: bersihkan kolom angka (PyArrow-safe)
# ─────────────────────────────────────────────────────────────
def _bersihkan_angka(series: pd.Series) -> pd.Series:
    """Hapus 'Rp', titik ribuan. Kompatibel dengan PyArrow dtype (Python 3.14+)."""
    s = series.astype(str)
    s = s.str.replace("Rp", "", regex=False)
    s = s.str.replace(".", "", regex=False)
    s = s.str.replace(",", "", regex=False)
    s = s.str.strip()
    return pd.to_numeric(s, errors="coerce").fillna(0)


# ─────────────────────────────────────────────────────────────
# HELPER: parse tanggal multi-format
# ─────────────────────────────────────────────────────────────
def _parse_tgl(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    for fmt in [
        "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
        "%d-%m-%Y %H:%M:%S", "%d-%m-%Y",
    ]:
        try:
            parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
            if parsed.notna().mean() > 0.8:
                return parsed
        except Exception:
            continue
    return pd.to_datetime(raw, dayfirst=True, errors="coerce")


# ─────────────────────────────────────────────────────────────
# STEP 1: CLEAN RAW  (Cell 2–17)
# ─────────────────────────────────────────────────────────────
def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan data mentah persis seperti Cell 2–17 notebook:
    - Fix satuan null per produk
    - Drop kolom tidak perlu
    - Normalisasi satuan (typo)
    - Parse Harga, Total Harga, Qty, Tanggal
    - Koreksi Qty manual (SAGU MUTIARA, BESEK, CREAMER, TOPLES, HD LOS BENING)
    """
    df = df.copy()

    # ── Validasi kolom wajib ──────────────────────────────────
    required = {"Nama Barang", "Qty", "Satuan", "Harga", "Total Harga", "Tanggal Transaksi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    # ── Cell 2: fix satuan null per produk ───────────────────
    SATUAN_NULL_FIX = {
        "NUTRIJELL RANDOM":         "Sachet",
        "HD KRESEK ECER":           "Pcs",
        "JOLLY FACIAL SOFTPACK 250S": "Pack",
        "SARUNG TANGAN KHARISMA":   "Pack",
        "Sarung Tangan Kharisma":   "Pack",
        "GARPU KUE/BUAH THREE STAR": "Pack",
        "KLIR SQ 120 ML":           "Pack",
        "WIPES SANITIZER - SANITER": "Pack",
        "PASEO SMART FACIAL 540 PLY": "Pack",
        "SEAL CUP TOS FRUIT":       "Roll",
    }
    for nama, satuan in SATUAN_NULL_FIX.items():
        mask = (df["Nama Barang"] == nama) & (df["Satuan"].isnull())
        df.loc[mask, "Satuan"] = satuan

    # ── Cell 3: drop kolom tidak perlu ───────────────────────
    drop_cols = [c for c in ["Id", "Id Transaksi", "Diskon", "SubTotal"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # ── Cell 4: normalisasi typo satuan ──────────────────────
    df["Satuan"] = df["Satuan"].replace({
        "ikat": "Ikat", "Ball": "Bal", "Gr": "Gram",
        "M": "Meter", "PACK": "Pack", "pack": "Pack",
    })

    # ── Cell 5–6: parse Harga, Total Harga ───────────────────
    for col in ["Harga", "Total Harga"]:
        df[col] = _bersihkan_angka(df[col]).astype("int64")

    # ── Cell 6: parse Qty & Tanggal ──────────────────────────
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0).astype(float)

    df["Tanggal Transaksi"] = _parse_tgl(df["Tanggal Transaksi"])
    n_nat = df["Tanggal Transaksi"].isna().sum()
    if n_nat > 0:
        import warnings
        warnings.warn(f"{n_nat} baris tanggal tidak dikenali dan dibuang.")
    df = df.dropna(subset=["Tanggal Transaksi"])
    if df.empty:
        raise ValueError("Tidak ada baris valid setelah parsing tanggal.")
    df["Tanggal Transaksi"] = df["Tanggal Transaksi"].dt.normalize()

    # ── Cell 7: SAGU MUTIARA × 0.25 ─────────────────────────
    df.loc[df["Nama Barang"] == "SAGU MUTIARA", "Qty"] *= 0.25

    # ── Cell 8–9: BESEK / 18, rename ─────────────────────────
    mask_kodi = (df["Nama Barang"] == "BESEK - 1 KODI") & (df["Satuan"].str.lower() == "kodi")
    df.loc[mask_kodi, "Qty"] /= 18
    df.loc[df["Nama Barang"] == "BESEK - 1 KODI", "Satuan"] = "Bal"
    df.loc[df["Nama Barang"] == "BESEK - 1 KODI", "Nama Barang"] = "BESEK"

    # ── Cell 10: CREAMER NDC DONGXIAO ────────────────────────
    df.loc[df["Nama Barang"] == "CREAMER NDC DONGXIAO 250 GR", "Qty"] *= 0.25
    df.loc[df["Nama Barang"] == "CREAMER NDC DONGXIAO 500 GR", "Qty"] *= 0.5
    df.loc[df["Nama Barang"] == "CREAMER NDC DONGXIAO 250 GR", "Satuan"] = "Kg"
    df.loc[df["Nama Barang"] == "CREAMER NDC DONGXIAO 500 GR", "Satuan"] = "Kg"
    df.loc[df["Nama Barang"].isin([
        "CREAMER NDC DONGXIAO 250 GR",
        "CREAMER NDC DONGXIAO 500 GR",
        "CREAMER NDC DONGXIAO 1 KG",
    ]), "Nama Barang"] = "CREAMER NDC DONGXIAO"

    # ── Cell 11–12: TOPLES TABUNG 1000 ML ────────────────────
    mask_t1 = (df["Nama Barang"] == "TOPLES TABUNG 1000 ML - Tebal") & (df["Harga"] == 224500)
    df.loc[mask_t1, "Qty"] *= 60
    df.loc[mask_t1, "Satuan"] = "Pcs"

    # ── Cell 13: TOPLES TABUNG 800 ML ────────────────────────
    mask_t2 = (df["Nama Barang"] == "TOPLES TABUNG 800 ML - Tebal") & (df["Harga"] == 212000)
    df.loc[mask_t2, "Qty"] *= 60
    df.loc[mask_t2, "Satuan"] = "Pcs"

    # ── Cell 14–17: HD LOS BENING × 0.5 ─────────────────────
    hd_los = [
        ("HD LOS BENING 15 - 9 ONS", 15000),
        ("HD LOS BENING 15 - 8 ONS", 14000),
        ("HD LOS BENING 24 - 9 ONS", 15000),
        ("HD LOS BENING 24 - 8 ONS", 14000),
    ]
    for nama, harga in hd_los:
        mask = (df["Nama Barang"] == nama) & (df["Harga"] == harga)
        df.loc[mask, "Qty"] *= 0.5

    return df


# ─────────────────────────────────────────────────────────────
# STEP 2: ANALISIS ABC  (Cell 18–24)
# ─────────────────────────────────────────────────────────────
def analisis_abc(df: pd.DataFrame):
    """
    Cell 18: groupby Nama Barang + Satuan → sum Qty & Total Harga
    Cell 19: groupby Nama Barang → sum Total Harga, sort descending
    Cell 20: n_a=20%, n_b=30%, n_c=sisanya (pakai round() seperti notebook)
    Cell 21: hitung pendapatan per grup
    Returns: (total_penjualan, grup_a, grup_b, grup_c, ringkasan)
    """
    # Cell 18
    set_1 = df.groupby(["Nama Barang", "Satuan"], as_index=False).agg(
        {"Qty": "sum", "Total Harga": "sum"}
    )

    # Cell 19
    total_penjualan = (
        set_1.groupby("Nama Barang", as_index=False)["Total Harga"]
        .sum()
        .sort_values("Total Harga", ascending=False)
    )
    total_penjualan["Total Harga"] = pd.to_numeric(total_penjualan["Total Harga"], errors="coerce").fillna(0)
    total_penjualan = total_penjualan.reset_index(drop=True)
    total_penjualan = total_penjualan.rename(columns={"Total Harga": "total_harga"})

    # Cell 20: gunakan round() persis seperti notebook
    n = len(total_penjualan)
    n_a = round(n * 0.2)
    n_b = round(n * 0.3)
    n_c = n - (n_a + n_b)

    grup_a = total_penjualan.iloc[:n_a].reset_index(drop=True)
    grup_b = total_penjualan.iloc[n_a:n_a + n_b].reset_index(drop=True)
    grup_c = total_penjualan.iloc[n_a + n_b:].reset_index(drop=True)

    # Cell 21
    pendapatan_all = float(total_penjualan["total_harga"].sum())
    pendapatan_a   = float(grup_a["total_harga"].sum())
    pendapatan_b   = float(grup_b["total_harga"].sum())
    pendapatan_c   = float(grup_c["total_harga"].sum())

    ringkasan = {
        "n_produk": n,
        "n_a": n_a, "n_b": n_b, "n_c": n_c,
        "persen_produk": {
            "A": n_a / n * 100,
            "B": n_b / n * 100,
            "C": n_c / n * 100,
        },
        "persen_pendapatan": {
            "A": pendapatan_a / pendapatan_all * 100,
            "B": pendapatan_b / pendapatan_all * 100,
            "C": pendapatan_c / pendapatan_all * 100,
        },
        "total_pendapatan": pendapatan_all,
    }

    return total_penjualan, grup_a, grup_b, grup_c, ringkasan


# ─────────────────────────────────────────────────────────────
# STEP 3: LOAD MASTER KONVERSI
# ─────────────────────────────────────────────────────────────
def load_master_konversi(path: str = "master_konversi.csv") -> pd.DataFrame:
    mk = pd.read_csv(path)
    mk["Nama Barang"]     = mk["Nama Barang"].astype(str).str.strip()
    mk["Dari Satuan"]     = mk["Dari Satuan"].astype(str).str.strip()
    mk["Ke Satuan Final"] = mk["Ke Satuan Final"].astype(str).str.strip()
    mk["Multiplier"] = (
        mk["Multiplier"].astype(str)
        .str.replace(",", ".", regex=False)
    )
    mk["Multiplier"]      = pd.to_numeric(mk["Multiplier"], errors="coerce")
    mk["_match_satuan"]   = mk["Dari Satuan"].str.lower()
    return mk


# ─────────────────────────────────────────────────────────────
# STEP 4: KONVERSI SATUAN via master_konversi  (Cell 26–28)
# ─────────────────────────────────────────────────────────────
def konversi_satuan(df: pd.DataFrame, master_konversi: pd.DataFrame) -> pd.DataFrame:
    """
    Cell 27 (preprocessing_pipeline):
    - Merge dengan master_konversi by (Nama Barang, satuan lowercase)
    - Qty × Multiplier
    - Satuan → Ke Satuan Final
    - Drop kolom bantu
    CATATAN: di notebook, Harga TIDAK dikonversi (tidak ada df["Harga"] / Multiplier)
    """
    df = df.copy()

    df["Nama Barang"] = df["Nama Barang"].astype(str).str.strip()
    df["Satuan"]      = df["Satuan"].astype(str).str.strip()
    df["Qty"]         = pd.to_numeric(df["Qty"], errors="coerce")
    df["Harga"]       = pd.to_numeric(df["Harga"], errors="coerce")
    df["Tanggal Transaksi"] = pd.to_datetime(df["Tanggal Transaksi"])

    df["_match_satuan"] = df["Satuan"].str.lower()

    df = df.merge(
        master_konversi[["Nama Barang", "_match_satuan", "Ke Satuan Final", "Multiplier"]],
        on=["Nama Barang", "_match_satuan"],
        how="left"
    )

    # Qty × Multiplier (fillna 1 = tidak ada konversi)
    df["Qty"] = df["Qty"] * df["Multiplier"].fillna(1)

    # Update satuan ke satuan final
    df["Satuan"] = df["Ke Satuan Final"].fillna(df["Satuan"])

    df = df.drop(columns=["_match_satuan", "Ke Satuan Final", "Multiplier"])

    return df


# ─────────────────────────────────────────────────────────────
# STEP 5: BUILD WEEKLY TIME SERIES  (Cell 30–33)
# ─────────────────────────────────────────────────────────────
def build_weekly_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cell 30: groupby Nama Barang + Satuan + Tanggal → sum Qty (harian)
    Cell 30: full cross-join produk × date_range → fill 0
    Cell 33: resample W (akhir minggu Minggu)
    """
    # Cell 30
    df_daily = (
        df.groupby(["Nama Barang", "Satuan", "Tanggal Transaksi"], as_index=False)
        .agg({"Qty": "sum"})
    )
    df_daily["Tanggal Transaksi"] = pd.to_datetime(df_daily["Tanggal Transaksi"])

    tanggal_range = pd.date_range(
        start=df_daily["Tanggal Transaksi"].min(),
        end=df_daily["Tanggal Transaksi"].max(),
        freq="D",
    )

    produk_list = df_daily[["Nama Barang", "Satuan"]].drop_duplicates()

    full_index = (
        produk_list.assign(key=1)
        .merge(pd.DataFrame({"Tanggal Transaksi": tanggal_range, "key": 1}), on="key")
        .drop("key", axis=1)
    )

    df_daily_full = full_index.merge(
        df_daily, on=["Nama Barang", "Satuan", "Tanggal Transaksi"], how="left"
    )
    df_daily_full["Qty"] = df_daily_full["Qty"].fillna(0)

    # Cell 31
    df_daily_full = df_daily_full.sort_values(
        ["Tanggal Transaksi", "Nama Barang"]
    ).reset_index(drop=True)

    # Cell 32–33: resample W
    df_weekly = (
        df_daily_full[["Tanggal Transaksi", "Nama Barang", "Satuan", "Qty"]]
        .set_index("Tanggal Transaksi")
        .groupby(["Nama Barang", "Satuan"])["Qty"]
        .resample("W")
        .sum()
        .reset_index()
    )

    return df_weekly


# ─────────────────────────────────────────────────────────────
# STEP 6: ANALISIS ZERO PERCENTAGE  (Cell 34–36)
# ─────────────────────────────────────────────────────────────
def hitung_zero_percentage(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Cell 34–36 notebook — TETAPI dikoreksi agar Total_Weeks selalu = n minggu unik
    (bukan size per produk yang bisa 2× lipat karena multi-satuan).

    Caranya: collapse satuan dulu (sum Qty per Nama Barang + Tanggal),
    lalu hitung zero dari data yang sudah di-collapse.
    """
    # Collapse multi-satuan → 1 baris per (produk, minggu)
    df = (
        df_weekly
        .groupby(["Nama Barang", "Tanggal Transaksi"], as_index=False)["Qty"]
        .sum()
    )

    # Total minggu unik di seluruh dataset (sama untuk semua produk)
    n_minggu_total = df["Tanggal Transaksi"].nunique()

    zero_count = (
        df[df["Qty"] == 0]
        .groupby("Nama Barang")
        .size()
        .reset_index(name="Zero_Count")
    )

    total_count = (
        df.groupby("Nama Barang")
        .size()
        .reset_index(name="Total_Weeks")
    )
    # Override Total_Weeks dengan jumlah minggu global
    total_count["Total_Weeks"] = n_minggu_total

    df_zero = total_count.merge(zero_count, on="Nama Barang", how="left")
    df_zero["Zero_Count"] = df_zero["Zero_Count"].fillna(0)
    df_zero["Zero_Percentage"] = (df_zero["Zero_Count"] / df_zero["Total_Weeks"]) * 100
    df_zero = df_zero.sort_values("Zero_Percentage", ascending=True).reset_index(drop=True)

    return df_zero


# ─────────────────────────────────────────────────────────────
# STEP 7: FILTER PRODUK  (Cell 37)
# ─────────────────────────────────────────────────────────────
def filter_produk_peramalan(
    df_weekly: pd.DataFrame,
    produk_grup_a: list,
    zero_threshold: float,
    df_zero: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cell 37: filter produk dengan Zero_Percentage ≤ threshold.
    Jika threshold = 100, semua produk Grup A diramal.
    """
    produk_a_set = set(produk_grup_a)

    if zero_threshold >= 100:
        valid = produk_a_set
    else:
        valid_zero = set(
            df_zero[df_zero["Zero_Percentage"] <= zero_threshold]["Nama Barang"].tolist()
        )
        valid = produk_a_set & valid_zero

    return df_weekly[df_weekly["Nama Barang"].isin(valid)].copy()


# ─────────────────────────────────────────────────────────────
# STEP 8: FEATURE ENGINEERING  (Cell 39–49)
# ─────────────────────────────────────────────────────────────
def buat_fitur(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Persis seperti Cell 39–49 notebook:
    - ID_Barang (category codes)
    - Bulan, Tahun, Minggu_dalam_Tahun, Minggu_dalam_Bulan
    - Lag 1,2,3,4,8,12
    - Avg/Std/Max/Min/Median 4,8,12 (shift(1).rolling)
    - sort by Tanggal Transaksi
    - buang tanggal 2026-01-04 jika ada (artefak resample)
    - fillna(0)
    """
    df = df_weekly.copy()
    df["Tanggal Transaksi"] = pd.to_datetime(df["Tanggal Transaksi"])
    df = df.sort_values(["Nama Barang", "Tanggal Transaksi"]).reset_index(drop=True)

    # Cell 39: ID_Barang
    df["ID_Barang"] = df["Nama Barang"].astype("category").cat.codes

    # Cell 40: fitur temporal
    df["Bulan"]             = df["Tanggal Transaksi"].dt.month
    df["Tahun"]             = df["Tanggal Transaksi"].dt.year
    df["Minggu_dalam_Tahun"] = df["Tanggal Transaksi"].dt.isocalendar().week.astype(int)
    df["Minggu_dalam_Bulan"] = (df["Tanggal Transaksi"].dt.day - 1) // 7 + 1

    # Cell 41: lag
    for lag in [1, 2, 3, 4, 8, 12]:
        df[f"Lag_{lag}_Minggu"] = df.groupby("Nama Barang")["Qty"].shift(lag)

    # Cell 42: rolling avg
    for w in [4, 8, 12]:
        df[f"Avg_{w}_Minggu"] = (
            df.groupby("Nama Barang")["Qty"]
            .transform(lambda x: x.shift(1).rolling(w).mean())
        )

    # Cell 39: rolling std (4, 8, 12)
    for w in [4, 8, 12]:
        df[f"Std_{w}_Minggu"] = (
            df.groupby("Nama Barang")["Qty"]
            .transform(lambda x: x.shift(1).rolling(w).std())
        )

    # Cell 40: rolling max (4, 8, 12)
    for w in [4, 8, 12]:
        df[f"Max_{w}_Minggu"] = (
            df.groupby("Nama Barang")["Qty"]
            .transform(lambda x: x.shift(1).rolling(w).max())
        )

    # Cell 41: rolling min (4, 8, 12)
    for w in [4, 8, 12]:
        df[f"Min_{w}_Minggu"] = (
            df.groupby("Nama Barang")["Qty"]
            .transform(lambda x: x.shift(1).rolling(w).min())
        )

    # Cell 46: rolling median
    for w in [4, 8, 12]:
        df[f"Median_{w}_Minggu"] = (
            df.groupby("Nama Barang")["Qty"]
            .transform(lambda x: x.shift(1).rolling(w).median())
        )

    # Cell 47: sort by Tanggal Transaksi
    df = df.sort_values(["Tanggal Transaksi","Nama Barang"]).reset_index(drop=True)

    # Cell 49: fillna(0)
    df = df.fillna(0).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────
# STEP 9: TRAIN/VAL SPLIT — DATE-BASED  (Cell 49–50)
# ─────────────────────────────────────────────────────────────
def split_train_val(df: pd.DataFrame, test_size: float = 0.20):
    """
    Cell 49: date_based_train_val_split.

    Split berdasarkan tanggal UNIK, bukan jumlah baris.
    → Semua produk di minggu yang sama selalu masuk ke split yang sama.
    → Tidak ada data leakage antar minggu lintas split.

    val_ratio = proporsi jumlah tanggal unik yang masuk ke validasi.
    """
    unique_dates = sorted(df["Tanggal Transaksi"].unique())
    n_dates      = len(unique_dates)

    n_val_dates   = max(1, round(n_dates * test_size))
    n_train_dates = n_dates - n_val_dates

    cutoff_date = unique_dates[n_train_dates - 1]   # tanggal terakhir train
    val_start   = unique_dates[n_train_dates]        # tanggal pertama val

    train = df[df["Tanggal Transaksi"] <= cutoff_date]
    val   = df[df["Tanggal Transaksi"] >= val_start]

    avail   = [f for f in FEATURES if f in df.columns]
    X_train = train[avail]
    y_train = train[TARGET]
    X_val   = val[avail]
    y_val   = val[TARGET]

    return X_train, y_train, X_val, y_val


def hitung_batas_train(df: pd.DataFrame, test_size: float = 0.20) -> pd.Timestamp:
    """Kembalikan tanggal cutoff (tanggal terakhir data train)."""
    unique_dates  = sorted(df["Tanggal Transaksi"].unique())
    n_dates       = len(unique_dates)
    n_val_dates   = max(1, round(n_dates * test_size))
    n_train_dates = n_dates - n_val_dates
    return pd.to_datetime(unique_dates[n_train_dates - 1])


def info_split(df: pd.DataFrame, test_size: float = 0.20) -> dict:
    """Info jumlah minggu dan baris per split (untuk ditampilkan di UI)."""
    unique_dates  = sorted(df["Tanggal Transaksi"].unique())
    n_dates       = len(unique_dates)
    n_val_dates   = max(1, round(n_dates * test_size))
    n_train_dates = n_dates - n_val_dates
    cutoff_date   = unique_dates[n_train_dates - 1]
    val_start     = unique_dates[n_train_dates]
    train = df[df["Tanggal Transaksi"] <= cutoff_date]
    val   = df[df["Tanggal Transaksi"] >= val_start]
    return {
        "n_minggu_total" : n_dates,
        "n_minggu_train" : n_train_dates,
        "n_minggu_val"   : n_val_dates,
        "cutoff_date"    : pd.to_datetime(cutoff_date),
        "val_start"      : pd.to_datetime(val_start),
        "n_rows_train"   : len(train),
        "n_rows_val"     : len(val),
    }


# ─────────────────────────────────────────────────────────────
# STEP 10: HITUNG MINGGU PREDIKSI
# ─────────────────────────────────────────────────────────────
def hitung_minggu_prediksi(df: pd.DataFrame, n_minggu: int = 4):
    """
    Hitung rentang (Senin–Minggu) untuk n_minggu minggu ke depan.
    Senin pertama = Senin di minggu setelah tanggal terakhir dataset.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame kosong.")
    if "Tanggal Transaksi" not in df.columns:
        raise ValueError("Kolom 'Tanggal Transaksi' tidak ditemukan.")

    tgl_valid = pd.to_datetime(df["Tanggal Transaksi"], errors="coerce").dropna()
    if tgl_valid.empty:
        raise ValueError("Semua nilai Tanggal Transaksi adalah NaT.")

    tgl_max = tgl_valid.max()
    wd = tgl_max.weekday()           # 0=Senin … 6=Minggu
    days_ahead = 1 if wd == 6 else (7 - wd)
    senin_pertama = tgl_max.normalize() + pd.Timedelta(days=days_ahead)

    return [(senin_pertama + pd.Timedelta(weeks=i),
             senin_pertama + pd.Timedelta(weeks=i, days=6))
            for i in range(n_minggu)]


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run_processing(raw_df: pd.DataFrame, master_konversi: pd.DataFrame):
    """
    Alur lengkap sesuai notebook:
    1. clean_raw()          → Cell 2–17
    2. analisis_abc()       → Cell 18–24
    3. filter ke Grup A     → Cell 25
    4. konversi_satuan()    → Cell 26–28
    5. build_weekly_full()  → Cell 30–33
    6. hitung_zero_pct()    → Cell 34–36

    Returns: (df_weekly_all, grup_a, grup_b, grup_c, ringkasan, df_zero_all)
    """
    try:
        # 1. Clean raw
        df_clean = clean_raw(raw_df)

        # 2. ABC Analysis
        _, grup_a, grup_b, grup_c, ringkasan = analisis_abc(df_clean)

        # 3. Filter ke Grup A (Cell 25)
        barang_a = grup_a["Nama Barang"].unique()
        df_filtered = df_clean[df_clean["Nama Barang"].isin(barang_a)].copy()
        df_filtered = df_filtered.drop(columns=["Total Harga"], errors="ignore")
        df_filtered["Tanggal Transaksi"] = pd.to_datetime(df_filtered["Tanggal Transaksi"])
        df_filtered = df_filtered.sort_values("Tanggal Transaksi").reset_index(drop=True)

        # 4. Konversi satuan via master_konversi (Cell 26–28)
        df_filtered_1 = konversi_satuan(df_filtered, master_konversi)

        # 5. Build weekly (Cell 30–33) — dari data SESUDAH konversi
        df_weekly_all = build_weekly_full(df_filtered_1)

        # 6. Zero percentage (Cell 34–36)
        df_zero_all = hitung_zero_percentage(df_weekly_all)

        return df_weekly_all, grup_a, grup_b, grup_c, ringkasan, df_zero_all

    except Exception as e:
        st.error(f"❌ Error saat preprocessing: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None, None, None