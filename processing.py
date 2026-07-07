import warnings
import pandas as pd
import numpy as np
import streamlit as st


# ─────────────────────────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────────────────────────
FEATURES = [
    "Tahun", "Bulan", "Minggu_dalam_Bulan", "Minggu_dalam_Tahun",
    "Lag_1_Minggu", "Lag_2_Minggu", "Lag_3_Minggu", "Lag_4_Minggu",
    "Lag_8_Minggu", "Lag_12_Minggu",
    "Avg_4_Minggu", "Avg_8_Minggu", "Avg_12_Minggu",
    "Std_4_Minggu", "Std_8_Minggu", "Std_12_Minggu",
    "Max_4_Minggu", "Max_8_Minggu", "Max_12_Minggu",
    "Min_4_Minggu", "Min_8_Minggu", "Min_12_Minggu",
    "Median_4_Minggu", "Median_8_Minggu", "Median_12_Minggu",
]
TARGET = "Qty"

KATEGORI_ADI = ["All", "Smooth Demand", "Erratic Demand", "Intermittent Demand", "Lumpy Demand"]

MODEL_FILES = {
    "XGBoost": {
        "All":          "xgb_all.pkl",
        "Smooth Demand":       "xgb_smooth.pkl",
        "Erratic Demand":      "xgb_erractic.pkl",
        "Intermittent Demand": "xgb_intermittent.pkl",
        "Lumpy Demand":        "xgb_lumpy.pkl",
    },
    "TabNet": {
        "All":          "tabnet_all.zip",
        "Smooth Demand":       "tabnet_smooth.zip",
        "Erratic Demand":      "tabnet_erractic.zip",
        "Intermittent Demand": "tabnet_intermittent.zip",
        "Lumpy Demand":        "tabnet_lumpy.zip",
    },
}


# ─────────────────────────────────────────────────────────────
# HELPER INTERNAL
# ─────────────────────────────────────────────────────────────
def _bersihkan_angka(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace("Rp", "", regex=False)
    s = s.str.replace(".", "", regex=False)
    s = s.str.replace(",", "", regex=False)
    s = s.str.strip()
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _parse_tgl(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    for fmt in [
        "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y",
        "%d-%m-%Y %H:%M:%S", "%d-%m-%Y",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
    ]:
        try:
            parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
            if parsed.notna().mean() > 0.8:
                return parsed
        except Exception:
            continue
    return pd.to_datetime(raw, dayfirst=True, errors="coerce")


# ─────────────────────────────────────────────────────────────
# STEP 1: CLEAN RAW
# ─────────────────────────────────────────────────────────────
def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = {"Nama Barang", "Qty", "Satuan", "Harga", "Total Harga", "Tanggal Transaksi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    drop_cols = [c for c in ["Id", "Id Transaksi", "Diskon", "SubTotal"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    for col in ["Harga", "Total Harga"]:
        df[col] = _bersihkan_angka(df[col]).astype("int64")

    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0).astype(float)

    df["Tanggal Transaksi"] = _parse_tgl(df["Tanggal Transaksi"])
    n_nat = df["Tanggal Transaksi"].isna().sum()
    if n_nat > 0:
        warnings.warn(f"{n_nat} baris tanggal tidak dikenali dan dibuang.")
    df = df.dropna(subset=["Tanggal Transaksi"])
    if df.empty:
        raise ValueError("Tidak ada baris valid setelah parsing tanggal.")
    df["Tanggal Transaksi"] = df["Tanggal Transaksi"].dt.normalize()

    df["Nama Barang"] = df["Nama Barang"].astype(str).str.strip().str.upper()

    SATUAN_NULL_FIX = {
        "NUTRIJELL RANDOM":           "Sachet",
        "HD KRESEK ECER":             "Pcs",
        "JOLLY FACIAL SOFTPACK 250S": "Pack",
        "SARUNG TANGAN KHARISMA":     "Pack",
        "GARPU KUE/BUAH THREE STAR":  "Pack",
        "KLIR SQ 120 ML":             "Pack",
        "WIPES SANITIZER - SANITER":  "Pack",
        "PASEO SMART FACIAL 540 PLY": "Pack",
        "SEAL CUP TOS FRUIT":         "Roll",
    }
    for nama, satuan in SATUAN_NULL_FIX.items():
        mask = (df["Nama Barang"] == nama) & (df["Satuan"].isnull())
        df.loc[mask, "Satuan"] = satuan

    df["Satuan"] = df["Satuan"].astype(str).str.strip().str.upper()
    df["Satuan"] = df["Satuan"].replace({"BALL": "BAL", "GR": "GRAM", "M": "METER"})

    return df


# ─────────────────────────────────────────────────────────────
# STEP 2: RENAME PRODUK (untuk Analisis ABC)
# ─────────────────────────────────────────────────────────────
def rename_produk_abc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    RENAME_MAP = [
        (["A. FOIL OV-290 (258/36) + LID", "A. FOIL OV-290 (258/36) - 10PCS"],       "A. FOIL OV-290 (258/36)"),
        (["A. FOIL OX-100 (111/34) + LID", "A. FOIL OX-100 (111/34) + LID - 10PCS"], "A. FOIL OX-100 (111/34)"),
        (["A. FOIL OX-1550 (1450/44) + LID"],                                          "A. FOIL OX-1550 (1450/44)"),
        (["ANTAKA BALADO - 10PCS"],                                                     "ANTAKA BALADO"),
        (["ANTAKA CAE BUBUK TUM-TUM - 10PCS", "ANTAKA CAE BUBUK TUM-TUM"],            "ANTAKA CABE BUBUK TUM-TUM"),
        (["ANTAKA JAGUNG MANIS - 10PCS"],                                               "ANTAKA JAGUNG MANIS"),
        (["BESEK - 1 KODI", "BESEK TELOR CEPER", "BESEK TELUR TEBAL", "BESEK TELUR TEBAL - 1 KODI"], "BESEK"),
        (["CREAMER NDC DONGXIAO 1 KG", "CREAMER NDC DONGXIAO 250 GR", "CREAMER NDC DONGXIAO 500 GR"], "CREAMER NDC DONGXIAO"),
        (["DM BENTO SEKAT 4 - LB 401S"],   "DM BENTO SEKAT 4"),
        (["DM RO 400 ML"],                  "DM R 400 ML"),
        (["DM 50 ML"],                      "DM SAUCE 50 ML"),
        (["GELAS 16 OZ BEST TEH TECHNO"],   "GELAS BEST TEH TECHNO 16 OZ"),
        (["GELAS 18 OZ SLIM 'TEN-TEN'"],    "GELAS SLIM 'TEN-TEN' 18 OZ"),
        (["GELAS 22 OZ SLIM 'TEN-TEN'"],    "GELAS SLIM 'TEN-TEN' 22 OZ"),
        (["GELAS TECHNO 22 OZ SLIM"],       "GELAS SLIM TECHNO 22 OZ"),
        (["GELAS TENTEN 18 OZ REGULER"],    "GELAS TENTEN 18 OZ"),
        (["GELAS TEN TEN 14 OZ"],           "GELAS TENTEN 14 OZ"),
        (["GELAS TEN TEN 16 OZ"],           "GELAS TENTEN 16 OZ"),
        (["JOLLY FACIAL SOFTPACK 200S - 3PACK"], "JOLLY FACIAL SOFTPACK 200S - 3 PACK"),
        (["KARET SMILE - KUNING - 10 PACK"], "KARET SMILE - KUNING"),
        (["KARET SMILE - MERAH - 10 PACK"],  "KARET SMILE - MERAH"),
        (["KERTAS ROTI - PUTIH *10", "KERTAS ROTI - PUTIH *5"], "KERTAS ROTI - PUTIH"),
        (["MIKA BG 225 - BENING (10PCS)", "MIKA BG 225 - BENING (20PCS)"], "MIKA BG 225 - BENING"),
        (["MIKA BGT 225 ISI 10 PCS"],       "MIKA BGT 225"),
        (["MIKA BGT 25 - 50 PCS"],          "MIKA BGT 25"),
        (["MIKA MB  BUAH 100"],             "MIKA MB BUAH 100"),
        (["MIKA TUMPENG CT - 20 MMPG"],     "MIKA TUMPENG CT - 20"),
        (["ORIPACK NPN S - 10 PCS"],        "ORIPACK NPN S"),
        (["PE TOMAT - 50 X 75 - 1 IKAT", "PE TOMAT - 50 X 75 - 1 PACK", "PE TOMAT - 50 X 75 - 2 PACK"], "PE TOMAT - 50 X 75"),
        (["PITA TP ROTI MIX 1KG", "PITA TP ROTI MIX 250GR", "PITA TP ROTI MIX 500GR"], "PITA TP ROTI MIX"),
        (["TEPUNG ROTI BINTANG 10 KG", "TEPUNG ROTI BINTANG 1KG", "TEPUNG ROTI BINTANG 250GR", "TEPUNG ROTI BINTANG 500GR"], "TEPUNG ROTI BINTANG"),
        (["TEPUNG ROTI JAWARA MIX - 500 GR", "TEPUNG ROTI JAWARA MIX - 1 KG"], "TEPUNG ROTI JAWARA MIX"),
        (["TOPLES TABUNG 600 ML ISI 35", "TOPLES TABUNG 600 ML ISI 50"], "TOPLES TABUNG 600 ML"),
        (["HDPE ATP TIGER - 15 X30"],       "HDPE ATP TIGER - 15 X 30"),
        (["HDPEX BENING 30"],               "HD PEX BENING 30"),
        (["KOTAK ASHLEY POLOS 20 X 20 - P.T"], "KOTAK ASHELY POLOS 20 X 20 - P.T"),
        (["PP MAKHOTA 07 X 12 X 25"],       "PP MAHKOTA 07 X 12 X 25"),
        (["PP MAKHOTA 07 X 15 X 25"],       "PP MAHKOTA 07 X 15 X 25"),
        (["PP MAKHOTA 07 X 15 X 30"],       "PP MAHKOTA 07 X 15 X 30"),
        (["PP MAKHOTA 07 X 17 X 25"],       "PP MAHKOTA 07 X 17 X 25"),
        (["PP MAKHOTA 07 X 17 X 30"],       "PP MAHKOTA 07 X 17 X 30"),
        (["PP MAKHOTA 07 X 17 X 35"],       "PP MAHKOTA 07 X 17 X 35"),
        (["PP MAKHOTA 07 X 20 X 35"],       "PP MAHKOTA 07 X 20 X 35"),
    ]

    df.loc[
        df["Nama Barang"].str.contains(r"^A\. FOIL OIV-450 \(400/50\)", regex=True, na=False),
        "Nama Barang"
    ] = "A. FOIL OIV-450 (400/50)"

    for nama_lama_list, nama_baru in RENAME_MAP:
        df.loc[df["Nama Barang"].isin(nama_lama_list), "Nama Barang"] = nama_baru

    BUNDLE_SPLITS = [
        ("NESTO PAPER BOWL + LID 360 ML",         "NESTO PAPER BOWL 360 ML",         "LID TUTUP 360/500 ML"),
        ("NESTO PAPER BOWL + LID 360 ML - KRAFT",  "NESTO PAPER BOWL 360 ML - KRAFT", "LID TUTUP 360/500 ML"),
        ("NESTO PAPER BOWL + LID 500 ML",          "NESTO PAPER BOWL 500 ML",         "LID TUTUP 360/500 ML"),
        ("NESTO PAPER BOWL + LID 500 ML - KRAFT",  "NESTO PAPER BOWL 500 ML - KRAFT", "LID TUTUP 360/500 ML"),
        ("NESTO PAPER BOWL + LID 650 ML",          "NESTO PAPER BOWL 650 ML",         "LID TUTUP 650/800 ML"),
        ("NESTO PAPER BOWL + LID 650 ML - KRAFT",  "NESTO PAPER BOWL 650 ML - KRAFT", "LID TUTUP 650/800 ML"),
        ("NESTO PAPER BOWL + LID 800 ML",          "NESTO PAPER BOWL 800 ML",         "LID TUTUP 650/800 ML"),
        ("NESTO PAPER BOWL + LID 800 ML - KRAFT",  "NESTO PAPER BOWL 800 ML - KRAFT", "LID TUTUP 650/800 ML"),
    ]

    def _mode_harga(nama, satuan=None):
        q = df[df["Nama Barang"] == nama]
        if satuan:
            q = q[q["Satuan"] == satuan]
        if q.empty:
            return 0
        return int(q["Harga"].mode()[0]) if not q["Harga"].mode().empty else 0

    bundle_names = [b[0] for b in BUNDLE_SPLITS]
    dfs = [df[~df["Nama Barang"].isin(bundle_names)]]

    for bundle_name, bowl_name, lid_name in BUNDLE_SPLITS:
        mask = df["Nama Barang"] == bundle_name
        if not mask.any():
            continue
        data = df[mask].copy()

        harga_bowl = _mode_harga(bowl_name, "PACK")
        harga_lid  = _mode_harga(lid_name, "PACK")
        if bundle_name == "NESTO PAPER BOWL + LID 500 ML - KRAFT" and harga_bowl == 0:
            harga_bowl = 12000

        df_bowl = data.copy()
        df_bowl["Nama Barang"] = bowl_name
        df_bowl["Satuan"]      = "PACK"
        df_bowl["Harga"]       = harga_bowl
        df_bowl["Total Harga"] = (df_bowl["Qty"] * harga_bowl).astype(int)

        df_lid = data.copy()
        df_lid["Nama Barang"]  = lid_name
        df_lid["Satuan"]       = "PACK"
        df_lid["Harga"]        = harga_lid
        df_lid["Total Harga"]  = (df_lid["Qty"] * harga_lid).astype(int)

        dfs.extend([df_bowl, df_lid])

    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values("Tanggal Transaksi").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# STEP 3: ANALISIS ABC
# ─────────────────────────────────────────────────────────────
def analisis_abc(df_abc: pd.DataFrame):
    total_penjualan = (
        df_abc.groupby("Nama Barang", as_index=False)["Total Harga"]
        .sum()
        .sort_values("Total Harga", ascending=False)
        .reset_index(drop=True)
        .rename(columns={"Total Harga": "total_harga"})
    )
    total_penjualan["total_harga"] = pd.to_numeric(
        total_penjualan["total_harga"], errors="coerce").fillna(0)

    n   = len(total_penjualan)
    n_a = round(n * 0.2)
    n_b = round(n * 0.3)
    n_c = n - (n_a + n_b)

    grup_a = total_penjualan.iloc[:n_a].reset_index(drop=True)
    grup_b = total_penjualan.iloc[n_a:n_a + n_b].reset_index(drop=True)
    grup_c = total_penjualan.iloc[n_a + n_b:].reset_index(drop=True)

    total_pend = float(total_penjualan["total_harga"].sum())
    ringkasan = {
        "n_produk": n, "n_a": n_a, "n_b": n_b, "n_c": n_c,
        "persen_produk": {
            "A": round(n_a / n * 100),
            "B": round(n_b / n * 100),
            "C": round(n_c / n * 100),
        },
        "persen_pendapatan": {
            "A": float(grup_a["total_harga"].sum()) / total_pend * 100,
            "B": float(grup_b["total_harga"].sum()) / total_pend * 100,
            "C": float(grup_c["total_harga"].sum()) / total_pend * 100,
        },
        "total_pendapatan": total_pend,
    }
    return total_penjualan, grup_a, grup_b, grup_c, ringkasan


# ─────────────────────────────────────────────────────────────
# STEP 4: RENAME PRODUK (untuk data modeling)
# ─────────────────────────────────────────────────────────────
def rename_produk_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    RENAME_MODEL = [
        (["GELAS 18 OZ SLIM 'TEN-TEN'"], "GELAS SLIM 'TEN-TEN' 18 OZ"),
        (["GELAS 22 OZ SLIM 'TEN-TEN'"], "GELAS SLIM 'TEN-TEN' 22 OZ"),
        (["GELAS TENTEN 18 OZ REGULER"], "GELAS TENTEN 18 OZ"),
        (["GELAS TEN TEN 16 OZ"],        "GELAS TENTEN 16 OZ"),
        (["HDPE ATP TIGER - 15 X30"],    "HDPE ATP TIGER - 15 X 30"),
    ]
    for nama_lama_list, nama_baru in RENAME_MODEL:
        df.loc[df["Nama Barang"].isin(nama_lama_list), "Nama Barang"] = nama_baru

    BUNDLE_SPLITS_MODEL = [
        ("NESTO PAPER BOWL + LID 360 ML",         "NESTO PAPER BOWL 360 ML",         "LID TUTUP 360/500 ML"),
        ("NESTO PAPER BOWL + LID 360 ML - KRAFT",  "NESTO PAPER BOWL 360 ML - KRAFT", "LID TUTUP 360/500 ML"),
        ("NESTO PAPER BOWL + LID 500 ML",          "NESTO PAPER BOWL 500 ML",         "LID TUTUP 360/500 ML"),
        ("NESTO PAPER BOWL + LID 500 ML - KRAFT",  "NESTO PAPER BOWL 500 ML - KRAFT", "LID TUTUP 360/500 ML"),
        ("NESTO PAPER BOWL + LID 650 ML",          "NESTO PAPER BOWL 650 ML",         "LID TUTUP 650/800 ML"),
        ("NESTO PAPER BOWL + LID 650 ML - KRAFT",  "NESTO PAPER BOWL 650 ML - KRAFT", "LID TUTUP 650/800 ML"),
        ("NESTO PAPER BOWL + LID 800 ML",          "NESTO PAPER BOWL 800 ML",         "LID TUTUP 650/800 ML"),
        ("NESTO PAPER BOWL + LID 800 ML - KRAFT",  "NESTO PAPER BOWL 800 ML - KRAFT", "LID TUTUP 650/800 ML"),
    ]

    bundle_names = [b[0] for b in BUNDLE_SPLITS_MODEL]
    dfs = [df[~df["Nama Barang"].isin(bundle_names)]]

    for bundle_name, bowl_name, lid_name in BUNDLE_SPLITS_MODEL:
        mask = df["Nama Barang"] == bundle_name
        if not mask.any():
            continue
        data = df[mask].copy()
        bowl = data.copy(); bowl["Nama Barang"] = bowl_name; bowl["Satuan"] = "PACK"
        lid  = data.copy(); lid["Nama Barang"]  = lid_name;  lid["Satuan"]  = "PACK"
        dfs.extend([bowl, lid])

    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values("Tanggal Transaksi").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# STEP 5: LOAD MASTER KONVERSI
# ─────────────────────────────────────────────────────────────
def load_master_konversi(path: str = "master_konversi.csv") -> pd.DataFrame:
    mk = pd.read_csv(path)
    mk["Nama Barang"]     = mk["Nama Barang"].astype(str).str.strip()
    mk["Dari Satuan"]     = mk["Dari Satuan"].astype(str).str.strip()
    mk["Ke Satuan Final"] = mk["Ke Satuan Final"].astype(str).str.strip()
    mk["Multiplier"]      = pd.to_numeric(
        mk["Multiplier"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    mk["_match_satuan"]   = mk["Dari Satuan"].str.lower()
    return mk


# ─────────────────────────────────────────────────────────────
# STEP 6: KONVERSI SATUAN
# ─────────────────────────────────────────────────────────────
def konversi_satuan(df: pd.DataFrame, master_konversi: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Nama Barang"]       = df["Nama Barang"].astype(str).str.strip()
    df["Satuan"]            = df["Satuan"].astype(str).str.strip()
    df["Qty"]               = pd.to_numeric(df["Qty"], errors="coerce")
    df["Tanggal Transaksi"] = pd.to_datetime(df["Tanggal Transaksi"])
    df["_match_satuan"]     = df["Satuan"].str.lower()

    df = df.merge(
        master_konversi[["Nama Barang", "_match_satuan", "Ke Satuan Final", "Multiplier"]],
        on=["Nama Barang", "_match_satuan"], how="left"
    )
    df["Qty"]    = df["Qty"] * df["Multiplier"].fillna(1)
    df["Satuan"] = df["Ke Satuan Final"].fillna(df["Satuan"])
    df = df.drop(columns=["_match_satuan", "Ke Satuan Final", "Multiplier"])
    return df


# ─────────────────────────────────────────────────────────────
# STEP 7: RENAME SETELAH KONVERSI SATUAN
# ─────────────────────────────────────────────────────────────
def rename_after_conversion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    RENAME_POST = [
        (["BESEK - 1 KODI", "BESEK TELOR CEPER", "BESEK TELUR TEBAL", "BESEK TELUR TEBAL - 1 KODI"], "BESEK"),
        (["CREAMER NDC DONGXIAO 1 KG", "CREAMER NDC DONGXIAO 250 GR", "CREAMER NDC DONGXIAO 500 GR"], "CREAMER NDC DONGXIAO"),
        (["KARET SMILE - KUNING - 10 PACK"], "KARET SMILE - KUNING"),
        (["KARET SMILE - MERAH - 10 PACK"],  "KARET SMILE - MERAH"),
        (["ORIPACK NPN S - 10 PCS"],         "ORIPACK NPN S"),
        (["PE TOMAT - 50 X 75 - 1 IKAT", "PE TOMAT - 50 X 75 - 1 PACK", "PE TOMAT - 50 X 75 - 2 PACK"], "PE TOMAT - 50 X 75"),
        (["TOPLES TABUNG 600 ML ISI 35", "TOPLES TABUNG 600 ML ISI 50"], "TOPLES TABUNG 600 ML"),
    ]
    for nama_lama_list, nama_baru in RENAME_POST:
        df.loc[df["Nama Barang"].isin(nama_lama_list), "Nama Barang"] = nama_baru

    return df


# ─────────────────────────────────────────────────────────────
# STEP 8: BUILD WEEKLY TIME SERIES
# ─────────────────────────────────────────────────────────────
def build_weekly_full(df: pd.DataFrame) -> pd.DataFrame:
    df["Tanggal Transaksi"] = pd.to_datetime(df["Tanggal Transaksi"])

    minggu_range = pd.date_range(
        start=df["Tanggal Transaksi"].min(),
        end=df["Tanggal Transaksi"].max(),
        freq="W"
    )

    satuan_dominan = (
        df.groupby("Nama Barang")["Satuan"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        .reset_index()
        .rename(columns={"Satuan": "Satuan_Final"})
    )

    df_weekly = (
        df.set_index("Tanggal Transaksi")
        .groupby("Nama Barang")["Qty"]
        .resample("W")
        .sum()
        .reset_index()
    )
    df_weekly = df_weekly.merge(satuan_dominan, on="Nama Barang", how="left")
    df_weekly = df_weekly.rename(columns={"Satuan_Final": "Satuan"})

    produk_list = df_weekly[["Nama Barang", "Satuan"]].drop_duplicates()
    full_index  = (
        produk_list.assign(key=1)
        .merge(pd.DataFrame({"Tanggal Transaksi": minggu_range, "key": 1}), on="key")
        .drop("key", axis=1)
    )

    df_model = full_index.merge(
        df_weekly, on=["Nama Barang", "Satuan", "Tanggal Transaksi"], how="left"
    )
    df_model["Qty"] = df_model["Qty"].fillna(0)
    df_model = df_model.sort_values(["Nama Barang", "Tanggal Transaksi"]).reset_index(drop=True)
    return df_model[["Tanggal Transaksi", "Nama Barang", "Satuan", "Qty"]]


# ─────────────────────────────────────────────────────────────
# STEP 9: ADI & CV²
# ─────────────────────────────────────────────────────────────
def hitung_adi_cv2(df_model: pd.DataFrame) -> pd.DataFrame:
    def _adi_cv2(group):
        total_weeks  = len(group)
        active       = group[group["Qty"] > 0]
        active_weeks = len(active)
        zero_weeks   = total_weeks - active_weeks

        if active_weeks == 0:
            return pd.Series({
                "Total_Weeks":  total_weeks,
                "Active_Weeks": 0,
                "Zero_Weeks":   zero_weeks,
                "Zero_Pct":     100.0,
                "ADI":          np.inf,
                "Mean_Demand":  0.0,
                "Std_Demand":   0.0,
                "CV2":          np.nan,
                "Kategori":     "Never Sold",
            })

        adi         = total_weeks / active_weeks
        demand      = active["Qty"]
        mean_demand = demand.mean()
        std_demand  = demand.std(ddof=1) if active_weeks > 1 else 0.0
        cv2         = (std_demand / mean_demand) ** 2 if mean_demand > 0 else 0.0

        if adi < 1.32 and cv2 < 0.49:
            kategori = "Smooth Demand"
        elif adi < 1.32 and cv2 >= 0.49:
            kategori = "Erratic Demand"
        elif adi >= 1.32 and cv2 < 0.49:
            kategori = "Intermittent Demand"
        else:
            kategori = "Lumpy Demand"

        return pd.Series({
            "Total_Weeks":  total_weeks,
            "Active_Weeks": active_weeks,
            "Zero_Weeks":   zero_weeks,
            "Zero_Pct":     round(zero_weeks / total_weeks * 100, 2),
            "ADI":          round(adi, 4),
            "Mean_Demand":  round(mean_demand, 2),
            "Std_Demand":   round(std_demand, 2),
            "CV2":          round(cv2, 4),
            "Kategori":     kategori,
        })

    hasil = (
        df_model.groupby("Nama Barang")
        .apply(_adi_cv2, include_groups=False)
        .reset_index()
    )
    return hasil.sort_values(["Kategori", "ADI", "CV2"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# STEP 10: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def buat_fitur(df_model: pd.DataFrame) -> pd.DataFrame:
    df = df_model.copy()
    df["Tanggal Transaksi"] = pd.to_datetime(df["Tanggal Transaksi"])
    df = df.sort_values(["Nama Barang", "Tanggal Transaksi"]).reset_index(drop=True)

    df["ID_Barang"]          = df["Nama Barang"].astype("category").cat.codes
    df["Bulan"]              = df["Tanggal Transaksi"].dt.month
    df["Tahun"]              = df["Tanggal Transaksi"].dt.year
    df["Minggu_dalam_Tahun"] = df["Tanggal Transaksi"].dt.isocalendar().week.astype(int)
    df["Minggu_dalam_Bulan"] = (df["Tanggal Transaksi"].dt.day - 1) // 7 + 1

    for lag in [1, 2, 3, 4, 8, 12]:
        df[f"Lag_{lag}_Minggu"] = df.groupby("Nama Barang")["Qty"].shift(lag)

    for w in [4, 8, 12]:
        df[f"Avg_{w}_Minggu"] = df.groupby("Nama Barang")["Qty"].transform(
            lambda x: x.shift(1).rolling(w).mean())
        df[f"Std_{w}_Minggu"] = df.groupby("Nama Barang")["Qty"].transform(
            lambda x: x.shift(1).rolling(w).std())
        df[f"Max_{w}_Minggu"] = df.groupby("Nama Barang")["Qty"].transform(
            lambda x: x.shift(1).rolling(w).max())
        df[f"Min_{w}_Minggu"] = df.groupby("Nama Barang")["Qty"].transform(
            lambda x: x.shift(1).rolling(w).min())
        df[f"Median_{w}_Minggu"] = df.groupby("Nama Barang")["Qty"].transform(
            lambda x: x.shift(1).rolling(w).median())

    df = df.sort_values(["Tanggal Transaksi", "Nama Barang"]).reset_index(drop=True)
    return df.fillna(0).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# STEP 11: SPLIT DATA
# ─────────────────────────────────────────────────────────────
def split_train_val(df: pd.DataFrame, test_size: float = 0.20):
    unique_dates  = sorted(df["Tanggal Transaksi"].unique())
    n_dates       = len(unique_dates)
    n_val_dates   = max(1, round(n_dates * test_size))
    n_train_dates = n_dates - n_val_dates
    cutoff_date   = unique_dates[n_train_dates - 1]
    val_start     = unique_dates[n_train_dates]

    train = df[df["Tanggal Transaksi"] <= cutoff_date]
    val   = df[df["Tanggal Transaksi"] >= val_start]

    avail = [f for f in FEATURES if f in df.columns]
    return train[avail], train[TARGET], val[avail], val[TARGET]


def hitung_batas_train(df: pd.DataFrame, test_size: float = 0.20) -> pd.Timestamp:
    unique_dates  = sorted(df["Tanggal Transaksi"].unique())
    n_val_dates   = max(1, round(len(unique_dates) * test_size))
    n_train_dates = len(unique_dates) - n_val_dates
    return pd.to_datetime(unique_dates[n_train_dates - 1])


def info_split(df: pd.DataFrame, test_size: float = 0.20) -> dict:
    unique_dates  = sorted(df["Tanggal Transaksi"].unique())
    n_dates       = len(unique_dates)
    n_val_dates   = max(1, round(n_dates * test_size))
    n_train_dates = n_dates - n_val_dates
    cutoff_date   = unique_dates[n_train_dates - 1]
    val_start     = unique_dates[n_train_dates]
    train = df[df["Tanggal Transaksi"] <= cutoff_date]
    val   = df[df["Tanggal Transaksi"] >= val_start]
    return {
        "n_minggu_total":  n_dates,
        "n_minggu_train":  n_train_dates,
        "n_minggu_val":    n_val_dates,
        "cutoff_date":     pd.to_datetime(cutoff_date),
        "val_start":       pd.to_datetime(val_start),
        "n_rows_train":    len(train),
        "n_rows_val":      len(val),
    }


# ─────────────────────────────────────────────────────────────
# STEP 12: HITUNG MINGGU PREDIKSI
# ─────────────────────────────────────────────────────────────
def hitung_minggu_prediksi(df: pd.DataFrame, n_minggu: int = 4):
    tgl_valid = pd.to_datetime(df["Tanggal Transaksi"], errors="coerce").dropna()
    if tgl_valid.empty:
        raise ValueError("Tidak ada tanggal valid.")
    tgl_max    = tgl_valid.max()
    wd         = tgl_max.weekday()
    days_ahead = 1 if wd == 6 else (7 - wd)
    senin      = tgl_max.normalize() + pd.Timedelta(days=days_ahead)
    return [
        (senin + pd.Timedelta(weeks=i), senin + pd.Timedelta(weeks=i, days=6))
        for i in range(n_minggu)
    ]


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run_processing(raw_df: pd.DataFrame, master_konversi: pd.DataFrame):
    try:
        df_clean = clean_raw(raw_df)

        df_abc = rename_produk_abc(df_clean.copy())
        _, grup_a, grup_b, grup_c, ringkasan = analisis_abc(df_abc)

        df_model_raw = df_clean.drop(columns=["Harga", "Total Harga"], errors="ignore").copy()
        df_model_raw = rename_produk_model(df_model_raw)
        df_model_raw = konversi_satuan(df_model_raw, master_konversi)
        df_model_raw = rename_after_conversion(df_model_raw)

        barang_a     = set(grup_a["Nama Barang"].unique())
        df_model_raw = df_model_raw[df_model_raw["Nama Barang"].isin(barang_a)].copy()
        df_model_raw = df_model_raw.sort_values("Tanggal Transaksi").reset_index(drop=True)

        df_weekly_all = build_weekly_full(df_model_raw)
        hasil_adi     = hitung_adi_cv2(df_weekly_all)

        return df_weekly_all, hasil_adi, grup_a, grup_b, grup_c, ringkasan

    except Exception as e:
        st.error(f"❌ Error saat preprocessing: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None, None, None
