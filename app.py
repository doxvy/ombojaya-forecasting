import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import os
import joblib

from processing import (
    run_processing, load_master_konversi, buat_fitur,
    split_data_70_15_15, hitung_batas_train, info_split,
    hitung_minggu_prediksi, FEATURES, TARGET,
    KATEGORI_ADI, MODEL_FILES,
)

# ════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sistem Peramalan Ombo Jaya",
    layout="wide",
    page_icon="LogoToko.png"
)

st.markdown("""
<style>
    [data-testid="stFileUploaderDropzoneInstructions"] div small { display: none; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════
for _k, _v in {
    "tahap": 1,
    "df_weekly_all": None, "hasil_adi": None,
    "grup_a": None, "grup_b": None, "grup_c": None, "ringkasan": None,
    "df_weekly_fit": None, "X_val": None, "y_val": None, "X_test": None, "y_test": None, 
    "batas_train": None, "fitur_cols": None,
    "model_nama": None, "kategori_dipilih": None, "opsi_minggu": None,
    "y_pred_val": None,
    "mae": None, "r2": None,
    "tgl_transaksi_min": None, "tgl_transaksi_max": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════
def predict(model, X: pd.DataFrame, model_nama: str) -> np.ndarray:
    if model_nama == "TabNet":
        return model.predict(X.values.astype(np.float32)).ravel()
    return model.predict(X)


def get_fitur_cols(available_cols) -> list:
    return [f for f in FEATURES if f in available_cols]


@st.cache_resource
def load_model(model_nama: str, kategori: str):
    path = MODEL_FILES[model_nama][kategori]
    try:
        if model_nama == "TabNet":
            from pytorch_tabnet.tab_model import TabNetRegressor
            m = TabNetRegressor()
            abs_path = os.path.join(os.getcwd(), path) if not os.path.isabs(path) else path
            m.load_model(abs_path)
            return m, None
        return joblib.load(path), None

    except FileNotFoundError:
        cwd = os.getcwd()
        try:
            files = [f for f in os.listdir(cwd) if f.endswith((".pkl", ".zip"))]
        except Exception:
            files = []
        file_hint = f"File ditemukan di folder: `{files}`" if files \
                    else "Tidak ada file `.pkl` / `.zip` di working directory."
        return None, (
            f"File model **`{path}`** tidak ditemukan.\n\n"
            f"Working directory: `{cwd}`\n\n"
            f"{file_hint}\n\n"
            f"Pastikan semua 10 file model ada di folder yang sama dengan `app.py`:\n"
            f"- TabNet: `tabnet_all.zip`, `tabnet_smooth.zip`, `tabnet_erractic.zip`, "
            f"`tabnet_intermittent.zip`, `tabnet_lumpy.zip`\n"
            f"- XGBoost: `xgb_all.pkl`, `xgb_smooth.pkl`, `xgb_erractic.pkl`, "
            f"`xgb_intermittent.pkl`, `xgb_lumpy.pkl`"
        )
    except ImportError:
        return None, (
            "Library **`pytorch_tabnet`** tidak terinstall.\n"
            "Jalankan: `pip install pytorch-tabnet`"
        )
    except Exception as e:
        return None, (
            f"Gagal load model: `{e}`\n\n"
            f"Pastikan versi `pytorch_tabnet` saat training dan load sama, "
            f"dan file tidak corrupt."
        )


def roll_forward(inp: pd.DataFrame, qty_pred: float, tgl_next: pd.Timestamp) -> pd.DataFrame:
    inp = inp.copy()

    if "Tahun"              in inp.columns: inp["Tahun"]              = tgl_next.year
    if "Bulan"              in inp.columns: inp["Bulan"]              = tgl_next.month
    if "Minggu_dalam_Tahun" in inp.columns: inp["Minggu_dalam_Tahun"] = int(tgl_next.isocalendar()[1])
    if "Minggu_dalam_Bulan" in inp.columns: inp["Minggu_dalam_Bulan"] = (tgl_next.day - 1) // 7 + 1

    if "Lag_12_Minggu" in inp.columns and "Lag_8_Minggu"  in inp.columns: inp["Lag_12_Minggu"] = inp["Lag_8_Minggu"].values[0]
    if "Lag_8_Minggu"  in inp.columns and "Lag_4_Minggu"  in inp.columns: inp["Lag_8_Minggu"]  = inp["Lag_4_Minggu"].values[0]
    if "Lag_4_Minggu"  in inp.columns and "Lag_3_Minggu"  in inp.columns: inp["Lag_4_Minggu"]  = inp["Lag_3_Minggu"].values[0]
    if "Lag_3_Minggu"  in inp.columns and "Lag_2_Minggu"  in inp.columns: inp["Lag_3_Minggu"]  = inp["Lag_2_Minggu"].values[0]
    if "Lag_2_Minggu"  in inp.columns and "Lag_1_Minggu"  in inp.columns: inp["Lag_2_Minggu"]  = inp["Lag_1_Minggu"].values[0]
    if "Lag_1_Minggu"  in inp.columns: inp["Lag_1_Minggu"] = qty_pred

    lag_vals = {
        c: float(inp[c].values[0])
        for c in ["Lag_1_Minggu", "Lag_2_Minggu", "Lag_3_Minggu",
                  "Lag_4_Minggu", "Lag_8_Minggu", "Lag_12_Minggu"]
        if c in inp.columns
    }

    def _win(n):
        b = [lag_vals.get("Lag_1_Minggu", 0), lag_vals.get("Lag_2_Minggu", 0),
             lag_vals.get("Lag_3_Minggu", 0), lag_vals.get("Lag_4_Minggu", 0)]
        if n > 4: b += [lag_vals.get("Lag_8_Minggu",  b[-1])] * (min(n, 8) - 4)
        if n > 8: b += [lag_vals.get("Lag_12_Minggu", b[-1])] * (n - 8)
        return np.array(b[:n], dtype=float)

    for w in [4, 8, 12]:
        arr = _win(w)
        if f"Avg_{w}_Minggu"    in inp.columns: inp[f"Avg_{w}_Minggu"]    = float(np.mean(arr))
        if f"Std_{w}_Minggu"    in inp.columns: inp[f"Std_{w}_Minggu"]    = float(np.std(arr, ddof=1) if w > 1 else 0.0)
        if f"Max_{w}_Minggu"    in inp.columns: inp[f"Max_{w}_Minggu"]    = float(np.max(arr))
        if f"Min_{w}_Minggu"    in inp.columns: inp[f"Min_{w}_Minggu"]    = float(np.min(arr))
        if f"Median_{w}_Minggu" in inp.columns: inp[f"Median_{w}_Minggu"] = float(np.median(arr))

    return inp


def _std_produk(nama: str, df_weekly_all: pd.DataFrame) -> float:
    grp = (
        df_weekly_all[df_weekly_all["Nama Barang"] == nama]
        .groupby("Tanggal Transaksi", as_index=False)["Qty"].sum()
    )
    d = grp["Qty"]
    return float(d.std()) if len(d) > 1 else 0.0


def fmt_tabel_abc(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    d.index += 1
    d["total_harga"] = d["total_harga"].apply(lambda x: f"Rp {x:,.0f}")
    d.columns = ["Nama Barang", "Total Pendapatan"]
    return d

def senin_berikutnya(tgl: pd.Timestamp) -> pd.Timestamp:
    wd = tgl.weekday()
    return tgl.normalize() + pd.Timedelta(days=(1 if wd == 6 else 7 - wd))


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
st.sidebar.title("Toko Plastik Ombo Jaya")
st.sidebar.markdown("---")

st.sidebar.header("Impor Data Transaksi")
st.sidebar.info(
    "**Data transaksi yang diimpor** merupakan **hasil ekspor** dari "
    "**aplikasi Point of Sale (POS)** yang digunakan."
)
uploaded_file = st.sidebar.file_uploader("Impor Data Transaksi Harian (CSV)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("Impor Data Konversi")
st.sidebar.info(
    "**File data konversi digunakan untuk menyeragamkan satuan produk yang berbeda** "
    "**sehingga jumlah penjualan (Qty) dapat dihitung secara akurat dan tidak terjadi**"
    " **duplikasi perhitungan.**"
)
st.sidebar.link_button("Lihat Template Master Konversi", "https://docs.google.com/spreadsheets/d/17HTrta68LjSoM6cgf_akwEvskpCmCHhXir-DB69zUlk/edit?usp=sharing")
with open("master_konversi.csv", "rb") as file:
    st.sidebar.download_button(
        label="Download Template Master Konversi",
        data=file,
        file_name="master_konversi.csv",
        mime="text/csv",
        help="Unduh file ini untuk menambahkan produk baru atau memperbarui data konversi satuan."
    )
uploaded_mk = st.sidebar.file_uploader("Impor Data Konversi Satuan (CSV)", type=["csv"])

if uploaded_file is not None and st.session_state.tahap == 1:
    if st.sidebar.button("Proses Data", use_container_width=True, type="primary"):
        st.session_state.tahap = 2

if st.session_state.tahap > 1:
    st.sidebar.markdown("---")
    if st.sidebar.button("Mulai Ulang", use_container_width=True):
        for k in list(st.session_state.keys()):
            st.session_state.pop(k)
        st.rerun()


# ════════════════════════════════════════════════════════════
# TAHAP 1 — Home
# ════════════════════════════════════════════════════════════
if st.session_state.tahap == 1:
    st.title("Sistem Peramalan Permintaan Produk")
    st.info(" ◀ Impor **Data Transaksi & Data Konversi** di sidebar lalu klik **Proses Data**.")
    st.image("Foto Toko.jpeg", use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAHAP 2 — Preprocessing
# ════════════════════════════════════════════════════════════
if st.session_state.tahap == 2:
    st.title("Memproses Data...")
    with st.spinner("Menjalankan preprocessing, Analisis ABC, dan ADI/CV²..."):
        try:
            raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") \
                     else pd.read_excel(uploaded_file)

            if uploaded_mk is not None:
                mk = pd.read_csv(uploaded_mk)
                mk["Nama Barang"]     = mk["Nama Barang"].astype(str).str.strip()
                mk["Dari Satuan"]     = mk["Dari Satuan"].astype(str).str.strip()
                mk["Ke Satuan Final"] = mk["Ke Satuan Final"].astype(str).str.strip()
                mk["Multiplier"]      = pd.to_numeric(
                    mk["Multiplier"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
                mk["_match_satuan"]   = mk["Dari Satuan"].str.lower()
            else:
                try:
                    mk = load_master_konversi("master_konversi.csv")
                    st.sidebar.caption("File konversi ditemukan.")
                except FileNotFoundError:
                    mk = pd.DataFrame(columns=[
                        "Nama Barang", "Dari Satuan", "Ke Satuan Final", "Multiplier", "_match_satuan"
                    ])
                    st.sidebar.warning("File konversi tidak ditemukan.")

            result = run_processing(raw_df, mk)
            if result[0] is None:
                st.session_state.tahap = 1
                st.stop()

            df_weekly_all, hasil_adi, grup_a, grup_b, grup_c, ringkasan = result
            tgl_asli = pd.to_datetime(raw_df["Tanggal Transaksi"], dayfirst=True, errors="coerce").dropna()

            st.session_state.update({
                "df_weekly_all": df_weekly_all, "hasil_adi": hasil_adi,
                "grup_a": grup_a, "grup_b": grup_b, "grup_c": grup_c,
                "ringkasan": ringkasan, "tahap": 3,
                "tgl_transaksi_min": tgl_asli.min(),
                "tgl_transaksi_max": tgl_asli.max(),
            })
            st.rerun()

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state.tahap = 1


# ════════════════════════════════════════════════════════════
# TAHAP 3 — Dashboard ABC + ADI/CV² + Konfigurasi
# ════════════════════════════════════════════════════════════
if st.session_state.tahap >= 3:
    st.title("Dashboard Analisis ABC & Klasifikasi Demand")
    st.success("Preprocessing selesai!")

    df_weekly_all = st.session_state.df_weekly_all
    hasil_adi     = st.session_state.hasil_adi
    grup_a        = st.session_state.grup_a
    grup_b        = st.session_state.grup_b
    grup_c        = st.session_state.grup_c
    ringkasan     = st.session_state.ringkasan

    tgl_min = st.session_state.tgl_transaksi_min or pd.to_datetime(df_weekly_all["Tanggal Transaksi"]).min()
    tgl_max = st.session_state.tgl_transaksi_max or pd.to_datetime(df_weekly_all["Tanggal Transaksi"]).max()
    n_minggu_ds = df_weekly_all["Tanggal Transaksi"].nunique()

    st.info(
        f" **Dataset:** {tgl_min.strftime('%d %b %Y')} — {tgl_max.strftime('%d %b %Y')}  \n"
        f" **Total Minggu:** {n_minggu_ds} minggu  \n"
        f" **Total Produk:** {ringkasan['n_produk']} produk"
    )

    # ── Analisis ABC ──────────────────────────────────────────
    st.subheader("Analisis ABC")
    st.markdown("""
    🔵 **Grup A** : Produk dengan kontribusi pendapatan tertinggi.  
    🟠 **Grup B** : Produk dengan kontribusi pendapatan menengah.  
    ⚫ **Grup C** : Produk dengan kontribusi pendapatan terendah.
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Produk", ringkasan["n_produk"])
    c2.metric(f"Grup A ({ringkasan['n_a']} produk)", f"{ringkasan['persen_produk']['A']:.1f}%",
              f"{ringkasan['persen_pendapatan']['A']:.1f}% pendapatan")
    c3.metric(f"Grup B ({ringkasan['n_b']} produk)", f"{ringkasan['persen_produk']['B']:.1f}%",
              f"{ringkasan['persen_pendapatan']['B']:.1f}% pendapatan")
    c4.metric(f"Grup C ({ringkasan['n_c']} produk)", f"{ringkasan['persen_produk']['C']:.1f}%",
              f"{ringkasan['persen_pendapatan']['C']:.1f}% pendapatan")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Distribusi Pendapatan per Grup**")
        fig = go.Figure(go.Pie(
            labels=["Grup A", "Grup B", "Grup C"],
            values=[ringkasan["persen_pendapatan"]["A"],
                    ringkasan["persen_pendapatan"]["B"],
                    ringkasan["persen_pendapatan"]["C"]],
            hole=0.55, marker_colors=["#2196F3", "#FF9800", "#9E9E9E"],
            textinfo="label+percent"))
        fig.update_layout(height=270, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Distribusi Jumlah Produk per Grup**")
        fig2 = go.Figure(go.Pie(
            labels=["Grup A", "Grup B", "Grup C"],
            values=[ringkasan["n_a"], ringkasan["n_b"], ringkasan["n_c"]],
            hole=0.55,
            marker_colors=["#2196F3", "#FF9800", "#9E9E9E"],
            textinfo="none",
            texttemplate="%{label}<br>%{percent:.0%}",
            hovertemplate="%{label}<br>%{value} produk<br>%{percent:.0%}<extra></extra>"))
        fig2.update_layout(height=270, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.markdown("**Top 10 Produk Grup A**")
        top10 = grup_a.head(10).sort_values("total_harga")
        figb = go.Figure(go.Bar(
            x=top10["total_harga"], y=top10["Nama Barang"], orientation="h",
            marker_color="#2196F3",
            text=top10["total_harga"].apply(lambda x: f"Rp {x/1e6:.1f}Jt"),
            textposition="outside"))
        figb.update_layout(height=270, margin=dict(l=0, r=60, t=10, b=0),
            xaxis=dict(showticklabels=False), yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(figb, use_container_width=True)

    tab_a, tab_b, tab_c = st.tabs(["🔵 Grup A", "🟠 Grup B", "⚫ Grup C"])
    with tab_a: st.dataframe(fmt_tabel_abc(grup_a), use_container_width=True)
    with tab_b: st.dataframe(fmt_tabel_abc(grup_b), use_container_width=True)
    with tab_c: st.dataframe(fmt_tabel_abc(grup_c), use_container_width=True)

    st.markdown("---")

    # ── Klasifikasi Demand ADI / CV² ──────────────────────────
    st.subheader("Klasifikasi Demand — ADI & CV²")
    st.markdown("""
    🔵 **Smooth** : Permintaan terjadi secara rutin dengan jumlah penjualan yang relatif stabil.  
    🟠 **Erratic** : Permintaan terjadi secara rutin, tetapi jumlah penjualannya berfluktuasi tinggi.  
    🟣 **Intermittent** : Permintaan terjadi secara tidak rutin, namun jumlah penjualannya relatif stabil.  
    🔴 **Lumpy** : Permintaan terjadi secara tidak rutin dan jumlah penjualannya berfluktuasi tinggi.
    """)

    WARNA_KAT = {
        "Smooth Demand":       "#2196F3",
        "Erratic Demand":      "#FF9800",
        "Intermittent Demand": "#9C27B0",
        "Lumpy Demand":        "#F44336",
    }
    URUTAN_KAT = ["Smooth Demand", "Erratic Demand", "Intermittent Demand", "Lumpy Demand"]

    kat_df = (
        hasil_adi["Kategori"]
        .value_counts()
        .reindex(URUTAN_KAT)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    kat_df.columns = ["Kategori", "Jumlah"]
    kat_df["Warna"] = kat_df["Kategori"].map(WARNA_KAT)

    fig_kat = go.Figure(go.Bar(
        x=kat_df["Kategori"], y=kat_df["Jumlah"],
        marker_color=kat_df["Warna"],
        text=kat_df["Jumlah"], textposition="outside",
        hovertemplate="%{x}: %{y} produk<extra></extra>",
    ))
    fig_kat.update_layout(
        height=340, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Kategori Demand",
        yaxis_title="Jumlah Produk Grup A",
        yaxis=dict(range=[0, kat_df["Jumlah"].max() * 1.3]),
        bargap=0.4,
    )
    st.plotly_chart(fig_kat, use_container_width=True)

    st.markdown("**Daftar Produk per Kategori Demand**")
    kat_tersedia = [k for k in URUTAN_KAT if k in hasil_adi["Kategori"].values]
    tabs_kat = st.tabs(["All"] + kat_tersedia)

    with tabs_kat[0]:
        disp = hasil_adi.copy()
        disp["_ord"] = disp["Kategori"].map({k: i for i, k in enumerate(URUTAN_KAT)}).fillna(99)
        disp = disp.sort_values(["_ord", "Nama Barang"]).drop(columns="_ord")
        disp = disp[["Nama Barang", "Kategori", "ADI", "CV2",
                     "Total_Weeks", "Active_Weeks", "Zero_Pct", "Mean_Demand", "Std_Demand"]].copy()
        disp.columns = ["Nama Barang", "Kategori", "ADI", "CV²",
                        "Total Minggu", "Aktif", "Zero %", "Rata-rata Demand", "Std Demand"]
        st.caption(f"Semua **{len(disp)}** produk Grup A")
        st.dataframe(disp, use_container_width=True, hide_index=True)

    for i, kat in enumerate(kat_tersedia, 1):
        with tabs_kat[i]:
            subset = hasil_adi[hasil_adi["Kategori"] == kat].sort_values("Nama Barang")[
                ["Nama Barang", "ADI", "CV2", "Total_Weeks", "Active_Weeks",
                 "Zero_Pct", "Mean_Demand", "Std_Demand"]
            ].copy()
            subset.columns = ["Nama Barang", "ADI", "CV²",
                              "Total Minggu", "Aktif", "Zero %", "Rata-rata Demand", "Std Demand"]
            st.caption(f"**{len(subset)}** produk kategori **{kat}**")
            st.dataframe(subset, use_container_width=True, hide_index=True)

    st.markdown("---")
    
    #── Manajemen Persediaan ─────────────────────────────────
    st.subheader("Rekomendasi Persediaan Stok")

    with st.expander("📚 Referensi Teori yang Digunakan", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
    ### 📘 Teori Safety Stock
    **Hadiguna, R. A. (2025)**
    *Pengantar Teknik dan Manajemen Logistik*
    📖 [Buka Buku Google Books](https://books.google.co.id/books?id=aY2bEQAAQBAJ)
    """)
        with col2:
            st.markdown("""
    ### 📄 Teori Reorder Point (ROP)
    **Simarmata, E., Simanjuntak, L. T. B., & Sembiring, A. C. (2026)**
    *Perencanaan Pengendalian Persediaan Produk Beras dan Minyak Goreng Menggunakan Metode Reorder Point dan Regresi Linier*
    📖 [Buka Jurnal](https://doi.org/10.55826/jtmit.v5i2.1737)
    """)
    
    produk_list_0   = sorted(df_weekly_all["Nama Barang"].unique().tolist())
    produk_pilih    = st.selectbox("Pilih Produk:", produk_list_0, key="sel_ram")

    # Ambil satuan produk dari df_weekly_all
    satuan = df_weekly_all.loc[df_weekly_all["Nama Barang"] == produk_pilih, "Satuan"].iloc[0]

    # Ambil baris data safety stock & ROP untuk produk yang dipilih
    data_produk = hasil_adi[hasil_adi["Nama Barang"] == produk_pilih].iloc[0]

    kategori      = data_produk["Kategori"]
    mean_demand   = data_produk["Mean_Demand"]
    safety_stock  = data_produk["Safety_Stock"]        
    reorder_point = data_produk["Reorder_Point"]        

    c1, c2, c3 = st.columns(3)
    c1.metric("Kategori Demand", kategori)
    c2.metric("Safety Stock (Lead Time 2 Minggu)", f"{safety_stock:.0f} {satuan}")
    c3.metric("Reorder Point (ROP)", f"{reorder_point:.0f} {satuan}")

    st.info(
        f"Rata-rata permintaan mingguan produk ini: **{mean_demand:.1f} {satuan}/minggu**. "
        f"Perhitungan dengan lead time pemesanan selama **2 minggu**."
    )

    # ── Rekomendasi berdasarkan Safety Stock & Reorder Point ──────────────────
    st.subheader("Panduan Pengambilan Keputusan Stok")

    st.markdown(
        f"""
        | Kondisi Stok Saat Ini | Status | Rekomendasi |
        |---|---|---|
        | Lebih dari **{reorder_point:.0f} {satuan}** | 🟢 Aman | Belum perlu melakukan pemesanan ulang. |
        | Antara **{safety_stock:.0f} – {reorder_point:.0f} {satuan}** | 🟡 Perhatian | Pantau kondisi stok, untuk perencanaan pemesanan. |
        | Kurang dari **{safety_stock:.0f} {satuan}** | 🔴 Kritis | Disarankan segera lakukan pemesanan ulang. |
        """
    )

    st.info(
        f"💡 **Cara membaca:** Reorder Point ({reorder_point:.0f} {satuan}) adalah **titik di mana pemesanan ulang "
        f"sebaiknya mulai diperhatikan**. Safety Stock ({safety_stock:.0f} {satuan}) adalah **cadangan minimum yang "
        f"seharusnya ada untuk menghadapi kondisi permintaan tak terduga**. Perhitungan ini dilakukan dari pola "
        f"histori penjualan produk dan **dapat disesuaikan oleh kondisi stok dan biaya.**"
    )

    st.markdown("---")
    

    # ── Konfigurasi Peramalan ─────────────────────────────────
    st.subheader("Konfigurasi Peramalan")

    col_m, col_k, col_w = st.columns(3)

    with col_m:
        pilih_model = st.selectbox("Pilih Model:", ["XGBoost", "TabNet"])

    with col_k:
        KATEGORI_URUT = ["All", "Smooth Demand", "Erratic Demand", "Intermittent Demand", "Lumpy Demand"]
        pilih_kategori = st.selectbox(
            "Pilih Kategori Demand yang Diramal:",
            options=KATEGORI_URUT,
            format_func=lambda k: (
                f"All — semua {len(hasil_adi)} produk Grup A" if k == "All"
                else f"{k} — {int((hasil_adi['Kategori'] == k).sum())} produk"
            ),
            help="Pilih kategori ADI/CV². Setiap kategori menggunakan model yang dilatih khusus."
        )
        n_prod_dipilih = len(hasil_adi) if pilih_kategori == "All" \
                         else int((hasil_adi["Kategori"] == pilih_kategori).sum())
        st.caption(f"➡ **{n_prod_dipilih} produk** akan diramal")

    with col_w:
        senin_min = senin_berikutnya(tgl_max)
        st.markdown("**Rentang Prediksi**")
        st.caption(
            f"Dimulai Senin: **{senin_min.strftime('%d %b %Y')}**  \n"
        )
        n_minggu = st.slider("Jumlah Minggu ke Depan:", 1, 12, 4)
        opsi_minggu_custom = [
            (senin_min + pd.Timedelta(weeks=i),
             senin_min + pd.Timedelta(weeks=i, days=6))
            for i in range(n_minggu)
        ]
        st.info(
            f"Meramal **{n_minggu} minggu**:  \n"
            f"**{opsi_minggu_custom[0][0].strftime('%d %b %Y')}** — "
            f"**{opsi_minggu_custom[-1][1].strftime('%d %b %Y')}**"
        )

    if st.button("Jalankan Peramalan", type="primary"):
        st.session_state.update({
            "model_nama": pilih_model, "kategori_dipilih": pilih_kategori,
            "opsi_minggu": opsi_minggu_custom, "df_weekly_fit": None, 
            "X_val": None, "y_val": None,
            "X_test": None, "y_test": None, 
            "y_pred_val": None,
            "mae": None,"r2": None,
            "tahap": 4,
        })
        st.rerun()


# ════════════════════════════════════════════════════════════
# TAHAP 4 — Hasil Peramalan
# ════════════════════════════════════════════════════════════
if st.session_state.tahap == 4:
    st.title("Dashboard Hasil Peramalan Penjualan")

    df_weekly_all    = st.session_state.df_weekly_all
    hasil_adi        = st.session_state.hasil_adi
    grup_a           = st.session_state.grup_a
    grup_b           = st.session_state.grup_b
    grup_c           = st.session_state.grup_c
    model_nama       = st.session_state.model_nama
    kategori_dipilih = st.session_state.kategori_dipilih
    opsi_minggu      = st.session_state.opsi_minggu
    rentang          = len(opsi_minggu)

    # ── Filter produk sesuai kategori ────────────────────────
    if kategori_dipilih == "All":
        produk_dipilih = set(hasil_adi["Nama Barang"].tolist())
    else:
        produk_dipilih = set(
            hasil_adi[hasil_adi["Kategori"] == kategori_dipilih]["Nama Barang"].tolist()
        )

    # ── Build fitur (sekali saja) ─────────────────────────────
    if st.session_state.df_weekly_fit is None:
        with st.spinner("Membangun fitur..."):
            df_fit = df_weekly_all[df_weekly_all["Nama Barang"].isin(produk_dipilih)].copy()
            df_fit = buat_fitur(df_fit)
            X_train, y_train, X_val, y_val, X_test, y_test = split_data_70_15_15(df_fit, train_size=0.70,val_size=0.15)
            batas_train = hitung_batas_train(df_fit, train_size=0.70)
            fitur_cols = get_fitur_cols(X_val.columns)
            st.session_state.update({
                "df_weekly_fit": df_fit, "X_val": X_val, 
                "y_val": y_val, "X_test": X_test,
                "y_test": y_test, "batas_train": batas_train,
                "fitur_cols": fitur_cols,
            })

    df_weekly_fit = st.session_state.df_weekly_fit
    X_eval = st.session_state.X_test
    y_eval = st.session_state.y_test
    batas_train   = st.session_state.batas_train
    fitur_cols    = st.session_state.fitur_cols

    # ── Load model ────────────────────────────────────────────
    with st.spinner(f"Memuat model {model_nama} – {kategori_dipilih}..."):
        model, err = load_model(model_nama, kategori_dipilih)
        if err:
            st.error(err)
            st.stop()

    # ── Evaluasi ──────────────────────────────────────────────
    if st.session_state.mae is None:
        with st.spinner("Evaluasi model pada data testing..."):
            y_pred_test = np.maximum(predict(model, X_eval[fitur_cols], model_nama), 0)
            y_true     = y_eval.values
            mae  = float(np.mean(np.abs(y_pred_test - y_true)))
            r2   = float(1 - np.sum((y_true - y_pred_test) ** 2) /
                         (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8))
            st.session_state.update({"y_pred_val": y_pred_test, "mae": mae,  "r2": r2})
    else:
        y_pred_test = st.session_state.y_pred_val
        mae, r2 = st.session_state.mae, st.session_state.r2

    st.success(f"Model **{model_nama}** – Kategori **{kategori_dipilih}** berhasil dievaluasi.")

    # ── KPI ───────────────────────────────────────────────────
    tgl_min_fit = pd.to_datetime(df_weekly_fit["Tanggal Transaksi"]).min()
    tgl_max_fit = pd.to_datetime(df_weekly_fit["Tanggal Transaksi"]).max()
    n_produk    = df_weekly_fit["Nama Barang"].nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Model",          model_nama)
    c2.metric("Kategori",       kategori_dipilih)
    c3.metric("Produk Diramal", n_produk)
    c4.metric("MAE",            f"{mae:.2f}")
    c5.metric("R²",             f"{r2:.2f}")

    if r2 >= 0.75:
        st.success(f"R² = **{r2:.2f}** — Model dikategorikan **kuat**.")
    elif r2 >= 0.50:
        st.info(f"R² = **{r2:.2f}** — Model dikategorikan **sedang (moderate)**.")
    elif r2 >= 0.25:
        st.warning(f"R² = **{r2:.2f}** — Model dikategorikan **lemah**.")
    else:
        st.error(f"R² = **{r2:.2f}** — Model dikategorikan **sangat lemah / tidak layak dijadikan dasar interpretasi**.")

    # ── Peramalan per Produk ──────────────────────────────────
    st.subheader("Peramalan per Produk")
    produk_list    = sorted(df_weekly_fit["Nama Barang"].unique().tolist())
    produk_ramalan = st.selectbox("Pilih produk:", produk_list, key="sel_ram_2")
    df_pr    = df_weekly_fit[df_weekly_fit["Nama Barang"] == produk_ramalan].sort_values("Tanggal Transaksi")
    satuan_r = df_pr["Satuan"].iloc[-1]
    inp_r    = df_pr[fitur_cols].iloc[[-1]].copy()

    tgl_pred_list, qty_pred_list = [], []
    for i, (s, e) in enumerate(opsi_minggu):
        p = int(round(max(0.0, float(predict(model, inp_r, model_nama)[0]))))
        tgl_pred_list.append(s + pd.Timedelta(days=3))
        qty_pred_list.append(p)
        inp_r = roll_forward(inp_r, p, s + pd.Timedelta(days=10))

    fig_fw = go.Figure()
    fig_fw.add_trace(go.Scatter(
        x=df_pr.tail(16)["Tanggal Transaksi"], y=df_pr.tail(16)[TARGET],
        mode="lines+markers", name="Histori Aktual", line=dict(color="#1f77b4", width=2)))
    fig_fw.add_trace(go.Scatter(
        x=tgl_pred_list, y=qty_pred_list,
        mode="lines+markers", name="Prediksi (ML)",
        line=dict(color="#ff7f0e", width=2, dash="dash"), marker=dict(size=9, symbol="diamond")))
    fig_fw.update_layout(
        height=340, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
        xaxis_title="Tanggal", yaxis_title=f"Qty ({satuan_r})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_fw, use_container_width=True)

    rows_detail = []
    for i, ((s, e), qty) in enumerate(zip(opsi_minggu, qty_pred_list), 1):
        rows_detail.append({
            "Minggu": f"Minggu {i}",
            "Periode (Senin – Minggu)": f"{s.strftime('%d %b %Y')} – {e.strftime('%d %b %Y')}",
            "Prediksi Qty": qty,
            "Satuan": satuan_r,
        })
    st.dataframe(pd.DataFrame(rows_detail), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Rekap Semua Produk ────────────────────────────────────
    st.subheader(f"Rekap Kebutuhan Stok {rentang} Minggu ke Depan")
    st.caption("  |  ".join(
        f"Minggu {i+1}: {s.strftime('%d %b')}–{e.strftime('%d %b %Y')}"
        for i, (s, e) in enumerate(opsi_minggu)
    ))

    with st.spinner("Menghitung prediksi semua produk..."):
        rows = []
        for produk in sorted(produk_dipilih):
            df_p2 = df_weekly_fit[df_weekly_fit["Nama Barang"] == produk].sort_values("Tanggal Transaksi")
            if df_p2.empty:
                continue
            satuan = df_p2["Satuan"].iloc[-1]
            inp    = df_p2[fitur_cols].iloc[[-1]].copy()

            preds = []
            for idx in range(rentang):
                pv = int(round(max(0.0, float(predict(model, inp, model_nama)[0]))))
                preds.append(pv)
                tgl_n = (opsi_minggu[idx + 1][0] if idx + 1 < rentang
                         else opsi_minggu[-1][0] + pd.Timedelta(weeks=1))
                inp = roll_forward(inp, pv, tgl_n + pd.Timedelta(days=3))

            row = {"Nama Barang": produk, "Satuan": satuan}
            for i, ((s, e), pv) in enumerate(zip(opsi_minggu, preds), 1):
                row[f"Minggu {i} ({s.strftime('%d/%m')}–{e.strftime('%d/%m')})"] = pv
            row["Total Prediksi"] = sum(preds)
            rows.append(row)

        df_rekap = pd.DataFrame(rows)

    st.dataframe(df_rekap, use_container_width=True)

    # ── Download Excel ────────────────────────────────────────
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_rekap.to_excel(writer, index=False, sheet_name="Rekap Peramalan")
        pd.DataFrame({
            "Metrik": ["Model", "Kategori Demand", "Produk Diramal", "Minggu Prediksi",
                       "MAE", "R²",
                       "Periode Dataset", "Batas Train", "Tanggal Export"],
            "Nilai": [
                model_nama, kategori_dipilih, n_produk, rentang,
                f"{mae:.4f}", f"{r2:.4f}",
                f"{tgl_min_fit.strftime('%d %b %Y')} – {tgl_max_fit.strftime('%d %b %Y')}",
                batas_train.strftime("%d %b %Y"),
                pd.Timestamp.today().strftime("%d %b %Y %H:%M"),
            ]
        }).to_excel(writer, index=False, sheet_name="Evaluasi Model")
        pd.concat([
            grup_a.assign(Grup="A"), grup_b.assign(Grup="B"), grup_c.assign(Grup="C")
        ]).rename(columns={"total_harga": "Total Pendapatan"}).to_excel(
            writer, index=False, sheet_name="Analisis ABC")
        hasil_adi.to_excel(writer, index=False, sheet_name="Klasifikasi ADI-CV2")

    buffer.seek(0)
    st.download_button(
        "Unduh Rekap Stok (Excel)", data=buffer,
        file_name=f"Peramalan_{model_nama}_{kategori_dipilih}_{pd.Timestamp.today().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True, type="primary"
    )
    st.caption("4 sheet: **Rekap Peramalan** | **Evaluasi Model** | **Analisis ABC** | **Klasifikasi ADI-CV2**")
    st.markdown("---")
    
    st.markdown("---")
    col_back, _ = st.columns([1, 4])
    with col_back:
        if st.button("◀ Kembali ke Konfigurasi"):
            st.session_state.update({
                "tahap": 3, "df_weekly_fit": None,
                "mae": None, "y_pred_val": None
            })
            st.rerun()
