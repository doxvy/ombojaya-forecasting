import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import joblib
from datetime import date, timedelta

from processing import (
    run_processing,
    load_master_konversi,
    filter_produk_peramalan,
    buat_fitur,
    split_train_val,
    hitung_batas_train,
    info_split,
    hitung_minggu_prediksi,
    FEATURES,
    TARGET,
)

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(page_title="Sistem Peramalan Ombo Jaya", layout="wide", page_icon="📦")

for _k, _v in {
    'tahap': 1,
    'df_weekly_all': None,
    'grup_a': None, 'grup_b': None, 'grup_c': None,
    'ringkasan': None, 'df_zero_all': None,
    'df_weekly_fit': None,
    'X_train': None, 'y_train': None, 'X_val': None, 'y_val': None,
    'batas_train': None,
    'model_nama': None, 'opsi_minggu': None, 'zero_threshold': 100,
    'y_pred_val': None, 'fitur_cols': None,
    'mae': None, 'rmse': None, 'r2': None, 'mape': None, 'wmape': None,
    'tgl_transaksi_min': None, 'tgl_transaksi_max': None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

PATH_MODEL_XGBOOST = "xgboost.pkl"
PATH_MODEL_TABNET  = "tabnet.zip"

# ============================================================
# FEATURES — sama untuk XGBoost dan TabNet (tanpa ID_Barang)
# ID_Barang tidak dipakai agar konsisten dengan notebook TabNet
# (Cell 53) dan agar kedua model menghasilkan hasil yang setara.
# ============================================================
def get_fitur_cols(model_nama: str, available_cols) -> list:
    """Pilih kolom fitur — sama untuk XGBoost & TabNet (tanpa ID_Barang)."""
    return [f for f in FEATURES if f != "ID_Barang" and f in available_cols]


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def predict(model, X: pd.DataFrame, model_nama: str) -> np.ndarray:
    if model_nama == 'TabNet':
        return model.predict(X.values.astype(np.float32)).ravel()
    return model.predict(X)


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

    lag_vals = {c: float(inp[c].values[0]) for c in
                ["Lag_1_Minggu","Lag_2_Minggu","Lag_3_Minggu",
                 "Lag_4_Minggu","Lag_8_Minggu","Lag_12_Minggu"] if c in inp.columns}

    def _win(n):
        b = [lag_vals.get("Lag_1_Minggu",0), lag_vals.get("Lag_2_Minggu",0),
             lag_vals.get("Lag_3_Minggu",0), lag_vals.get("Lag_4_Minggu",0)]
        if n > 4: b += [lag_vals.get("Lag_8_Minggu",  b[-1])] * (min(n,8) - 4)
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


Z_SERVICE = 1.65  # 95% tingkat layanan (service level)

def hitung_safety_stock(std_p: float) -> int:
    """
    Safety Stock = Z × σ  (Z=1.65, service level 95%).
    Berdiri sendiri — tidak ditambahkan ke prediksi.
    Dibulatkan ke bilangan bulat (round).
    """
    return int(round(Z_SERVICE * std_p))


@st.cache_resource
def load_model(nama: str):
    import os
    try:
        if nama == 'TabNet':
            from pytorch_tabnet.tab_model import TabNetRegressor
            m = TabNetRegressor()
            m.load_model(PATH_MODEL_TABNET)
            return m, None
        return joblib.load(PATH_MODEL_XGBOOST), None
    except FileNotFoundError:
        path = PATH_MODEL_TABNET if nama == 'TabNet' else PATH_MODEL_XGBOOST
        cwd  = os.getcwd()
        return None, (
            f"File model **'{path}'** tidak ditemukan.  \n"
            f"Working directory saat ini: `{cwd}`  \n"
            f"Pastikan file model berada di folder yang sama dengan `app.py`, "
            f"Pastikan file tersebut ada di folder yang sama dengan app.py."
        )
    except Exception as e:
        return None, str(e)


def fmt_tabel(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    d.index += 1
    d['total_harga'] = d['total_harga'].apply(lambda x: f"Rp {x:,.0f}")
    d.columns = ['Nama Barang', 'Total Pendapatan']
    return d


def senin_berikutnya(tgl: pd.Timestamp) -> pd.Timestamp:
    wd = tgl.weekday()
    days = 1 if wd == 6 else (7 - wd)
    return tgl.normalize() + pd.Timedelta(days=days)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🛍️ Toko Ombo Jaya")
st.sidebar.markdown("---")
st.sidebar.header("Impor Data Transaksi")

uploaded_file = st.sidebar.file_uploader("Upload Data Transaksi (CSV / Excel)", type=['csv','xlsx'])

st.sidebar.markdown("---")
st.sidebar.header("Master Konversi Satuan")
uploaded_mk = st.sidebar.file_uploader(
    "Upload master_konversi.csv (opsional)", type=['csv'],
    help="Jika tidak diupload, dicari otomatis di folder yang sama."
)

if uploaded_file is not None and st.session_state.tahap == 1:
    if st.sidebar.button("🔍 Proses Data", use_container_width=True, type="primary"):
        st.session_state.tahap = 2

if st.session_state.tahap > 1:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Mulai Ulang / Upload Data Baru", use_container_width=True):
        for k in list(st.session_state.keys()):
            st.session_state.pop(k)
        st.rerun()


# ============================================================
# TAHAP 1 — Sambutan
# ============================================================
if st.session_state.tahap == 1:
    st.title("Sistem Manajemen Stok & Peramalan Cerdas")
    st.info("👈 Upload **Data Transaksi** di sidebar lalu klik **Proses Data**.")
    st.markdown(
        '<img src="https://images.unsplash.com/photo-1553413077-190dd305871c'
        '?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" '
        'style="width:100%;border-radius:12px;">', unsafe_allow_html=True
    )


# ============================================================
# TAHAP 2 — Preprocessing
# ============================================================
if st.session_state.tahap == 2:
    st.title("⚙️ Memproses Data...")
    with st.spinner("Menjalankan preprocessing & Analisis ABC..."):
        try:
            raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') \
                     else pd.read_excel(uploaded_file)

            if uploaded_mk is not None:
                mk = pd.read_csv(uploaded_mk)
                mk["Nama Barang"]     = mk["Nama Barang"].astype(str).str.strip()
                mk["Dari Satuan"]     = mk["Dari Satuan"].astype(str).str.strip()
                mk["Ke Satuan Final"] = mk["Ke Satuan Final"].astype(str).str.strip()
                mk["Multiplier"]      = pd.to_numeric(
                    mk["Multiplier"].astype(str).str.replace(",",".",regex=False), errors="coerce")
                mk["_match_satuan"]   = mk["Dari Satuan"].str.lower()
            else:
                try:
                    mk = load_master_konversi("master_konversi.csv")
                    st.sidebar.caption("✅ master_konversi.csv ditemukan di folder lokal.")
                except FileNotFoundError:
                    mk = pd.DataFrame(columns=["Nama Barang","Dari Satuan","Ke Satuan Final","Multiplier","_match_satuan"])
                    st.sidebar.warning("⚠️ master_konversi.csv tidak ditemukan.")

            result = run_processing(raw_df, mk)
            if result[0] is None:
                st.session_state.tahap = 1
                st.stop()

            df_weekly_all, grup_a, grup_b, grup_c, ringkasan, df_zero_all = result
            st.session_state.update({
                'df_weekly_all': df_weekly_all, 'grup_a': grup_a,
                'grup_b': grup_b, 'grup_c': grup_c,
                'ringkasan': ringkasan, 'df_zero_all': df_zero_all,
                'tahap': 3,
            })
            # Simpan tanggal transaksi asli (sebelum diubah ke weekly)
            tgl_asli = pd.to_datetime(raw_df['Tanggal Transaksi'], dayfirst=True, errors='coerce').dropna()
            st.session_state['tgl_transaksi_min'] = tgl_asli.min()
            st.session_state['tgl_transaksi_max'] = tgl_asli.max()
            st.rerun()
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            import traceback; st.code(traceback.format_exc())
            st.session_state.tahap = 1


# ============================================================
# TAHAP 3 — Dashboard ABC & Konfigurasi
# ============================================================
if st.session_state.tahap >= 3:
    st.title("📊 Dashboard Analisis ABC")
    st.success("✅ Preprocessing selesai!")

    df_weekly_all = st.session_state.df_weekly_all
    grup_a        = st.session_state.grup_a
    grup_b        = st.session_state.grup_b
    grup_c        = st.session_state.grup_c
    ringkasan     = st.session_state.ringkasan
    df_zero_all   = st.session_state.df_zero_all

    # Tanggal asli (sebelum resampling ke weekly) untuk ditampilkan
    tgl_min_asli = st.session_state.get('tgl_transaksi_min')
    tgl_max_asli = st.session_state.get('tgl_transaksi_max')
    tgl_min_weekly = pd.to_datetime(df_weekly_all['Tanggal Transaksi']).min()
    tgl_max_weekly = pd.to_datetime(df_weekly_all['Tanggal Transaksi']).max()
    n_minggu_ds    = df_weekly_all['Tanggal Transaksi'].nunique()

    # Gunakan tanggal asli jika tersedia, fallback ke weekly
    tgl_min = tgl_min_asli if tgl_min_asli is not None else tgl_min_weekly
    tgl_max = tgl_max_asli if tgl_max_asli is not None else tgl_max_weekly

    st.info(
        f"📅 **Dataset:** {tgl_min.strftime('%d %b %Y')} — {tgl_max.strftime('%d %b %Y')}  \n"
        f"📆 **Total Minggu:** {n_minggu_ds} minggu  \n"
        f"📦 **Total Produk (SKU):** {ringkasan['n_produk']} produk"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrik ABC ────────────────────────────────────────────
    st.subheader("Ringkasan Portofolio Produk")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Produk", ringkasan['n_produk'])
    c2.metric(f"Grup A ({ringkasan['n_a']} produk)", f"{ringkasan['persen_produk']['A']:.1f}%",
              f"{ringkasan['persen_pendapatan']['A']:.1f}% pendapatan")
    c3.metric(f"Grup B ({ringkasan['n_b']} produk)", f"{ringkasan['persen_produk']['B']:.1f}%",
              f"{ringkasan['persen_pendapatan']['B']:.1f}% pendapatan")
    c4.metric(f"Grup C ({ringkasan['n_c']} produk)", f"{ringkasan['persen_produk']['C']:.1f}%",
              f"{ringkasan['persen_pendapatan']['C']:.1f}% pendapatan")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ABC ────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Distribusi Pendapatan per Grup**")
        fig_rev = go.Figure(go.Pie(
            labels=['Grup A','Grup B','Grup C'],
            values=[ringkasan['persen_pendapatan']['A'],
                    ringkasan['persen_pendapatan']['B'],
                    ringkasan['persen_pendapatan']['C']],
            hole=0.55, marker_colors=['#2196F3','#FF9800','#9E9E9E'],
            textinfo='label+percent'))
        fig_rev.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        st.plotly_chart(fig_rev, use_container_width=True)

    with col2:
        st.markdown("**Distribusi Jumlah Produk per Grup**")
        fig_qty = go.Figure(go.Pie(
            labels=['Grup A','Grup B','Grup C'],
            values=[ringkasan['n_a'], ringkasan['n_b'], ringkasan['n_c']],
            hole=0.55, marker_colors=['#2196F3','#FF9800','#9E9E9E'],
            textinfo='label+percent'))
        fig_qty.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        st.plotly_chart(fig_qty, use_container_width=True)

    with col3:
        st.markdown("**Top 10 Produk Grup A (Pendapatan Tertinggi)**")
        top10 = grup_a.head(10).sort_values('total_harga')
        fig_bar = go.Figure(go.Bar(
            x=top10['total_harga'], y=top10['Nama Barang'], orientation='h',
            marker_color='#2196F3',
            text=top10['total_harga'].apply(lambda x: f"Rp {x/1e6:.1f}Jt"),
            textposition='outside'))
        fig_bar.update_layout(height=300, margin=dict(l=0,r=60,t=10,b=0),
                              xaxis=dict(showticklabels=False), yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Detail Produk per Grup")
    tab_a, tab_b, tab_c = st.tabs(["🔵 Grup A (Prioritas)","🟠 Grup B","⚫ Grup C"])
    with tab_a: st.dataframe(fmt_tabel(grup_a), use_container_width=True)
    with tab_b: st.dataframe(fmt_tabel(grup_b), use_container_width=True)
    with tab_c: st.dataframe(fmt_tabel(grup_c), use_container_width=True)

    st.markdown("---")

    # ── FIX #2: Analisis Zero Demand — bar chart jumlah produk per threshold ──
    st.subheader("📉 Analisis Zero Demand Produk Grup A")
    st.caption(
        "Menunjukkan berapa banyak produk Grup A yang akan diramal pada setiap nilai threshold. "
        "Semakin ketat threshold, semakin sedikit produk yang lolos."
    )

    produk_a_set = set(grup_a['Nama Barang'].tolist())
    df_zero_a    = df_zero_all[df_zero_all['Nama Barang'].isin(produk_a_set)].copy()
    pct          = df_zero_a['Zero_Percentage']

    THRESHOLDS = [5, 8, 10, 20, 30, 50, 100]
    n_per_thr  = [len(df_zero_a[pct <= t]) for t in THRESHOLDS]
    label_thr  = [f"≤{t}%" if t < 100 else "100%\n(Semua)" for t in THRESHOLDS]

    fig_bar_zero = go.Figure(go.Bar(
        x=label_thr, y=n_per_thr,
        marker_color=['#2196F3' if t <= 20 else '#90CAF9' for t in THRESHOLDS],
        text=n_per_thr, textposition='outside',
    ))
    fig_bar_zero.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Threshold Zero Percentage",
        yaxis_title="Jumlah Produk Grup A",
        yaxis=dict(range=[0, max(n_per_thr) * 1.2]),
    )
    st.plotly_chart(fig_bar_zero, use_container_width=True)

    with st.expander("Lihat tabel detail zero demand Grup A"):
        df_zd = df_zero_a.copy()
        df_zd['Zero_Percentage'] = df_zd['Zero_Percentage'].round(1)
        df_zd.columns = ['Nama Barang','Total Minggu','Minggu Nol','Zero %']
        st.dataframe(df_zd.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Konfigurasi Peramalan ─────────────────────────────────
    st.subheader("⚙️ Konfigurasi Peramalan")

    col_m, col_z, col_w = st.columns(3)

    with col_m:
        pilih_model = st.selectbox("Pilih Model:", ["XGBoost", "TabNet"])

    with col_z:
        zero_threshold = st.selectbox(
            "Filter Zero Demand (Grup A):",
            options=[100, 50, 30, 20, 10, 8, 5],
            format_func=lambda x: "100% — semua Grup A" if x == 100 else f"≤ {x}% zero weeks",
            index=1,  # default 50%
        )
        n_diramal = len(df_zero_a) if zero_threshold >= 100 else len(df_zero_a[pct <= zero_threshold])
        st.caption(f"➡ **{n_diramal} produk** akan diramal dari {len(produk_a_set)} Grup A")

    with col_w:
        st.markdown("**Rentang Prediksi**")

        senin_min = senin_berikutnya(tgl_max)
        st.caption(
            f"Tanggal data terakhir: **{tgl_max.strftime('%d %b %Y')}**  \n"
        )

        n_minggu = st.slider("Jumlah Minggu ke Depan:", min_value=1, max_value=16, value=4)

        opsi_minggu_custom = [
            (senin_min + pd.Timedelta(weeks=i),
             senin_min + pd.Timedelta(weeks=i, days=6))
            for i in range(n_minggu)
        ]
        s_tgl = opsi_minggu_custom[0][0]
        e_tgl = opsi_minggu_custom[-1][1]
        st.info(
            f"📅 Meramal **{n_minggu} minggu**:  \n"
            f"**{s_tgl.strftime('%d %b %Y')}** — **{e_tgl.strftime('%d %b %Y')}**"
        )

    if st.button("🚀 Jalankan Peramalan", type="primary"):
        st.session_state.update({
            'model_nama': pilih_model,
            'opsi_minggu': opsi_minggu_custom,
            'zero_threshold': zero_threshold,
            'df_weekly_fit': None,
            'y_pred_val': None,
            'mae': None,
            'tahap': 4,
        })
        st.rerun()


# ============================================================
# TAHAP 4 — Dashboard Hasil Peramalan
# ============================================================
if st.session_state.tahap == 4:
    st.title("📈 Dashboard Hasil Peramalan Stok")

    df_weekly_all  = st.session_state.df_weekly_all
    grup_a         = st.session_state.grup_a
    grup_b         = st.session_state.grup_b
    grup_c         = st.session_state.grup_c
    df_zero_all    = st.session_state.df_zero_all
    model_nama     = st.session_state.model_nama
    opsi_minggu    = st.session_state.opsi_minggu
    zero_threshold = st.session_state.zero_threshold
    rentang        = len(opsi_minggu)

    # ── A. Filter + Feature Engineering ──────────────────────
    if st.session_state.df_weekly_fit is None:
        with st.spinner("Memfilter produk & membangun fitur..."):
            produk_a_list = grup_a['Nama Barang'].tolist()
            df_wf = filter_produk_peramalan(df_weekly_all, produk_a_list, zero_threshold, df_zero_all)
            df_wf = buat_fitur(df_wf)
            X_train, y_train, X_val, y_val = split_train_val(df_wf, test_size=0.20)
            batas_train = hitung_batas_train(df_wf, test_size=0.20)
            # FIX #1: fitur sesuai model
            fitur_cols = get_fitur_cols(model_nama, X_val.columns)
            st.session_state.update({
                'df_weekly_fit': df_wf, 'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val, 'batas_train': batas_train,
                'fitur_cols': fitur_cols,
            })

    df_weekly_fit = st.session_state.df_weekly_fit
    X_val         = st.session_state.X_val
    y_val         = st.session_state.y_val
    batas_train   = st.session_state.batas_train
    fitur_cols    = st.session_state.fitur_cols

    # ── B. Load Model ──────────────────────────────────────────
    with st.spinner(f"Memuat model {model_nama}..."):
        model, err = load_model(model_nama)
        if err:
            st.error(err); st.stop()

    # ── C. Evaluasi ────────────────────────────────────────────
    if st.session_state.mae is None:
        with st.spinner("Mengevaluasi model pada data validasi..."):
            # FIX #1: pakai fitur_cols yang sudah disesuaikan per model
            y_pred_val = np.maximum(predict(model, X_val[fitur_cols], model_nama), 0)
            y_true     = y_val.values
            mae   = float(np.mean(np.abs(y_pred_val - y_true)))
            rmse  = float(np.sqrt(np.mean((y_pred_val - y_true)**2)))
            r2    = float(1 - np.sum((y_true - y_pred_val)**2) /
                          (np.sum((y_true - np.mean(y_true))**2) + 1e-8))
            mape  = float(np.mean(np.abs((y_true - y_pred_val) /
                          np.where(y_true == 0, 1, y_true))) * 100)
            wmape = float(np.sum(np.abs(y_true - y_pred_val)) /
                          (np.sum(np.abs(y_true)) + 1e-8) * 100)
            st.session_state.update({
                'y_pred_val': y_pred_val, 'mae': mae, 'rmse': rmse,
                'r2': r2, 'mape': mape, 'wmape': wmape,
            })
    else:
        y_pred_val = st.session_state.y_pred_val
        mae, rmse, r2, mape, wmape = (
            st.session_state.mae, st.session_state.rmse, st.session_state.r2,
            st.session_state.mape, st.session_state.wmape)

    st.success(f"Model **{model_nama}** berhasil dievaluasi.")

    # ── KPI ────────────────────────────────────────────────────
    tgl_min_fit = pd.to_datetime(df_weekly_fit['Tanggal Transaksi']).min()
    tgl_max_fit = pd.to_datetime(df_weekly_fit['Tanggal Transaksi']).max()
    n_produk    = df_weekly_fit['Nama Barang'].nunique()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Model",           model_nama)
    c2.metric("Rentang",         f"{rentang} Minggu")
    c3.metric("Produk Diramal",  n_produk)
    c4.metric("Skor Evaluasi Model (R²)",              f"{r2:.4f}")

    if r2 >= 0.8:
        st.success(f"🎯 Skor Evaluasi Model (R²)  =  **{r2:.4f}** — Model sangat baik")
    elif r2 >= 0.6:
        st.warning(f"⚠️ Skor Evaluasi Model (R²)  =  **{r2:.4f}** — Model cukup baik, masih ada ruang peningkatan.")
    else:
        st.error(f"❌ Skor Evaluasi Model (R²)  =  **{r2:.4f}** — Performa model perlu ditinjau kembali.")

    split_info = info_split(df_weekly_fit, test_size=0.20)
    st.info(
        f"🏋️ **Train:** {tgl_min_fit.strftime('%d %b %Y')} — "
        f"{split_info['cutoff_date'].strftime('%d %b %Y')} "
        f"({split_info['n_minggu_train']} minggu, {split_info['n_rows_train']} baris)  \n"
        f"🔎 **Validasi:** {split_info['val_start'].strftime('%d %b %Y')} — "
        f"{tgl_max_fit.strftime('%d %b %Y')} "
        f"({split_info['n_minggu_val']} minggu, {split_info['n_rows_val']} baris)"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Grafik Aktual vs Prediksi ──────────────────────────────
    st.subheader("Visualisasi Aktual vs Prediksi (Data Validasi)")

    df_val_plot             = df_weekly_fit.loc[X_val.index].copy()
    df_val_plot['Prediksi'] = y_pred_val
    produk_list             = sorted(df_weekly_fit['Nama Barang'].unique())
    produk_pilihan          = st.selectbox("Pilih produk:", produk_list, key="sel_validasi")

    df_p = df_val_plot[df_val_plot['Nama Barang'] == produk_pilihan].sort_values('Tanggal Transaksi')
    if df_p.empty:
        st.info("Produk ini tidak ada di data validasi.")
    else:
        satuan_v = df_p['Satuan'].iloc[0]
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=df_p['Tanggal Transaksi'], y=df_p[TARGET],
            mode='lines+markers', name='Aktual', line=dict(color='#1f77b4', width=2)))
        fig_v.add_trace(go.Scatter(x=df_p['Tanggal Transaksi'], y=df_p['Prediksi'],
            mode='lines+markers', name='Prediksi', line=dict(color='#ff7f0e', width=2, dash='dot')))
        fig_v.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0), hovermode='x unified',
            xaxis_title='Tanggal', yaxis_title=f'Qty ({satuan_v})',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        st.plotly_chart(fig_v, use_container_width=True)

    st.markdown("---")

    # ── Peramalan ke Depan per Produk ──────────────────────────
    st.subheader("🔮 Peramalan ke Depan per Produk")

    def _std_produk(nama):
        """
        Std deviasi demand mingguan produk dari data weekly asli (df_weekly_all).
        Collapse multi-satuan dulu agar std tidak dobel karena 2 satuan per produk.
        Sumber: df_weekly_all (Qty asli, sebelum feature engineering).
        """
        grp = df_weekly_all[df_weekly_all['Nama Barang'] == nama].copy()
        # Collapse satuan: jumlahkan Qty per minggu
        grp_collapsed = (
            grp.groupby('Tanggal Transaksi', as_index=False)['Qty'].sum()
        )
        d = grp_collapsed['Qty']
        return float(d.std()) if len(d) > 1 else 0.0

    produk_ramalan = st.selectbox("Pilih produk:", produk_list, key="sel_ramalan")
    df_pr    = df_weekly_fit[df_weekly_fit['Nama Barang'] == produk_ramalan].sort_values('Tanggal Transaksi')
    satuan_r = df_pr['Satuan'].iloc[-1]
    inp_r    = df_pr[fitur_cols].iloc[[-1]].copy()

    std_prod    = _std_produk(produk_ramalan)
    safety_prod = hitung_safety_stock(std_prod)  # tetap per minggu (berdiri sendiri)

    tgl_pred_list, qty_pred_list = [], []
    for i, (s, e) in enumerate(opsi_minggu):
        tgl_tengah = s + pd.Timedelta(days=3)
        p = int(round(max(0.0, float(predict(model, inp_r, model_nama)[0]))))
        tgl_pred_list.append(tgl_tengah)
        qty_pred_list.append(p)
        tgl_next = s + pd.Timedelta(days=3+7)
        inp_r = roll_forward(inp_r, p, tgl_next)

    # Safety stock sebagai garis horizontal referensi di grafik
    safety_line = [safety_prod] * len(tgl_pred_list)

    fig_fw = go.Figure()
    df_hist = df_pr.tail(16)
    fig_fw.add_trace(go.Scatter(x=df_hist['Tanggal Transaksi'], y=df_hist[TARGET],
        mode='lines+markers', name='Histori Aktual', line=dict(color='#1f77b4', width=2)))
    fig_fw.add_trace(go.Scatter(x=tgl_pred_list, y=qty_pred_list,
        mode='lines+markers', name='Prediksi (ML)', line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=9, symbol='diamond')))
    fig_fw.add_trace(go.Scatter(x=tgl_pred_list, y=safety_line,
        mode='lines', name=f'Safety Stock (Z×σ = {safety_prod})',
        line=dict(color='#4CAF50', width=2, dash='dot')))
    fig_fw.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0), hovermode='x unified',
        xaxis_title='Tanggal', yaxis_title=f'Qty ({satuan_r})',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    st.plotly_chart(fig_fw, use_container_width=True)

    rows_detail = []
    for i, ((s, e), qty) in enumerate(zip(opsi_minggu, qty_pred_list), 1):
        rows_detail.append({
            'Minggu': f'Minggu {i}',
            'Periode (Senin – Minggu)': f'{s.strftime("%d %b %Y")} – {e.strftime("%d %b %Y")}',
            'Safety Stock (Z×σ)': safety_prod,
            'Prediksi Qty': qty,
            'Satuan': satuan_r,
        })
    st.dataframe(pd.DataFrame(rows_detail), use_container_width=True, hide_index=True)
    st.caption(
        f"ℹ️ **Safety Stock** = Z × σ = {Z_SERVICE} × {std_prod:.2f} = **{safety_prod}** {satuan_r} per minggu   "
    )

    st.markdown("---")

    # ── FIX #5: Rekap Semua Produk (tanpa kolom MAE Produk) ───
    st.subheader(f"📋 Rekap Kebutuhan Stok {rentang} Minggu ke Depan")
    label_rentang = "  |  ".join(
        f"Minggu {i+1}: {s.strftime('%d %b')}–{e.strftime('%d %b %Y')}"
        for i, (s,e) in enumerate(opsi_minggu))
    st.caption(label_rentang)

    with st.spinner("Menghitung prediksi untuk semua produk..."):
        rows = []
        for produk in sorted(df_weekly_fit['Nama Barang'].unique()):
            df_p2  = df_weekly_fit[df_weekly_fit['Nama Barang'] == produk].sort_values('Tanggal Transaksi')
            satuan = df_p2['Satuan'].iloc[-1]
            inp    = df_p2[fitur_cols].iloc[[-1]].copy()
            std_p  = _std_produk(produk)

            preds = []
            for idx in range(rentang):
                pv = int(round(max(0.0, float(predict(model, inp, model_nama)[0]))))
                preds.append(pv)
                tgl_next = (opsi_minggu[idx+1][0] if idx+1 < rentang
                            else opsi_minggu[-1][0] + pd.Timedelta(weeks=1))
                tgl_next += pd.Timedelta(days=3)
                inp = roll_forward(inp, pv, tgl_next)

            safety_per_minggu = hitung_safety_stock(std_p)   # bulat, per minggu
            safety_total      = safety_per_minggu * rentang   # akumulasi n minggu

            row = {'Nama Barang': produk, 'Satuan': satuan}
            for i, ((s, e), pv) in enumerate(zip(opsi_minggu, preds), 1):
                row[f'Minggu {i} ({s.strftime("%d/%m")}–{e.strftime("%d/%m")})'] = pv

            # FIX 2: Safety Stock berdiri sendiri (sebelum kolom Total Prediksi)
            # FIX 3: semua angka dibulatkan (int)
            row['Safety Stock (Z×σ×n)'] = safety_total
            row['Total Prediksi']        = sum(preds)
            rows.append(row)

        df_rekap = pd.DataFrame(rows)

    st.dataframe(df_rekap, use_container_width=True)

    # ── Download Excel ─────────────────────────────────────────
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_rekap.to_excel(writer, index=False, sheet_name='Rekap Peramalan')
        pd.DataFrame({
            'Metrik': ['Model','Zero Threshold (%)','Produk Diramal','Minggu Prediksi',
                       'Safety Stock Z','MAE','RMSE','R²','WMAPE (%)','MAPE (%)',
                       'Periode Dataset','Batas Train','Tanggal Export'],
            'Nilai': [
                model_nama, zero_threshold, n_produk, rentang, f'Z={Z_SERVICE} (95% SL)',
                f'{mae:.4f}', f'{rmse:.4f}', f'{r2:.4f}', f'{wmape:.2f}', f'{mape:.2f}',
                f'{tgl_min_fit.strftime("%d %b %Y")} – {tgl_max_fit.strftime("%d %b %Y")}',
                batas_train.strftime('%d %b %Y'),
                pd.Timestamp.today().strftime('%d %b %Y %H:%M'),
            ]
        }).to_excel(writer, index=False, sheet_name='Evaluasi Model')
        pd.concat([
            grup_a.assign(Grup='A'), grup_b.assign(Grup='B'), grup_c.assign(Grup='C')
        ]).rename(columns={'total_harga':'Total Pendapatan'}).to_excel(
            writer, index=False, sheet_name='Analisis ABC')

    buffer.seek(0)
    nama_file = f"Peramalan_{model_nama}_{pd.Timestamp.today().strftime('%Y%m%d')}.xlsx"
    st.download_button("📥 Unduh Rekap Stok (Excel)", data=buffer, file_name=nama_file,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        use_container_width=True, type='primary')
    st.caption("File Excel berisi 3 sheet: **Rekap Peramalan**, **Evaluasi Model**, **Analisis ABC**")

    st.markdown("---")
    col_back, _ = st.columns([1, 4])
    with col_back:
        if st.button("◀ Kembali ke Konfigurasi"):
            st.session_state.update({
                'tahap': 3, 'df_weekly_fit': None,
                'mae': None, 'y_pred_val': None,
            })
            st.rerun()