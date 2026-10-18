import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO

st.set_page_config(page_title="Gait Signal Analysis", page_icon="📈", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
    .stApp { background: #e8f1e8; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .main .block-container { padding-top: 1.1rem; padding-bottom: 1.5rem; max-width: 1600px; }
    .app-title { color: #1647ff; font-size: 1.65rem; font-weight: 650; margin-bottom: .15rem; }
    .app-subtitle { color: #4e6253; font-size: .9rem; margin-bottom: 1rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #f7f8f7; border: 1px solid #d5ded5; border-radius: 4px;
        padding: 10px 12px;
    }
    div[data-testid="stFileUploader"] { background: #f7f8f7; padding: 10px; border-radius: 4px; }
    div.stButton > button { border-radius: 3px; }
    .section-label { color: #1647ff; font-weight: 600; text-align: center; margin-bottom: .2rem; }
    .small-note { color: #647468; font-size: .8rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-title">Gait Signal Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Analisis sinyal Heel–Toe, cross correlation, dan normalisasi</div>', unsafe_allow_html=True)

# ---------- Sidebar / upload ----------
with st.sidebar:
    st.header("📂 Input Data")
    uploaded = st.file_uploader(
        "Upload file data",
        type=["csv", "txt", "xlsx", "xls"],
        help="CSV/TXT/Excel. Kolom yang dikenali: sample/time, heel/x, toe/y, rxy/correlation."
    )
    st.caption("UI dapat dicoba dengan data contoh sebelum file asli tersedia.")
    st.divider()
    st.subheader("Pengaturan Grafik")
    show_grid = st.checkbox("Tampilkan grid", value=True)
    line_width = st.slider("Ketebalan garis", 1, 5, 2)
    st.caption("Setelah file diunggah, pilih kolom data di bagian pemetaan kolom.")

def demo_data():
    # Dummy signal purely for preview; replace with uploaded data for analysis.
    n = 6500
    t = np.arange(n)
    heel = np.zeros(n, dtype=float)
    toe = np.zeros(n, dtype=float)
    for start, width, amp in [(350, 220, 1.15), (1350, 250, 2.0), (2350, 260, 1.85),
                              (3350, 240, 1.9), (4350, 260, 1.95), (5350, 280, 1.8)]:
        idx = np.arange(start, min(start+width, n))
        phase = (idx-start)/max(width, 1)
        heel[idx] = amp * np.sin(np.pi*np.clip(phase, 0, 1))**1.2
        idx2 = np.arange(start+90, min(start+width+260, n))
        phase2 = (idx2-(start+90))/max(width+170, 1)
        toe[idx2] = (amp*.85) * np.sin(np.pi*np.clip(phase2, 0, 1))**1.3
    # Example decaying correlation peaks
    lag = np.arange(n)
    corr = np.zeros(n, dtype=float)
    for i, (center, amp, width) in enumerate([(650, 2.4, 180), (1650, 1.8, 170),
                                               (2650, 1.35, 170), (3650, .75, 150),
                                               (4650, .38, 130)]):
        corr += amp*np.exp(-0.5*((lag-center)/width)**2)
    corr += 0.02*np.exp(-lag/900)
    norm = corr / max(np.max(corr), 1e-9) * 0.42
    return pd.DataFrame({"Sample": t, "Heel (x)": heel, "Toe (y)": toe,
                         "Lag": lag, "Rxy": corr*1000, "Norm": norm})

def read_uploaded(file):
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    try:
        return pd.read_csv(file, sep=None, engine="python", comment="#")
    except Exception:
        file.seek(0)
        return pd.read_csv(file, delim_whitespace=True, comment="#", header=None)

if uploaded is not None:
    try:
        df = read_uploaded(uploaded)
        st.success(f"File berhasil dibaca: {uploaded.name} · {len(df):,} baris")
    except Exception as e:
        st.error(f"File belum bisa dibaca. Periksa format file. Detail: {e}")
        df = demo_data()
else:
    df = demo_data()
    st.info("Belum ada file diunggah — grafik contoh ditampilkan sebagai pratinjau UI, bukan hasil analisis asli.")

# ---------- Column mapping ----------
with st.expander("⚙️ Pemetaan Kolom Data", expanded=uploaded is not None):
    cols = list(df.columns)
    def guess_col(keys, fallback_idx=0):
        for col in cols:
            if any(k in str(col).lower().replace("_", " ") for k in keys):
                return col
        return cols[min(fallback_idx, len(cols)-1)]
    c1, c2, c3, c4, c5 = st.columns(5)
    sample_col = c1.selectbox("Sample / waktu", cols, index=cols.index(guess_col(["sample", "time", "index", "lag"], 0)))
    heel_col = c2.selectbox("Heel (x)", cols, index=cols.index(guess_col(["heel", "x"], 1 if len(cols)>1 else 0)))
    toe_col = c3.selectbox("Toe (y)", cols, index=cols.index(guess_col(["toe", "y"], 2 if len(cols)>2 else 0)))
    corr_col = c4.selectbox("Cross correlation (Rxy)", cols, index=cols.index(guess_col(["rxy", "corr", "cross"], min(3,len(cols)-1))))
    norm_col = c5.selectbox("Normalization", cols, index=cols.index(guess_col(["norm", "normal"], min(4,len(cols)-1))))

def numeric_series(column):
    return pd.to_numeric(df[column], errors="coerce").fillna(0).to_numpy(dtype=float)

sample = numeric_series(sample_col)
heel = numeric_series(heel_col)
toe = numeric_series(toe_col)
corr = numeric_series(corr_col)
norm = numeric_series(norm_col)

# Keep arrays aligned if columns have different/odd values; all originate from same table.
n = min(map(len, [sample, heel, toe, corr, norm]))
sample, heel, toe, corr, norm = [a[:n] for a in [sample, heel, toe, corr, norm]]

left, right = st.columns([1.55, 1], gap="medium")

def make_chart(title, x, traces, x_title, y_title, height=235):
    fig = go.Figure()
    for label, y, color in traces:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label,
                                 line=dict(color=color, width=line_width)))
    fig.update_layout(
        title=dict(text=title, x=.5, xanchor="center", font=dict(size=13, color="#1647ff")),
        height=height, margin=dict(l=48, r=15, t=38, b=42),
        paper_bgcolor="#f7f8f7", plot_bgcolor="#f1f2f1",
        font=dict(family="Arial, sans-serif", size=10, color="#222"),
        legend=dict(orientation="v", x=1.01, y=1, font=dict(size=10)),
        xaxis_title=x_title, yaxis_title=y_title,
        xaxis=dict(showgrid=show_grid, gridcolor="#d4d4d4", zeroline=False),
        yaxis=dict(showgrid=show_grid, gridcolor="#d4d4d4", zeroline=False),
    )
    return fig

with left:
    with st.container(border=True):
        st.plotly_chart(make_chart("Gait Signal", sample, [
            ("Heel (x)", heel, "#ff4545"), ("Toe (y)", toe, "#279b35")
        ], "Sample Number", "Amplitudo", 230), use_container_width=True, config={"displayModeBar": False})

    max_idx = max(n-1, 0)
    s1, s2 = st.columns([5, 1])
    with s1:
        selected = st.slider("Indeks data", min_value=0, max_value=max_idx, value=min(0, max_idx), label_visibility="collapsed")
    with s2:
        st.metric("Index", selected)

    with st.container(border=True):
        st.plotly_chart(make_chart("Cross Correlation", sample, [
            ("Rxy", corr, "#ff4545")
        ], "Index Time Lag (L)", "Rxy Value", 220), use_container_width=True, config={"displayModeBar": False})

    with st.container(border=True):
        st.plotly_chart(make_chart("Normalization", sample, [
            ("Norm", norm, "#ff4545")
        ], "Index Time Lag (L)", "Rxy Normalization", 220), use_container_width=True, config={"displayModeBar": False})

with right:
    with st.container(border=True):
        st.markdown('<div class="section-label">Data Numerik</div>', unsafe_allow_html=True)
        start = max(0, selected-15)
        end = min(n, selected+36)
        view = pd.DataFrame({
            "Index": np.arange(start, end),
            "x[i]": heel[start:end],
            "y[i]": toe[start:end],
            "rxy[i]": corr[start:end],
            "Norm[i]": norm[start:end]
        })
        st.dataframe(view, height=420, use_container_width=True, hide_index=True)
        st.caption(f"Menampilkan indeks {start}–{max(start, end-1)} dari {n:,} baris.")

    if st.button("📊 Tampilkan Data", use_container_width=True):
        st.session_state["show_selected_data"] = True
    if st.session_state.get("show_selected_data", False):
        st.markdown("**Nilai pada indeks terpilih**")
        if n:
            st.dataframe(pd.DataFrame([{
                "Index": selected,
                "Sample": sample[selected],
                "Heel (x)": heel[selected],
                "Toe (y)": toe[selected],
                "Rxy": corr[selected],
                "Norm": norm[selected]
            }]), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Ekspor data yang ditampilkan (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="gait_data_preview.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown('<div class="small-note">Catatan: perhitungan cross-correlation dan normalization belum dihitung ulang oleh UI ini. Kolom tersebut dibaca dari file yang diunggah; bila file hanya berisi sinyal Heel dan Toe, tahap perhitungan bisa ditambahkan berikutnya.</div>', unsafe_allow_html=True)
