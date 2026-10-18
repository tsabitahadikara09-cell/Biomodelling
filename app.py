
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Gait Signal Analysis",
    page_icon="📈",
    layout="wide"
)

# ==================================================
# STYLE: BLACK - GREEN
# ==================================================
st.markdown("""
<style>
.stApp {
    background: #0b100e;
    color: #e8f5ec;
}
[data-testid="stHeader"] {
    background: #0b100e;
}
[data-testid="stSidebar"] {
    background: #141b17;
    border-right: 1px solid #294536;
}
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1800px;
}
h1, h2, h3, p, label {
    color: #e8f5ec;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #111915;
    border: 1px solid #294536;
    border-radius: 8px;
    padding: 10px;
}
[data-testid="stMetricValue"] {
    color: #54e38e;
}
div.stButton > button,
div.stDownloadButton > button {
    background: #176b42;
    color: white;
    border: 1px solid #39b86b;
    border-radius: 6px;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background: #218c56;
    color: white;
}
.section-title {
    text-align: center;
    color: #54e38e;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# HEADER
# ==================================================
st.markdown(
    "<h1 style='color:#54e38e'>Gait Signal Analysis</h1>",
    unsafe_allow_html=True
)
st.caption(
    "Heel–Toe Signal | Cross Correlation | Normalization"
)


# ==================================================
# SIDEBAR: UPLOAD FILE
# ==================================================
with st.sidebar:
    st.markdown(
        "<h2 style='color:#54e38e'>📂 Input Data</h2>",
        unsafe_allow_html=True
    )

    heel_file = st.file_uploader(
        "Upload Heel Signal",
        type=["txt", "csv"],
        key="heel_upload"
    )

    toe_file = st.file_uploader(
        "Upload Toe Signal",
        type=["txt", "csv"],
        key="toe_upload"
    )

    st.caption(
        "Upload Heel123.txt dan Toe123.txt secara terpisah. "
        "Setiap file berisi satu nilai sinyal per baris."
    )

    st.divider()

    st.subheader("Pengaturan Grafik")

    show_grid = st.checkbox(
        "Tampilkan grid",
        value=True
    )

    line_width = st.slider(
        "Ketebalan garis",
        1, 5, 2
    )

    signal_window = st.selectbox(
        "Jumlah sampel sinyal",
        [250, 500, 1000, 1500, 2000],
        index=2
    )


# ==================================================
# READ SIGNAL FILE
# ==================================================
def read_signal(uploaded_file):
    """Membaca sinyal numerik dari TXT atau CSV."""
    uploaded_file.seek(0)

    data = pd.read_csv(
        uploaded_file,
        header=None,
        comment="#",
        sep=r"[\s,;]+",
        engine="python"
    )

    values = data.to_numpy().ravel()

    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).dropna().to_numpy(dtype=float)

    if len(values) == 0:
        raise ValueError("File tidak berisi data numerik.")

    if not np.all(np.isfinite(values)):
        raise ValueError("Data mengandung nilai tidak valid.")

    return values


# ==================================================
# DEMO DATA
# ==================================================
def demo_data():
    n = 6500
    t = np.arange(n)

    heel = np.zeros(n)
    toe = np.zeros(n)

    cycles = [
        (350, 220, 1.15),
        (1350, 250, 2.0),
        (2350, 260, 1.85),
        (3350, 240, 1.9),
        (4350, 260, 1.95),
        (5350, 280, 1.8)
    ]

    for start, width, amp in cycles:
        end = min(start + width, n)
        idx = np.arange(start, end)
        phase = (idx - start) / width

        heel[idx] = amp * np.sin(
            np.pi * np.clip(phase, 0, 1)
        ) ** 1.2

        toe_start = start + 90
        toe_end = min(start + width + 260, n)
        idx_toe = np.arange(toe_start, toe_end)
        phase_toe = (
            (idx_toe - toe_start) / (width + 170)
        )

        toe[idx_toe] = amp * 0.85 * np.sin(
            np.pi * np.clip(phase_toe, 0, 1)
        ) ** 1.3

    return heel, toe


# ==================================================
# LOAD HEEL AND TOE
# ==================================================
if heel_file is not None and toe_file is not None:
    try:
        heel = read_signal(heel_file)
        toe = read_signal(toe_file)

        # Use equal-length signals
        n = min(len(heel), len(toe))
        heel = heel[:n]
        toe = toe[:n]

        st.success(
            f"Data berhasil dimuat: {n:,} sampel"
        )

    except Exception as error:
        st.error(f"Gagal membaca file: {error}")
        heel, toe = demo_data()
        n = min(len(heel), len(toe))
        heel, toe = heel[:n], toe[:n]

elif heel_file is None and toe_file is None:
    heel, toe = demo_data()
    n = min(len(heel), len(toe))
    st.info(
        "Upload Heel dan Toe untuk menganalisis data asli. "
        "Saat ini ditampilkan data contoh."
    )

else:
    heel, toe = demo_data()
    n = min(len(heel), len(toe))
    st.warning(
        "Upload kedua file Heel dan Toe agar analisis "
        "menggunakan data asli. Sementara ini data contoh ditampilkan."
    )


# ==================================================
# CROSS CORRELATION
# ==================================================
# Remove DC component so the correlation can have
# positive and negative values.
heel_centered = heel - np.mean(heel)
toe_centered = toe - np.mean(toe)

# Full cross-correlation:
# lag < 0 : left side
# lag = 0 : center
# lag > 0 : right side
rxy = np.correlate(
    heel_centered,
    toe_centered,
    mode="full"
)

lags = np.arange(
    -(len(toe_centered) - 1),
    len(heel_centered)
)

# Normalized cross-correlation
energy = np.sqrt(
    np.sum(heel_centered ** 2)
    * np.sum(toe_centered ** 2)
)

if energy > 0:
    rxy_norm = rxy / energy
else:
    rxy_norm = np.zeros_like(rxy, dtype=float)

# Peak correlation
peak_index = int(np.argmax(np.abs(rxy_norm)))
peak_lag = int(lags[peak_index])
peak_value = float(rxy_norm[peak_index])


# ==================================================
# CORRELATION LAG SLIDER
# ==================================================
st.markdown(
    "<h3 style='color:#54e38e'>Rentang Analisis</h3>",
    unsafe_allow_html=True
)

max_lag = max(n - 1, 1)

lag_window = st.slider(
    "Rentang lag dari titik 0 (kiri negatif, kanan positif)",
    min_value=1,
    max_value=max_lag,
    value=min(1000, max_lag),
    step=max(1, max_lag // 500)
)

signal_window = min(signal_window, n)
max_start = max(n - signal_window, 0)

start_idx = st.slider(
    "Geser untuk melihat bagian sinyal Heel–Toe",
    min_value=0,
    max_value=max_start,
    value=0,
    step=max(1, min(50, max_start or 1))
)

end_idx = min(start_idx + signal_window, n)

sample_axis = np.arange(start_idx, end_idx)
heel_view = heel[start_idx:end_idx]
toe_view = toe[start_idx:end_idx]

# Symmetric lag view around zero
lag_mask = (
    (lags >= -lag_window)
    & (lags <= lag_window)
)

lags_view = lags[lag_mask]
rxy_view = rxy[lag_mask]
norm_view = rxy_norm[lag_mask]


# ==================================================
# GRAPH FUNCTION
# ==================================================
def create_chart(
    title,
    x,
    traces,
    x_title,
    y_title,
    height=280,
    zero_center=False
):
    fig = go.Figure()

    for label, y, color in traces:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=label,
                line=dict(
                    color=color,
                    width=line_width
                )
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(
                size=15,
                color="#54e38e"
            )
        ),
        height=height,
        margin=dict(l=55, r=20, t=45, b=50),
        paper_bgcolor="#111915",
        plot_bgcolor="#0b100e",
        font=dict(
            family="Arial",
            size=11,
            color="#e8f5ec"
        ),
        legend=dict(
            orientation="h",
            y=1.12,
            x=1,
            xanchor="right"
        ),
        xaxis=dict(
            title=x_title,
            showgrid=show_grid,
            gridcolor="#26382e",
            zeroline=True,
            zerolinecolor="#e8f5ec",
            zerolinewidth=1.5,
            range=(
                [-lag_window, lag_window]
                if zero_center else None
            )
        ),
        yaxis=dict(
            title=y_title,
            showgrid=show_grid,
            gridcolor="#26382e",
            zeroline=True,
            zerolinecolor="#496454"
        )
    )

    # Vertical line at lag = 0
    if zero_center:
        fig.add_vline(
            x=0,
            line_color="#54e38e",
            line_width=2,
            line_dash="dash"
        )

    return fig


# ==================================================
# MAIN LAYOUT
# ==================================================
left, right = st.columns(
    [1.65, 1],
    gap="medium"
)

with left:

    # ----------------------------------------------
    # GAIT SIGNAL
    # ----------------------------------------------
    with st.container(border=True):
        st.plotly_chart(
            create_chart(
                "Gait Signal",
                sample_axis,
                [
                    ("Heel (x)", heel_view, "#ff4545"),
                    ("Toe (y)", toe_view, "#39d353")
                ],
                "Sample Number",
                "Amplitude",
                height=280
            ),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    # ----------------------------------------------
    # CROSS CORRELATION
    # ----------------------------------------------
    with st.container(border=True):
        st.plotly_chart(
            create_chart(
                "Cross Correlation",
                lags_view,
                [
                    ("Rxy", rxy_view, "#ff4545")
                ],
                "Lag (L)",
                "Cross Correlation",
                height=300,
                zero_center=True
            ),
            use_container_width=True,
            config={"displayModeBar": False}
        )

        st.caption(
            "Lag negatif ← | 0 = titik tengah | → Lag positif"
        )

    # ----------------------------------------------
    # NORMALIZATION
    # ----------------------------------------------
    with st.container(border=True):
        st.plotly_chart(
            create_chart(
                "Normalized Cross Correlation",
                lags_view,
                [
                    ("Normalized Rxy", norm_view, "#39d353")
                ],
                "Lag (L)",
                "Normalized Rxy",
                height=280,
                zero_center=True
            ),
            use_container_width=True,
            config={"displayModeBar": False}
        )


# ==================================================
# NUMERICAL DATA PANEL
# ==================================================
with right:

    st.markdown(
        "<h3 style='color:#54e38e;text-align:center'>"
        "Data Numerik</h3>",
        unsafe_allow_html=True
    )

    with st.container(border=True):

        signal_table = pd.DataFrame({
            "Index": np.arange(start_idx, end_idx),
            "Heel (x)": heel_view,
            "Toe (y)": toe_view
        })

        st.dataframe(
            signal_table,
            height=270,
            use_container_width=True,
            hide_index=True
        )

    with st.container(border=True):

        corr_table = pd.DataFrame({
            "Lag": lags_view,
            "Rxy": rxy_view,
            "Normalized Rxy": norm_view
        })

        # Show rows around lag zero in the table
        center = len(corr_table) // 2
        table_start = max(0, center - 15)
        table_end = min(len(corr_table), center + 16)

        st.markdown(
            "<div class='section-title'>"
            "Cross Correlation Values</div>",
            unsafe_allow_html=True
        )

        st.dataframe(
            corr_table.iloc[table_start:table_end],
            height=300,
            use_container_width=True,
            hide_index=True
        )

    st.metric(
        "Lag pada korelasi absolut maksimum",
        f"{peak_lag:+d}"
    )

    st.metric(
        "Nilai korelasi ternormalisasi",
        f"{peak_value:.4f}"
    )

    st.download_button(
        "⬇️ Download Hasil Cross Correlation",
        data=corr_table.to_csv(index=False).encode("utf-8"),
        file_name="cross_correlation.csv",
        mime="text/csv",
        use_container_width=True
    )

st.caption(
    "Cross-correlation dihitung dari sinyal Heel dan Toe "
    "yang telah dikurangi nilai rata-ratanya. "
    "Normalisasi menggunakan energi total kedua sinyal."
)
