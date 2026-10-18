
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Gait Signal Analysis",
    page_icon="📈",
    layout="wide"
)

# =========================
# DARK GREEN THEME
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #0b100e;
    color: #e8f5ec;
}
[data-testid="stHeader"] {
    background-color: #0b100e;
}
[data-testid="stSidebar"] {
    background-color: #141b17;
    border-right: 1px solid #294536;
}
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1800px;
}
h1, h2, h3, p, label, span {
    color: #e8f5ec;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #111915;
    border: 1px solid #294536;
    border-radius: 8px;
    padding: 10px;
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
[data-testid="stMetricValue"] {
    color: #54e38e;
}
.section-title {
    color: #54e38e;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================
st.markdown(
    "<h1 style='color:#54e38e'>Gait Signal Analysis</h1>",
    unsafe_allow_html=True
)
st.caption(
    "Analisis sinyal Heel-Toe, Cross Correlation, dan Normalisasi"
)


# =========================
# CONTOH DATA UNTUK PREVIEW
# =========================
def create_demo_data():
    n = 6500
    sample = np.arange(n)

    heel = np.zeros(n)
    toe = np.zeros(n)

    gait_cycles = [
        (350, 220, 1.15),
        (1350, 250, 2.0),
        (2350, 260, 1.85),
        (3350, 240, 1.9),
        (4350, 260, 1.95),
        (5350, 280, 1.8)
    ]

    for start, width, amp in gait_cycles:
        end = min(start + width, n)
        idx = np.arange(start, end)
        phase = (idx - start) / width

        heel[idx] = amp * np.sin(
            np.pi * np.clip(phase, 0, 1)
        ) ** 1.2

        start_toe = start + 90
        end_toe = min(start + width + 260, n)
        idx_toe = np.arange(start_toe, end_toe)
        phase_toe = (
            (idx_toe - start_toe) / (width + 170)
        )

        toe[idx_toe] = amp * 0.85 * np.sin(
            np.pi * np.clip(phase_toe, 0, 1)
        ) ** 1.3

    lag = np.arange(n)
    rxy = np.zeros(n)

    peaks = [
        (650, 2.4, 180),
        (1650, 1.8, 170),
        (2650, 1.35, 170),
        (3650, 0.75, 150),
        (4650, 0.38, 130)
    ]

    for center, amp, width in peaks:
        rxy += amp * np.exp(
            -0.5 * ((lag - center) / width) ** 2
        )

    rxy += 0.02 * np.exp(-lag / 900)

    norm = rxy / max(np.max(rxy), 1e-9) * 0.42

    return pd.DataFrame({
        "Sample": sample,
        "Heel (x)": heel,
        "Toe (y)": toe,
        "Rxy": rxy * 1000,
        "Norm": norm
    })


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown(
        "<h2 style='color:#54e38e'>📂 Input Data</h2>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload file data",
        type=["csv", "txt", "xlsx", "xls"]
    )

    st.divider()
    st.subheader("Pengaturan Grafik")

    show_grid = st.checkbox(
        "Tampilkan grid",
        value=True
    )

    line_width = st.slider(
        "Ketebalan garis",
        min_value=1,
        max_value=5,
        value=2
    )

    window_size = st.selectbox(
        "Rentang sinyal",
        [500, 1000, 1500, 2000],
        index=1
    )


# =========================
# BACA DATA
# =========================
if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(
            (".xlsx", ".xls")
        ):
            df = pd.read_excel(uploaded_file)
        else:
            try:
                df = pd.read_csv(
                    uploaded_file,
                    sep=None,
                    engine="python",
                    comment="#"
                )
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(
                    uploaded_file,
                    sep=r"\s+",
                    comment="#",
                    header=None
                )

        if df.empty or len(df.columns) == 0:
            raise ValueError("File tidak memiliki data.")

        st.success(
            f"File berhasil dibaca: {uploaded_file.name}"
        )

    except Exception as error:
        st.error(f"Gagal membaca file: {error}")
        df = create_demo_data()
else:
    df = create_demo_data()
    st.info(
        "Belum ada file diunggah. "
        "Grafik contoh ditampilkan sebagai pratinjau UI."
    )


# =========================
# PEMETAAN KOLOM
# =========================
with st.expander(
    "⚙️ Pemetaan Kolom Data",
    expanded=uploaded_file is not None
):
    columns = list(df.columns)

    def guess_column(keywords, fallback=0):
        for col in columns:
            name = str(col).lower().replace("_", " ")
            if any(word in name for word in keywords):
                return col
        return columns[min(fallback, len(columns) - 1)]

    c1, c2, c3, c4, c5 = st.columns(5)

    sample_col = c1.selectbox(
        "Sample / waktu",
        columns,
        index=columns.index(
            guess_column(["sample", "time", "index"], 0)
        )
    )

    heel_col = c2.selectbox(
        "Heel (x)",
        columns,
        index=columns.index(
            guess_column(["heel"], min(1, len(columns)-1))
        )
    )

    toe_col = c3.selectbox(
        "Toe (y)",
        columns,
        index=columns.index(
            guess_column(["toe"], min(2, len(columns)-1))
        )
    )

    corr_col = c4.selectbox(
        "Cross Correlation",
        columns,
        index=columns.index(
            guess_column(["rxy", "corr"], min(3, len(columns)-1))
        )
    )

    norm_col = c5.selectbox(
        "Normalization",
        columns,
        index=columns.index(
            guess_column(["norm"], min(4, len(columns)-1))
        )
    )


def numeric_data(column):
    return pd.to_numeric(
        df[column], errors="coerce"
    ).fillna(0).to_numpy(dtype=float)


sample = numeric_data(sample_col)
heel = numeric_data(heel_col)
toe = numeric_data(toe_col)
rxy = numeric_data(corr_col)
norm = numeric_data(norm_col)

n = min(
    len(sample), len(heel), len(toe),
    len(rxy), len(norm)
)

sample = sample[:n]
heel = heel[:n]
toe = toe[:n]
rxy = rxy[:n]
norm = norm[:n]


# =========================
# SLIDER RENTANG SINYAL
# =========================
st.markdown(
    "<h3 style='color:#54e38e'>Rentang Sinyal</h3>",
    unsafe_allow_html=True
)

window_size = min(window_size, max(n, 1))
max_start = max(n - window_size, 0)

start_idx = st.slider(
    "Geser untuk melihat bagian sinyal yang berbeda",
    min_value=0,
    max_value=max_start,
    value=0,
    step=max(1, min(50, max_start or 1))
)

end_idx = min(start_idx + window_size, n)

st.caption(
    f"Menampilkan sampel {start_idx} sampai {end_idx - 1}"
)

# Grafik berubah sesuai posisi slider
sx = sample[start_idx:end_idx]
hx = heel[start_idx:end_idx]
tx = toe[start_idx:end_idx]
cx = rxy[start_idx:end_idx]
nx = norm[start_idx:end_idx]


# =========================
# FUNGSI GRAFIK
# =========================
def create_chart(
    title,
    x,
    traces,
    x_title,
    y_title,
    height=260
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
        margin=dict(l=50, r=20, t=45, b=45),
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
            zeroline=False
        ),
        yaxis=dict(
            title=y_title,
            showgrid=show_grid,
            gridcolor="#26382e",
            zeroline=False
        )
    )

    return fig


# =========================
# LAYOUT UTAMA
# =========================
left, right = st.columns(
    [1.65, 1],
    gap="medium"
)

with left:

    # Gait Signal
    with st.container(border=True):
        st.plotly_chart(
            create_chart(
                "Gait Signal",
                sx,
                [
                    ("Heel (x)", hx, "#ff4545"),
                    ("Toe (y)", tx, "#39d353")
                ],
                "Sample Number",
                "Amplitude",
                280
            ),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    # Cross Correlation
    with st.container(border=True):
        st.plotly_chart(
            create_chart(
                "Cross Correlation",
                sx,
                [
                    ("Rxy", cx, "#ff4545")
                ],
                "Index Time Lag (L)",
                "Rxy Value",
                270
            ),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    # Normalization
    with st.container(border=True):
        st.plotly_chart(
            create_chart(
                "Normalization",
                sx,
                [
                    ("Norm", nx, "#39d353")
                ],
                "Index Time Lag (L)",
                "Normalized Rxy",
                270
            ),
            use_container_width=True,
            config={"displayModeBar": False}
        )


# =========================
# DATA NUMERIK
# =========================
with right:
    with st.container(border=True):

        st.markdown(
            "<h3 style='color:#54e38e;text-align:center'>"
            "Data Numerik</h3>",
            unsafe_allow_html=True
        )

        selected = min(start_idx, max(n - 1, 0))

        table_start = max(0, selected - 10)
        table_end = min(n, selected + 31)

        table = pd.DataFrame({
            "Index": np.arange(table_start, table_end),
            "x[i]": heel[table_start:table_end],
            "y[i]": toe[table_start:table_end],
            "rxy[i]": rxy[table_start:table_end],
            "Norm[i]": norm[table_start:table_end]
        })

        st.dataframe(
            table,
            height=440,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Menampilkan indeks {table_start}–"
            f"{max(table_start, table_end - 1)} "
            f"dari {n:,} baris."
        )

    if st.button(
        "📊 Tampilkan Data",
        use_container_width=True
    ):
        st.session_state["show_data"] = True

    if st.session_state.get("show_data", False) and n > 0:
        st.subheader("Nilai pada Indeks Terpilih")

        selected_data = pd.DataFrame([{
            "Index": selected,
            "Sample": sample[selected],
            "Heel (x)": heel[selected],
            "Toe (y)": toe[selected],
            "Rxy": rxy[selected],
            "Norm": norm[selected]
        }])

        st.dataframe(
            selected_data,
            use_container_width=True,
            hide_index=True
        )

    st.download_button(
        "⬇️ Download Data (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="gait_data.csv",
        mime="text/csv",
        use_container_width=True
    )


st.caption(
    "Catatan: Rxy dan Normalization saat ini dibaca dari kolom file. "
    "Jika file hanya berisi Heel dan Toe, perhitungan cross-correlation "
    "dan normalisasi perlu ditambahkan agar dihitung otomatis."
)
