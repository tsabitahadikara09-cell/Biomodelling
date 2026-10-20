import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# KONFIGURASI APLIKASI
# =========================================================
st.set_page_config(
    page_title="Gait Signal Analysis - Barbie Edition",
    page_icon="💖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# TEMA PINK BARBIE MENTOK (HOT PINK & BLACK/DARK PURPLE)
# =========================================================
st.markdown(
    """
<style>
.stApp {
    background-color: #120810;
    color: #ffd6ec;
}
[data-testid="stHeader"] {
    background-color: #120810;
}
[data-testid="stSidebar"] {
    background-color: #1f0d1a;
    border-right: 2px solid #ff1493;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1800px;
}
h1, h2, h3, p, label {
    color: #ffb3d9 !important;
}
[data-testid="stMetricValue"] {
    color: #ff1493 !important;
    text-shadow: 0 0 10px rgba(255, 20, 147, 0.5);
}
div.stButton > button,
div.stDownloadButton > button {
    background-color: #ff1493;
    color: white;
    font-weight: bold;
    border: 2px solid #ff69b4;
    border-radius: 8px;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background-color: #ff69b4;
    color: white;
    border: 2px solid #ffffff;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #1a0a15;
    border: 1px solid #ff1493;
    border-radius: 10px;
    box-shadow: 0 0 15px rgba(255, 20, 147, 0.15);
}
div[data-testid="stDataFrame"] {
    border: 1px solid #ff1493;
    border-radius: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

BARBIE_HOT = "#ff1493"  # Deep Pink / Hot Pink khas Barbie
BARBIE_LIGHT = "#ff69b4"  # Light Hot Pink
BARBIE_ACCENT = "#ff007f"  # Magenta cetar
BG = "#120810"
PANEL = "#1a0a15"
GRID = "#4a1535"


# =========================================================
# MEMBACA FILE TXT / CSV
# =========================================================
def read_signal(uploaded_file):
  """Membaca satu kolom sinyal dari TXT atau CSV.

  Mendukung pemisah spasi, tab, koma, dan titik koma. Baris yang bukan angka
  akan diabaikan.
  """
  uploaded_file.seek(0)

  content = uploaded_file.getvalue().decode("utf-8-sig", errors="replace")

  values = []

  for line in content.splitlines():
    line = line.strip()

    if not line or line.startswith("#"):
      continue

    token = line.replace(",", " ").replace(";", " ").split()

    if not token:
      continue

    try:
      value = float(token[0])
      if np.isfinite(value):
        values.append(value)
    except ValueError:
      continue

  if not values:
    raise ValueError(
        "Tidak ditemukan data numerik. Pastikan file berisi angka sinyal."
    )

  return np.asarray(values, dtype=float)


# =========================================================
# DATA CONTOH UNTUK PREVIEW
# =========================================================
def create_demo_data():
  n = 6500
  heel = np.zeros(n)
  toe = np.zeros(n)

  cycles = [
      (350, 220, 1.15),
      (1350, 250, 2.00),
      (2350, 260, 1.85),
      (3350, 240, 1.90),
      (4350, 260, 1.95),
      (5350, 280, 1.80),
  ]

  for start, width, amp in cycles:
    end = min(start + width, n)
    idx = np.arange(start, end)
    phase = (idx - start) / width

    heel[idx] = amp * np.sin(np.pi * np.clip(phase, 0, 1)) ** 1.2

    toe_start = start + 90
    toe_end = min(start + width + 260, n)

    idx_toe = np.arange(toe_start, toe_end)
    phase_toe = (idx_toe - toe_start) / (width + 170)

    toe[idx_toe] = (
        amp * 0.85 * np.sin(np.pi * np.clip(phase_toe, 0, 1)) ** 1.3
    )

  return heel, toe


# =========================================================
# SIDEBAR: INPUT DATA
# =========================================================
with st.sidebar:
  st.markdown(
      "<h2 style='color:#ff1493'>💖 Barbie Input Data</h2>",
      unsafe_allow_html=True,
  )

  heel_file = st.file_uploader(
      "Upload Heel Signal", type=["txt", "csv"], key="heel_file"
  )

  toe_file = st.file_uploader(
      "Upload Toe Signal", type=["txt", "csv"], key="toe_file"
  )

  st.caption(
      "Upload Heel123.txt dan Toe123.txt secara terpisah. "
      "Gunakan file berisi satu nilai sinyal per baris."
  )

  st.divider()

  st.markdown(
      "<h3 style='color:#ff1493'>Pengaturan Grafik</h3>", unsafe_allow_html=True
  )

  show_grid = st.checkbox("Tampilkan grid", value=True)

  line_width = st.slider(
      "Ketebalan garis", min_value=1, max_value=5, value=2
  )

  signal_window = st.selectbox(
      "Jumlah sampel pada tampilan", [500, 1000, 2000, 3000, 5000], index=4
  )

  show_table_rows = st.selectbox("Jumlah baris tabel", [10, 25, 50, 100], index=1)


# =========================================================
# LOAD DATA
# =========================================================
using_demo = False

if heel_file is not None and toe_file is not None:
  try:
    heel_raw = read_signal(heel_file)
    toe_raw = read_signal(toe_file)

    if len(heel_raw) != len(toe_raw):
      st.warning(
          f"Panjang sinyal berbeda: Heel = {len(heel_raw):,}, "
          f"Toe = {len(toe_raw):,}. "
          "Analisis menggunakan jumlah sampel yang sama."
      )

    n = min(len(heel_raw), len(toe_raw))
    heel = heel_raw[:n]
    toe = toe_raw[:n]

    st.success(f"Data asli berhasil dimuat: {n:,} sampel.")

  except Exception as error:
    st.error(f"Gagal membaca file: {error}")
    heel, toe = create_demo_data()
    n = len(heel)
    using_demo = True

elif heel_file is None and toe_file is None:
  heel, toe = create_demo_data()
  n = len(heel)
  using_demo = True

  st.info(
      "Upload file Heel dan Toe di sidebar. "
      "Grafik contoh ditampilkan sebelum file tersedia."
  )

else:
  heel, toe = create_demo_data()
  n = len(heel)
  using_demo = True

  st.warning(
      "Kedua file harus diunggah agar analisis menggunakan "
      "data asli. Untuk sementara ditampilkan data contoh."
  )


# =========================================================
# HEADER
# =========================================================
st.markdown(
    "<h1 style='color:#ff1493; text-align:center;'>✨ 💖 GAIT SIGNAL ANALYSIS: BARBIE EDITION 💖 ✨</h1>",
    unsafe_allow_html=True,
)

st.caption(
    "Heel–Toe Signal | Cross Correlation | Normalization (Super Fabulous)"
)

if using_demo:
  st.info("Mode preview: hasil di bawah belum merupakan analisis file gait asli.")


# =========================================================
# CROSS CORRELATION
# =========================================================
x = heel - np.mean(heel)
y = toe - np.mean(toe)

lags = np.arange(-(n - 1), n)
rxy = np.correlate(x, y, mode="full")

energy = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
if energy > 0:
  rxy_norm = rxy / energy
else:
  rxy_norm = np.zeros_like(rxy, dtype=float)

zero_index = n - 1
peak_positive_idx = int(np.argmax(rxy_norm))
peak_positive_lag = int(lags[peak_positive_idx])
peak_positive_value = float(rxy_norm[peak_positive_idx])

peak_abs_idx = int(np.argmax(np.abs(rxy_norm)))
peak_abs_lag = int(lags[peak_abs_idx])
peak_abs_value = float(rxy_norm[peak_abs_idx])

zero_rxy = float(rxy[zero_index])
zero_norm = float(rxy_norm[zero_index])


# =========================================================
# SLIDER 1: LAG YANG DIPILIH
# =========================================================
st.markdown(
    "<h3 style='color:#ff1493'>Pengaturan Lag</h3>", unsafe_allow_html=True
)

selected_lag = st.slider(
    "Pergeseran sinyal: kiri = lag negatif, tengah = 0, kanan = lag positif",
    min_value=-(n - 1),
    max_value=n - 1,
    value=0,
    step=1,
    key="selected_lag",
)

sample_axis = np.arange(n)
toe_axis = sample_axis + selected_lag
corr_idx = selected_lag + zero_index

selected_rxy = float(rxy[corr_idx])
selected_norm = float(rxy_norm[corr_idx])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Lag Terpilih", f"{selected_lag:+d}")
m2.metric("Cross Correlation", f"{selected_rxy:.4f}")
m3.metric("Normalized Rxy", f"{selected_norm:.4f}")
m4.metric("Lag Puncak Absolut", f"{peak_abs_lag:+d}")


# =========================================================
# SLIDER 2: RENTANG TAMPILAN SINYAL
# =========================================================
st.markdown(
    "<h3 style='color:#ff1493'>Rentang Tampilan Gait Signal</h3>",
    unsafe_allow_html=True,
)

window_size = min(signal_window, n)
max_start = max(n - window_size, 0)

start_idx = st.slider(
    "Geser untuk melihat siklus gait pada sampel yang berbeda",
    min_value=0,
    max_value=max_start,
    value=min(st.session_state.get("gait_start", 0), max_start),
    step=max(1, min(50, max_start or 1)),
    key="gait_start",
)

end_idx = min(start_idx + window_size, n)
heel_view = heel[start_idx:end_idx]
toe_view = toe[start_idx:end_idx]

sample_view = np.arange(start_idx, end_idx)
toe_view_axis = sample_view + selected_lag


# =========================================================
# FUNGSI FORMAT GRAFIK
# =========================================================
def chart_layout(
    fig,
    title,
    x_title,
    y_title,
    height=300,
    x_range=None,
    y_range=None,
):
  fig.update_layout(
      title=dict(
          text=title,
          x=0.5,
          font=dict(size=15, color=BARBIE_LIGHT, family="Arial Black"),
      ),
      height=height,
      margin=dict(l=55, r=25, t=50, b=50),
      paper_bgcolor=PANEL,
      plot_bgcolor=BG,
      font=dict(family="Arial", size=11, color="#ffd6ec"),
      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
      xaxis=dict(
          title=x_title,
          showgrid=show_grid,
          gridcolor=GRID,
          zeroline=True,
          zerolinecolor="#ff69b4",
          range=x_range,
      ),
      yaxis=dict(
          title=y_title,
          showgrid=show_grid,
          gridcolor=GRID,
          zeroline=True,
          zerolinecolor="#ff69b4",
          range=y_range,
      ),
  )
  return fig


# =========================================================
# LAYOUT UTAMA
# =========================================================
left, right = st.columns([1.65, 1], gap="medium")


# =========================================================
# KOLOM KIRI: TIGA GRAFIK
# =========================================================
with left:

  # CHART 1: GAIT SIGNAL (Hot Pink vs Neon White/Cyan terang untuk kontras maksimal)
  with st.container(border=True):
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=sample_view,
            y=heel_view,
            mode="lines",
            name="Heel (x)",
            line=dict(color="#ff1493", width=line_width),  # Hot Pink Barbie
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=toe_view_axis,
            y=toe_view,
            mode="lines",
            name="Toe (y)",
            line=dict(
                color="#00ffff", width=line_width
            ),  # Kontras Cyan Terang ala Yas Queen
        )
    )
    fig1.add_vline(
        x=start_idx, line_color="#ff69b4", line_dash="dot", line_width=1
    )
    fig1 = chart_layout(
        fig1,
        "Gait Signal",
        "Sample Number",
        "Amplitude",
        height=310,
        x_range=[start_idx, end_idx - 1],
    )
    st.plotly_chart(
        fig1, use_container_width=True, config={"displayModeBar": False}
    )
    st.caption(
        f"Menampilkan sampel {start_idx:,}–"
        f"{end_idx - 1:,} dari {n:,} sampel. "
        f"Toe digeser sebesar {selected_lag:+d} sampel."
    )

  # CHART 2: CROSS CORRELATION
  with st.container(border=True):
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=lags,
            y=rxy,
            mode="lines",
            name="Rxy",
            line=dict(color="#ff1493", width=line_width),
        )
    )
    fig2.add_vline(
        x=selected_lag, line_color="#ff69b4", line_dash="dash", line_width=2
    )
    fig2.add_trace(
        go.Scatter(
            x=[selected_lag],
            y=[selected_rxy],
            mode="markers",
            name="Lag terpilih",
            marker=dict(color="#ff69b4", size=10),
        )
    )
    fig2 = chart_layout(
        fig2, "Cross Correlation", "Index Time Lag (L)", "Rxy Value", height=310
    )
    st.plotly_chart(
        fig2, use_container_width=True, config={"displayModeBar": False}
    )
    st.caption("Lag negatif di kiri, lag 0 di tengah, lag positif di kanan.")

  # CHART 3: NORMALIZATION
  with st.container(border=True):
    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=lags,
            y=rxy_norm,
            mode="lines",
            name="Normalized Rxy",
            line=dict(color="#ff1493", width=line_width),
        )
    )
    fig3.add_hline(y=0, line_color="#ff69b4", line_width=1)
    fig3.add_vline(
        x=selected_lag, line_color="#ff69b4", line_dash="dash", line_width=2
    )
    fig3.add_trace(
        go.Scatter(
            x=[selected_lag],
            y=[selected_norm],
            mode="markers",
            name="Lag terpilih",
            marker=dict(color="#ff69b4", size=10),
        )
    )
    fig3 = chart_layout(
        fig3,
        "Normalization",
        "Index Time Lag (L)",
        "Normalized Rxy",
        height=310,
    )
    st.plotly_chart(
        fig3, use_container_width=True, config={"displayModeBar": False}
    )


# =========================================================
# KOLOM KANAN: DATA NUMERIK
# =========================================================
with right:
  st.markdown(
      "<h3 style='color:#ff1493;text-align:center'>"
      "Data Numerik</h3>",
      unsafe_allow_html=True,
  )

  # TABEL SINYAL
  with st.container(border=True):
    st.markdown(
        "<p style='color:#ff1493;font-weight:bold'>" "Heel dan Toe</p>",
        unsafe_allow_html=True,
    )
    table_start = min(start_idx, max(n - 1, 0))
    table_end = min(table_start + show_table_rows, n)

    signal_table = pd.DataFrame({
        "Index": np.arange(table_start, table_end),
        "Heel (x)": heel[table_start:table_end],
        "Toe (y)": toe[table_start:table_end],
    })

    st.dataframe(
        signal_table,
        height=320,
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Indeks {table_start}–{max(table_start, table_end - 1)}")

  # TABEL CROSS CORRELATION
  with st.container(border=True):
    st.markdown(
        "<p style='color:#ff1493;font-weight:bold'>"
        "Cross Correlation Values</p>",
        unsafe_allow_html=True,
    )
    center = corr_idx
    table_corr_start = max(0, center - show_table_rows // 2)
    table_corr_end = min(len(lags), table_corr_start + show_table_rows)

    corr_table = pd.DataFrame({
        "Lag": lags[table_corr_start:table_corr_end],
        "Rxy": rxy[table_corr_start:table_corr_end],
        "Normalized Rxy": rxy_norm[table_corr_start:table_corr_end],
    })

    st.dataframe(
        corr_table,
        height=320,
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Nilai ditampilkan di sekitar lag {selected_lag:+d}.")

  # INFORMASI KORELASI
  with st.container(border=True):
    st.markdown(
        "<p style='color:#ff1493;font-weight:bold'>" "Ringkasan Analisis</p>",
        unsafe_allow_html=True,
    )
    st.write(f"**Lag saat ini:** {selected_lag:+d}")
    st.write(f"**Normalized Rxy saat ini:** {selected_norm:.5f}")
    st.write(f"**Lag puncak positif:** {peak_positive_lag:+d}")
    st.write(f"**Nilai puncak positif:** {peak_positive_value:.5f}")
    st.write(f"**Lag puncak absolut:** {peak_abs_lag:+d}")
    st.write(f"**Nilai pada lag 0:** {zero_norm:.5f}")

    if abs(selected_norm) >= 0.7:
      st.success("Korelasi ternormalisasi relatif kuat pada lag terpilih.")
    elif abs(selected_norm) >= 0.3:
      st.info("Korelasi ternormalisasi sedang pada lag terpilih.")
    else:
      st.warning("Korelasi ternormalisasi relatif rendah pada lag terpilih.")

  # DOWNLOAD HASIL
  full_corr_table = pd.DataFrame(
      {"Lag": lags, "Rxy": rxy, "Normalized Rxy": rxy_norm}
  )
  st.download_button(
      "⬇️ Download Cross Correlation",
      data=full_corr_table.to_csv(index=False).encode("utf-8"),
      file_name="cross_correlation.csv",
      mime="text/csv",
      use_container_width=True,
  )

  full_signal_table = pd.DataFrame(
      {"Index": np.arange(n), "Heel (x)": heel, "Toe (y)": toe}
  )
  st.download_button(
      "⬇️ Download Sinyal",
      data=full_signal_table.to_csv(index=False).encode("utf-8"),
      file_name="gait_signal.csv",
      mime="text/csv",
      use_container_width=True,
  )

# =========================================================
# CATATAN METODE
# =========================================================
st.divider()
st.caption(
    "Metode: cross-correlation dihitung dari sinyal Heel dan Toe "
    "yang dikurangi rata-ratanya. Normalized Rxy dibagi akar "
    "hasil kali energi kedua sinyal. Nilai ini berada pada "
    "rentang -1 hingga +1."
)
