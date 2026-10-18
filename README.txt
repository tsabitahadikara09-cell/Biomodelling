# Gait Signal Analysis — Streamlit UI

## Menjalankan aplikasi
1. Ekstrak folder/ZIP.
2. Buka terminal di folder ini.
3. Instal dependensi:
   `pip install -r requirements.txt`
4. Jalankan:
   `streamlit run app.py`

## Format data
Mendukung CSV, TXT, XLS, XLSX. Pilih kolom di bagian **Pemetaan Kolom Data**.
Kolom yang disarankan:
- Sample / waktu
- Heel (x)
- Toe (y)
- Cross correlation (Rxy)
- Normalization

Jika belum upload file, grafik dummy hanya untuk pratinjau tampilan. Nilai korelasi dan normalisasi tidak dihitung ulang otomatis oleh UI versi awal.
