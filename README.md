# 🌲 Tree Ring Detector & Age Estimator

Aplikasi berbasis **Streamlit** dan **Computer Vision** untuk mendeteksi serta menghitung jumlah lingkaran tahun (*tree rings*) pada penampang batang pohon guna mengestimasi usia pohon secara otomatis.

🌐 **Live Demo:** [Buka Aplikasi Tree Ring Detector](https://tree-ring-detector.streamlit.app/)

---

## ✨ Fitur Utama
- **Pra-Pemrosesan Kontras Tinggi:** Menggabungkan kanal pencahayaan **CIELAB ($L^*$)** dan **Blue Channel**, *Bilateral Filtering* (reduksi *noise* serat kayu tanpa memburamkan garis), *CLAHE*, serta *Unsharp Masking*.
- **Segmentasi Otomatis:** Menggunakan deteksi warna HSV untuk foto berlatar studio putih dan AI (`rembg` dengan model ringan `u2netp`) untuk foto berlatar alam/kompleks.
- **Lokalisasi Pusat (*Pith*) & Pelacakan 360°:** Menggunakan *Distance Transform*, gradien *Sobel*, *Polar Remap*, serta *Wrapped Dynamic Programming* dengan estimasi kepadatan cincin adaptif.
- **Dua Mode Input:** Unggah gambar satuan/banyak sekaligus (`.JPG`, `.JPEG`, `.PNG`) atau jalankan pengujian massal menggunakan dataset berformat `.ZIP`.
- **Ekspor Hasil:** Unduh gambar hasil deteksi (`.PNG`) dan tabel rekapitulasi hasil (`.CSV`).

---

## 📦 Sumber & Unduh Dataset
Karena ukuran dataset melebihi batas ukuran file repositori GitHub (100 MB), file dataset disimpan secara terpisah. Anda dapat mengakses sumber dataset asli maupun mengunduh paket `.ZIP` siap pakai melalui tautan berikut:

- **Sumber Dataset Asli (Zenodo):** [Zenodo Record #15653340](https://zenodo.org/records/15653340)
- **Dataset Proyek Siap Pakai (Google Drive):** [Download Dataset_Tree Rings.zip](http://drive.google.com/drive/folders/1ImW1HAowRYFyEkHqa8R9DdDRrtXSG-IE)

Setelah mengunduh `Dataset_Tree Rings.zip` dari Google Drive, letakkan file tersebut ke dalam folder `Streamlit/dataset/` untuk deteksi otomatis di mode lokal, atau unggah langsung melalui menu **Dataset Proyek (.ZIP)** di antarmuka aplikasi.

---

## 📁 Struktur Proyek

```text
Tree_Rings/
├── .streamlit/
│   └── config.toml                 # Konfigurasi tema gelap (Dark Mode) & batas upload 500 MB
├── Streamlit/
│   └── dataset/
│       └── Dataset_Tree Rings.zip  # Diunduh terpisah dari Google Drive (tidak ikut di-push ke Git)
├── .gitignore                      # Pengecualian folder venv, cache, dan file .zip besar
├── app.py                          # Kode utama aplikasi Streamlit & algoritma deteksi
├── requirements.txt                # Daftar library Python yang dibutuhkan
└── README.md                       # Dokumentasi dan panduan penggunaan
```

---

## 💻 Cara Menjalankan di Perangkat Sendiri (Lokal)

### 1. Clone Repositori & Siapkan Dataset
```bash
git clone [https://github.com/GadingSihol/tree-ring-detector.git](https://github.com/GadingSihol/tree-ring-detector.git)
cd tree-ring-detector
```
*(Opsional) Buat folder `Streamlit/dataset/` lalu masukkan file `Dataset_Tree Rings.zip` yang sudah diunduh dari tautan Google Drive di atas.*

### 2. Buat & Aktifkan Virtual Environment (`venv`)
Disarankan menggunakan **Python 3.12** agar seluruh *library* berjalan optimal:

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan di Windows (Command Prompt)
venv\Scripts\activate

# Aktifkan di Windows (PowerShell)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1

# Aktifkan di macOS / Linux
source venv/bin/activate
```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```
*(Catatan untuk pengguna Windows: Jika muncul error `DLL load failed while importing cv2` karena kebijakan keamanan Windows, jalankan `pip uninstall -y opencv-python` lalu `pip install opencv-python-headless`)*.

### 4. Jalankan Aplikasi
```bash
streamlit run app.py
```
Browser akan otomatis terbuka di alamat `http://localhost:8501`.

---

## 🌐 Cara Lain untuk Mengetes Aplikasi
### Akses Online Langsung (Tanpa Instalasi)
Buka tautan aplikasi yang sudah di-*deploy* di **[tree-ring-detector.streamlit.app](https://tree-ring-detector.streamlit.app/)** melalui browser HP atau laptop untuk langsung menguji deteksi lingkaran pohon secara online.
