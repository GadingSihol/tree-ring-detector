# 🌲 Tree Ring Detector & Age Estimator

Aplikasi berbasis **Streamlit** dan **Computer Vision** untuk mendeteksi serta menghitung jumlah lingkaran tahun (*tree rings*) pada penampang batang pohon guna mengestimasi usia pohon secara otomatis.

## ✨ Fitur Utama
- **Pra-Pemrosesan Kontras Tinggi:** Menggabungkan kanal pencahayaan **CIELAB ($L^*$)** dan **Blue Channel**, *Bilateral Filtering* (reduksi *noise* serat kayu tanpa memburamkan garis), *CLAHE*, serta *Unsharp Masking*.
- **Segmentasi Otomatis:** Menggunakan deteksi warna HSV untuk foto berlatar studio putih dan AI (`rembg` dengan model ringan `u2netp`) untuk foto berlatar alam/kompleks.
- **Lokalisasi Pusat (*Pith*) & Pelacakan 360°:** Menggunakan *Distance Transform*, gradien *Sobel*, *Polar Remap*, serta *Wrapped Dynamic Programming* dengan estimasi kepadatan cincin adaptif.
- **Dua Mode Input:** Unggah gambar satuan/banyak sekaligus (`.JPG`, `.JPEG`, `.PNG`) atau unggah dataset utuh berformat `.ZIP`.
- **Ekspor Hasil:** Unduh gambar hasil deteksi (`.PNG`) dan tabel rekapitulasi hasil (`.CSV`).

---

## 📦 Unduh Dataset Lengkap
Karena ukuran dataset melebihi batas ukuran file repositori GitHub (100 MB), file dataset disimpan secara terpisah. Unduh dataset lengkap melalui tautan berikut:

- **Google Drive:** [Download Dataset_Tree Rings.zip](http://drive.google.com/drive/folders/1ImW1HAowRYFyEkHqa8R9DdDRrtXSG-IE)

Setelah diunduh, letakkan file `Dataset_Tree Rings.zip` ke dalam folder `Streamlit/dataset/` (atau langsung unggah melalui menu **Upload Dataset (.ZIP)** di antarmuka aplikasi).

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

## 🌐 Cara Agar Orang Lain Bisa Ikut Mengetes di Perangkat Mereka

### Opsi 1: Berbagi Lewat Satu Jaringan Wi-Fi / Hotspot (Tanpa Instalasi di HP/Laptop Penguji)
1. Pastikan perangkat kamu dan perangkat penguji terhubung ke **jaringan Wi-Fi atau Hotspot yang sama**.
2. Jalankan aplikasi di laptopmu dengan perintah:
   ```bash
   streamlit run app.py --server.address 0.0.0.0
   ```
3. Lihat alamat **Network URL** yang muncul di terminal (contoh: `http://192.168.1.15:8501`).
4. Bagikan tautan **Network URL** tersebut ke penguji untuk dibuka langsung di browser HP atau laptop mereka.

### Opsi 2: Deploy Gratis ke Streamlit Community Cloud (Akses Online dari Mana Saja)
1. Buka [share.streamlit.io](https://share.streamlit.io) dan masuk menggunakan akun GitHub.
2. Klik **Create app** → pilih repositori `GadingSihol/tree-ring-detector` → isi **Main file path** dengan `app.py` → klik **Deploy**.
3. Bagikan tautan aplikasi yang muncul (`https://nama-aplikasi.streamlit.app`) ke siapa saja untuk langsung dicoba tanpa perlu menginstal Python.
