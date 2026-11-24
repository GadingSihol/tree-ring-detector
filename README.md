# 🌲 Tree Ring Detector & Age Estimator

Aplikasi berbasis **Streamlit** dan **Computer Vision** untuk mendeteksi serta menghitung jumlah lingkaran tahun (*tree rings*) pada penampang batang pohon guna mengestimasi usia pohon secara otomatis.

## ✨ Fitur Utama
- **Pra-Pemrosesan Kontras Tinggi:** Menggabungkan kanal pencahayaan **CIELAB ($L^*$)** dan **Blue Channel**, *Bilateral Filtering* (reduksi *noise* serat kayu tanpa memburamkan garis), *CLAHE*, serta *Unsharp Masking*.
- **Segmentasi Otomatis:** Menggunakan deteksi warna HSV untuk foto berlatar studio putih dan AI (`rembg` dengan model ringan `u2netp`) untuk foto berlatar alam/kompleks.
- **Lokalisasi Pusat (*Pith*) & Pelacakan 360°:** Menggunakan *Distance Transform*, gradien *Sobel*, *Polar Remap*, serta *Wrapped Dynamic Programming* dengan estimasi kepadatan cincin adaptif.
- **Dua Mode Input:** Unggah gambar satuan/banyak sekaligus (`.JPG`, `.JPEG`, `.PNG`) atau unggah dataset utuh berformat `.ZIP`.
- **Ekspor Hasil:** Unduh gambar hasil deteksi (`.PNG`) dan tabel rekapitulasi hasil (`.CSV`).

---

## 📁 Struktur Proyek

```text
Tree_Ring_Detector/
├── .streamlit/
│   └── config.toml          # Konfigurasi tema gelap (Dark Mode) & batas upload 500 MB
├── app.py                   # Kode utama aplikasi Streamlit & algoritma deteksi
├── requirements.txt         # Daftar library Python yang dibutuhkan
└── README.md                # Dokumentasi dan panduan penggunaan
```

---

## 💻 Cara Menjalankan di Perangkat Sendiri (Lokal)

### 1. Clone atau Unduh Proyek
Pastikan seluruh file proyek (`app.py`, `requirements.txt`, dan folder `.streamlit`) berada dalam satu folder yang sama, lalu buka folder tersebut di terminal / VS Code.

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

Terdapat **3 cara** agar teman, dosen, atau penguji dapat mencoba aplikasi ini di laptop maupun HP mereka:

### Opsi 1: Berbagi Lewat Satu Jaringan Wi-Fi / Hotspot (Tanpa Instalasi di HP/Laptop Penguji)
1. Pastikan perangkat kamu dan perangkat penguji terhubung ke **jaringan Wi-Fi atau Hotspot yang sama**.
2. Jalankan aplikasi di laptopmu dengan perintah:
   ```bash
   streamlit run app.py --server.address 0.0.0.0
   ```
3. Lihat alamat **Network URL** yang muncul di terminal (contoh: `http://192.168.1.15:8501`).
4. Bagikan tautan **Network URL** tersebut ke penguji untuk dibuka langsung di browser HP atau laptop mereka.

### Opsi 2: Deploy Gratis ke Streamlit Community Cloud (Akses Online dari Mana Saja)
1. Unggah folder proyek ini (`app.py`, `requirements.txt`, dan `.streamlit/config.toml`) ke repositori **GitHub** publik.
2. Buka [share.streamlit.io](https://share.streamlit.io) dan masuk