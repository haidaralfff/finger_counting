# 📖 User Guide - Panduan Menjalankan Finger Counting Demo

## Cara Menjalankan Aplikasi

### ✅ Prasyarat

Pastikan Anda telah menginstall semua dependencies yang diperlukan.

#### 1. Install Python 3.8+
Download dari https://www.python.org/ (Windows, macOS, atau Linux)

#### 2. Install Dependencies
Buka Command Prompt/Terminal di folder proyek dan jalankan:

```bash
pip install opencv-python mediapipe scikit-learn gTTS playsound numpy
```

**Atau** gunakan file requirements (jika ada):
```bash
pip install -r requirements.txt
```

---

## 🚀 Menjalankan Aplikasi

### Step 1: Persiapan Dataset (Opsional)
Jika Anda ingin melatih ulang model, pastikan folder `dataset_fingers/` sudah berisi:
- `dataset_fingers/0/` - Gambar tangan menutup (0 jari)
- `dataset_fingers/1/` - Gambar 1 jari terbuka
- `dataset_fingers/2/` - Gambar 2 jari terbuka
- `dataset_fingers/3/` - Gambar 3 jari terbuka
- `dataset_fingers/4/` - Gambar 4 jari terbuka
- `dataset_fingers/5/` - Gambar 5 jari terbuka

Setiap folder berisi file gambar (`.jpg`, `.jpeg`, `.png`)

### Step 2: Jalankan Aplikasi

Buka Command Prompt/Terminal di folder proyek dan ketik:

```bash
python main.py
```

### Step 3: Tunggu Inisialisasi

Aplikasi akan menampilkan:
```
==================================================
🚀 INISIALISASI SISTEM DETEKSI JARI
==================================================
```

- **Pertama kali**: Aplikasi akan melatih model dari dataset (mungkin memakan waktu 1-5 menit tergantung ukuran dataset)
- **Kali berikutnya**: Aplikasi akan langsung load model yang sudah tersimpan (lebih cepat ~5 detik)

Tunggu hingga melihat pesan:
```
✅ Model ML berhasil dimuat
```

---

## 🎮 Cara Menggunakan Demo

### Saat Aplikasi Berjalan

1. **Posisikan Tangan**
   - Letakkan tangan Anda di depan webcam
   - Pastikan seluruh tangan terlihat dalam frame
   - Pencahayaan yang baik akan meningkatkan akurasi

2. **Tampilkan Jari**
   - Tutup semua jari (tunjukkan 0 jari)
   - Buka 1 jari, 2 jari, sampai 5 jari secara bergantian

3. **Dengarkan Feedback**
   - Aplikasi akan mengeluarkan suara berbahasa Indonesia
   - Contoh: "Nol", "Satu", "Dua", "Tiga", "Empat", "Lima"

4. **Lihat Hasil di Layar**
   - Frame video akan menampilkan:
     - Skeleton tangan (garis-garis landmark)
     - Jumlah jari yang terdeteksi
     - Confidence score (tingkat kepercayaan prediksi)

### Kontrol Keyboard

| Tombol | Fungsi |
|--------|--------|
| `q` atau `ESC` | Keluar dari aplikasi |
| `r` | Reset/Restart detection |
| `s` | Screenshot hasil deteksi (jika ada) |

---

## 🔧 Troubleshooting

### ❌ Masalah: "ModuleNotFoundError: No module named 'cv2'"
**Solusi:**
```bash
pip install opencv-python
```

### ❌ Masalah: "Webcam tidak terdeteksi"
**Solusi:**
- Periksa izin akses kamera di sistem Anda
- Pastikan tidak ada aplikasi lain menggunakan kamera
- Coba restart aplikasi

### ❌ Masalah: "Dataset kosong / Folder dataset_fingers tidak ditemukan"
**Solusi:**
- Aplikasi akan otomatis menggunakan fallback heuristic method
- Untuk akurasi lebih baik, tambahkan gambar dataset ke folder `dataset_fingers/0/` hingga `dataset_fingers/5/`

### ❌ Masalah: Model tidak akurat
**Solusi:**
- Tambah lebih banyak gambar training ke dataset
- Pastikan lighting/pencahayaan konsisten
- Coba atur `confidence_threshold` di `main.py`
- Hapus file `finger_model.pkl` dan `scaler.pkl` untuk retraining

### ❌ Masalah: Audio tidak terdengar
**Solusi:**
- Periksa volume sistem (jangan di-mute)
- Pastikan speaker/headphone terhubung
- Coba install ulang `playsound`: `pip install --upgrade playsound`

---

## 💡 Tips Optimal

✅ **Untuk Akurasi Terbaik:**
1. Gunakan lighting yang cukup terang (tidak terlalu gelap)
2. Letakkan tangan setidaknya 30cm dari webcam
3. Tunjukkan seluruh tangan tanpa menghalangi landmark
4. Gerakan tangan secara perlahan
5. Tunggu feedback audio sebelum mengubah posisi jari

✅ **Untuk Demo yang Smooth:**
1. Pastikan background tidak terlalu rumit (preferably solid color)
2. Hapus aksesoris di tangan yang dapat menghalangi deteksi (cincin besar, gelang tebal)
3. Gunakan resolusi kamera yang cukup baik

---

## 📊 Output Aplikasi

Saat berjalan, aplikasi menampilkan:
- **Video frame** dengan visualisasi skeleton tangan
- **Jumlah jari terdeteksi** (0-5)
- **Confidence score** (persentase kepercayaan model)
- **Audio feedback** dalam Bahasa Indonesia
- **FPS** (frame per second) untuk performa

---

## 🎯 Demo Singkat

### Skenario Pengujian (2-3 menit):

1. **Mulai aplikasi** → tunggu sampai model loaded
2. **Tunjukkan 0 jari** → dengarkan "Nol"
3. **Tunjukkan 1 jari** → dengarkan "Satu"
4. **Tunjukkan 2 jari** → dengarkan "Dua"
5. **Tunjukkan 3 jari** → dengarkan "Tiga"
6. **Tunjukkan 4 jari** → dengarkan "Empat"
7. **Tunjukkan 5 jari** → dengarkan "Lima"
8. **Tekan `q`** → keluar aplikasi

---

## 📞 Butuh Bantuan?

Jika mengalami masalah:
1. Periksa `docs.md` untuk informasi teknis
2. Baca bagian **Troubleshooting** di atas
3. Periksa console/terminal untuk error messages
4. Pastikan semua dependencies sudah terinstall dengan benar

---

**Selamat Mencoba! 🎉**

*Untuk dokumentasi teknis lengkap, lihat file **docs.md***
