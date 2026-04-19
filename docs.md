# 🖐️ Finger Counting - Aplikasi Penghitungan Jari Berbasis Pengolahan Citra

## Deskripsi Proyek

**Finger Counting** adalah aplikasi yang memanfaatkan teknologi **Pengolahan Citra Digital** dan **Machine Learning** untuk mendeteksi dan menghitung jumlah jari dari tangan yang terlihat di kamera secara real-time.

### Fitur Utama

- 🎥 **Deteksi Tangan Real-Time**: Menggunakan MediaPipe untuk mendeteksi posisi tangan dan landmark (titik kunci) dengan akurasi tinggi
- 🧠 **Machine Learning Classification**: Model Random Forest terlatih untuk mengklasifikasi jumlah jari (0-5)
- 🔊 **Audio Feedback**: Text-to-Speech (gTTS) dalam Bahasa Indonesia untuk memberikan feedback audio
- 📊 **Model Persistence**: Model yang sudah dilatih disimpan untuk penggunaan kembali tanpa perlu training ulang
- ⚡ **Fallback Mechanism**: Heuristic berbasis posisi landmark jika model ML tidak tersedia
- 🎯 **Smoothing Algorithm**: Majority voting untuk hasil prediksi yang lebih stabil

### Teknologi yang Digunakan

| Teknologi | Fungsi |
|-----------|--------|
| **OpenCV** | Capture video dari webcam dan pemrosesan citra |
| **MediaPipe** | Deteksi tangan dan ekstraksi landmark (21 points) |
| **Scikit-learn** | Training model Random Forest Classifier |
| **gTTS** | Text-to-Speech untuk feedback audio |
| **NumPy** | Operasi numerik dan manipulasi array |

### Dataset

Aplikasi menggunakan dataset terstruktur dengan folder:
```
dataset_fingers/
├── 0/  (gambar tangan menutup - 0 jari)
├── 1/  (gambar 1 jari terbuka)
├── 2/  (gambar 2 jari terbuka)
├── 3/  (gambar 3 jari terbuka)
├── 4/  (gambar 4 jari terbuka)
└── 5/  (gambar 5 jari terbuka)
```

### 📊 Sumber Data

Dataset yang digunakan dapat diunduh dari:

**Kaggle Dataset: Counting Fingers Dataset**
- 🔗 **URL**: https://www.kaggle.com/datasets/piyushjoshi01/counting-fingers-dataset/data
- 📝 **Deskripsi**: Dataset publik berisi gambar-gambar tangan dengan jumlah jari yang berbeda (0-5)
- 📦 **Format**: Gambar JPEG/PNG terorganisir dalam folder sesuai jumlah jari
- 📈 **Ukuran**: Dataset komprehensif dengan ribuan gambar training

**Cara Menggunakan Dataset:**
1. Buka link Kaggle di atas
2. Download dataset (memerlukan akun Kaggle gratis)
3. Ekstrak file dan letakkan folder sesuai struktur `dataset_fingers/` di atas
4. Jalankan `main.py` - model akan otomatis melatih ulang

---

## 👨‍💻 Pengembang

- **Nama**: Haidar Habibi Al Farisi
- **Program**: Semester 4 - Ilmu Komputer
- **Mata Kuliah**: Pengolahan Citra Digital
- **Tahun**: 2026

---

## 📋 Persyaratan Sistem

### Hardware Minimum
- Webcam/Kamera terintegrasi
- Processor: Intel Core i5 atau setara
- RAM: 4GB minimum

### Software
- Python 3.8 atau lebih baru
- Operating System: Windows, macOS, atau Linux

---

## 📦 Dependencies

Semua dependencies tertulis dalam baris import di `main.py`:
- `opencv-python` - Pemrosesan video
- `mediapipe` - Deteksi tangan
- `scikit-learn` - Machine Learning
- `gTTS` - Text-to-Speech
- `playsound` - Playback audio
- `numpy` - Komputasi numerik

---

## 🎯 Cara Kerja Sistem

### 1. **Training Phase**
```
Dataset (folder 0-5) 
    ↓
Load & Preprocess Gambar
    ↓
Ekstraksi Landmarks (21 points)
    ↓
Training Random Forest Model
    ↓
Save Model & Scaler
```

### 2. **Inference Phase**
```
Capture Frame dari Webcam
    ↓
Deteksi Tangan (MediaPipe)
    ↓
Ekstraksi Landmarks
    ↓
Load Model & Predict
    ↓
Smoothing (Majority Voting)
    ↓
Display & Audio Feedback
```

---

## 📝 Catatan

- Aplikasi akan otomatis melatih model saat pertama kali dijalankan (jika dataset tersedia)
- Hasil prediksi di-smooth menggunakan majority voting untuk stabilitas
- Confidence threshold dapat diatur untuk meningkatkan akurasi
- Audio feedback mencegah spam dengan cooldown 2 detik

---

*Untuk panduan menjalankan aplikasi, lihat file **userguide.md***
