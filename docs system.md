# 📋 Dokumentasi Proyek: Finger Counting - Pengolahan Citra

## 🎯 Deskripsi Proyek

Proyek ini adalah **sistem deteksi dan penghitung jumlah jari** menggunakan teknologi **pengolahan citra** dan **machine learning**. Sistem ini menggunakan webcam real-time untuk mendeteksi tangan dan menghitung jumlah jari yang terbuka, dengan output suara bahasa Indonesia.

---

## 📁 Struktur Proyek

```
FINGER COUTING-PENGOLAHAN CITRA/
├── docs.md                    # File dokumentasi (ini)
├── main.py                    # Script utama deteksi finger counting
├── finger_model.pkl           # Model ML yang sudah dilatih (cache)
├── scaler.pkl                 # StandardScaler untuk preprocessing
└── dataset_fingers/           # Dataset training (6 folder)
    ├── 0/                     # Dataset dengan 0 jari
    ├── 1/                     # Dataset dengan 1 jari
    ├── 2/                     # Dataset dengan 2 jari
    ├── 3/                     # Dataset dengan 3 jari
    ├── 4/                     # Dataset dengan 4 jari
    └── 5/                     # Dataset dengan 5 jari
```

---

## ⚙️ Teknologi yang Digunakan

### 1. **Perpustakaan Utama**
| Library | Fungsi | Versi |
|---------|--------|-------|
| **OpenCV (cv2)** | Pemrosesan citra & video | Latest |
| **MediaPipe** | Hand detection & landmark | Latest |
| **scikit-learn** | Machine learning (Random Forest) | Latest |
| **gTTS** | Text-to-speech (Bahasa Indonesia) | Latest |
| **playsound** | Audio playback | Latest |
| **NumPy** | Operasi numerik | Latest |

### 2. **Teknologi Machine Learning**
- **Model**: Random Forest Classifier (100 estimators)
- **Fitur**: 63 landmark points dari MediaPipe (21 points × 3 dimensi: x, y, z)
- **Preprocessing**: StandardScaler untuk normalisasi
- **Output**: Klasifikasi jumlah jari (0-5)

---

## 🖼️ PENGOLAHAN CITRA DI PROYEK INI

### ✅ **ADA! Proyeknya Menggunakan Pengolahan Citra**

Proyek ini **SANGAT BANYAK** menggunakan teknik pengolahan citra. Berikut detail lengkapnya:

### **1. Konversi Warna (Color Space Conversion)**
```python
# Grayscale Conversion
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# RGB untuk MediaPipe hand detection
rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
```
- Mengubah frame dari BGR ke Grayscale untuk analisis intensitas piksel
- Mengubah ke RGB untuk deteksi tangan dengan MediaPipe

---

### **2. Filtering (Penyaringan Citra)**
```python
# Gaussian Blur - smoothing untuk noise reduction
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
```
- **Fungsi**: Mengurangi noise dan membuat citra lebih halus
- **Kernel**: 5×5 Gaussian kernel
- **Impact**: Meningkatkan kualitas binerisasi

---

### **3. Thresholding & Binarization (Ambang Batas)**
```python
# Otsu's Thresholding - mencari threshold otomatis yang optimal
_, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive Thresholding - untuk pencahayaan tidak merata
binary_adaptive = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
```
- **Otsu's Method**: Otomatis menentukan threshold yang optimal
- **Adaptive Threshold**: Cocok untuk kondisi pencahayaan yang tidak konsisten
- **Output**: Binary image (hitam putih) untuk analisis struktur

---

### **4. Morphological Operations (Operasi Morfologi)**
```python
# Structuring Element
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Closing - Dilasi + Erosi (menutup lubang kecil)
closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Opening - Erosi + Dilasi (menghilangkan noise kecil)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
```
- **Closing**: Menutup hole kecil di dalam objek
- **Opening**: Menghilangkan noise & partikel kecil
- **Kernel Shape**: Ellipse (circular structure element)

---

### **5. Contour Detection (Deteksi Kontur)**
```python
contours, hierarchy = cv2.findContours(
    binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)
```
- **Fungsi**: Mendeteksi batas objek dalam citra biner
- **Retrieval**: RETR_TREE (semua contour dengan hirarki)
- **Approximation**: CHAIN_APPROX_SIMPLE (simplifikasi contour)

---

### **6. Hand Landmark Detection (MediaPipe)**
```python
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
result = hands.process(rgb)

# 21 landmark points per tangan
landmarks = result.multi_hand_landmarks[0].landmark
features = []
for lm in landmarks:
    features.extend([lm.x, lm.y, lm.z])  # x, y, z coordinates
```
- **21 Points**: Deteksi 21 landmark pada tangan
- **3D Coordinates**: x, y (2D) + z (depth estimate)
- **Output Feature Vector**: 63 dimensi (21 × 3)

---

### **7. Image Visualization & Rendering**
```python
# Draw hand landmarks
mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# Text overlay
cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

# Image resizing & concatenation
frame_resized = cv2.resize(frame, size)
combined = np.hstack([...])  # Horizontal concat
combined = np.vstack([...])  # Vertical concat
```
- Menampilkan landmark dan koneksi di frame asli
- Visualisasi text dengan informasi real-time
- Membuat composite image dari berbagai tahap processing

---

## 🎛️ Fungsi-Fungsi Pengolahan Citra

### **Function 1: `image_processing_threshold(frame)`**
**Tujuan**: Melakukan thresholding dan binarization pada frame
```python
def image_processing_threshold(frame):
    """
    Returns: gray, binary_otsu, binary_adaptive
    """
```
**Langkah**:
1. Konversi BGR → Grayscale
2. Gaussian Blur (5×5)
3. Otsu's Thresholding
4. Adaptive Thresholding

---

### **Function 2: `morphological_operations(binary_image)`**
**Tujuan**: Membersihkan binary image menggunakan operasi morfologi
```python
def morphological_operations(binary_image):
    """
    Closing + Opening untuk noise reduction
    """
```
**Langkah**:
1. Create structuring element (kernel 5×5 ellipse)
2. Morphological Closing (MORPH_CLOSE)
3. Morphological Opening (MORPH_OPEN)

---

### **Function 3: `detect_contours(binary_image)`**
**Tujuan**: Deteksi contour dari binary image
```python
def detect_contours(binary_image):
    """
    Returns: contours, hierarchy
    """
```
**Output**: List of contours dengan informasi hirarki

---

### **Function 4: `visualize_processing_steps(...)`**
**Tujuan**: Menampilkan semua tahap pengolahan dalam satu window
```python
def visualize_processing_steps(frame, gray, binary_otsu, binary_adaptive, morpho):
    """
    Gabung 4 image dalam 2×2 grid dengan label
    """
```
**Output**: Composite image dengan 4 tahap berbeda

---

## 🤖 Pipeline Machine Learning

### **1. Data Loading & Preprocessing**
```
Dataset (0-5 jari) 
    ↓
Per-image: MediaPipe extraction (21 landmark × 3 = 63 features)
    ↓
Stack into feature matrix X
    ↓
StandardScaler normalization
```

### **2. Model Training**
```
Random Forest Classifier
├── n_estimators: 100 trees
├── random_state: 42 (reproducible)
└── Output: fingers_count (0-5)
```

### **3. Prediction Pipeline**
```
Real-time Frame
    ↓
Hand Detection (MediaPipe)
    ↓
Extract 63-D feature vector
    ↓
Scale dengan StandardScaler
    ↓
Random Forest prediction
    ↓
Confidence threshold check (50%)
    ↓
Smoothing dengan majority voting (3-frame window)
    ↓
Output: fingers_count + confidence
```

---

## 🔄 Algoritma Smoothing (Majority Voting)

Program menggunakan **majority voting** untuk menstabilkan prediksi:

```python
def smooth_prediction(prediction, history, window_size=3):
    """
    Menggunakan 3 frame sebelumnya untuk weighted voting
    """
```

**Logika**:
- Collect 3 prediksi terakhir
- Hitung modus (most frequent value)
- Return hasil yang paling sering muncul
- **Benefit**: Mengurangi noise dan false positive

---

## 🎙️ Text-to-Speech Integration

Program menggunakan **gTTS** (Google Text-to-Speech) dengan bahasa Indonesia:

```python
def speak(text):
    tts = gTTS(text=text, lang='id')  # Bahasa Indonesia
    tts.save(filename)
    playsound(filename)
```

**Features**:
- **Cooldown**: 2 detik anti-spam
- **Threading**: Non-blocking audio playback
- **Cleanup**: Auto-remove file setelah diplay

---

## 📊 Fitur & Optimasi

### **1. Frame Skipping**
```python
frame_skip = 3  # Process setiap 3 frame
if frame_count % frame_skip == 0:
    # Deteksi & prediksi
```
- **Benefit**: Mengurangi beban komputasi
- **Trade-off**: Sedikit lag dalam deteksi

### **2. Confidence Thresholding**
```python
confidence_threshold = 0.5  # 50%
if confidence < confidence_threshold:
    # Gunakan fallback
```
- **Safety**: Hanya trust prediksi dengan confidence ≥ 50%
- **Fallback**: Gunakan heuristic jika confidence rendah

### **3. Fallback Heuristic (Jika model tidak ada)**
```python
def count_fingers_fallback(lm):
    """
    Heuristic berdasarkan landmark position
    """
```
- Ibu jari: Check x-coordinate
- Jari lain: Check y-coordinate vs tip position

### **4. Model Caching**
- Jika model sudah ada → Load dari pickle
- Jika belum → Train dari dataset (sekali saja)

---

## 🚀 Cara Menjalankan Program

### **Prerequisite**
```bash
pip install opencv-python mediapipe scikit-learn gtts playsound numpy
```

### **Persiapan Dataset**
```
dataset_fingers/
├── 0/  ← Foto tangan dengan 0 jari (telapak terbuka)
├── 1/  ← Foto tangan dengan 1 jari
├── 2/  ← Foto tangan dengan 2 jari
├── 3/  ← Foto tangan dengan 3 jari
├── 4/  ← Foto tangan dengan 4 jari
└── 5/  ← Foto tangan dengan 5 jari (semua jari)
```

### **Menjalankan Program**
```bash
python main.py
```

**Kontrol**:
- Tekan **'Q'** untuk keluar
- Arahkan tangan ke kamera
- Program akan berbicara jumlah jari yang terdeteksi

---

## 📈 Output & Monitoring

Program menampilkan informasi real-time di frame:

```
Jumlah Jari: 3 [ML]              ← Model prediction
Confidence: 95.2%                ← Confidence score
Frame: 243 | Smoothing: 3/3      ← Frame info
```

**Console Output**:
```
✓ Perubahan terdeteksi: -1 -> 0 jari
✓ Perubahan terdeteksi: 0 -> 3 jari
✓ Perubahan terdeteksi: 3 -> 5 jari
```

---

## 🎓 Teknik Pengolahan Citra yang Digunakan

| # | Teknik | Fungsi | Lokasi |
|----|--------|--------|--------|
| 1 | Color Space Conversion | Ubah BGR→Gray, BGR→RGB | `image_processing_threshold()` |
| 2 | Gaussian Blur | Noise reduction | `image_processing_threshold()` |
| 3 | Otsu's Thresholding | Automatic threshold | `image_processing_threshold()` |
| 4 | Adaptive Thresholding | Uneven lighting handling | `image_processing_threshold()` |
| 5 | Morphological Closing | Close holes in objects | `morphological_operations()` |
| 6 | Morphological Opening | Remove small noise | `morphological_operations()` |
| 7 | Contour Detection | Find object boundaries | `detect_contours()` |
| 8 | Landmark Detection | Hand keypoint extraction | MediaPipe native |
| 9 | Image Resizing | Scale untuk display | `visualize_processing_steps()` |
| 10 | Image Concatenation | Composite visualization | `visualize_processing_steps()` |

---

## 💾 File Output & Cache

Program secara otomatis membuat file:

```
finger_model.pkl      ← Trained Random Forest model (binary)
scaler.pkl            ← StandardScaler object (binary)
voice.mp3             ← Temporary audio file (dihapus auto)
```

---

## ⚠️ Error Handling

Program memiliki error handling untuk:
- ✅ Webcam tidak found → exit gracefully
- ✅ Dataset folder kosong → fallback ke heuristic
- ✅ Model training gagal → use fallback method
- ✅ Confidence rendah → fallback ke heuristic
- ✅ Audio playback error → log warning, continue

---

## 🔬 Keakuratan & Limitations

### **Faktor yang Mempengaruhi Akurasi**:
- ✅ Ukuran & kualitas dataset training
- ✅ Pencahayaan (brightness & contrast)
- ✅ Jarak tangan dari kamera
- ✅ Background color vs hand color
- ✅ Gesture variety (berbagai posisi jari)

### **Optimasi**:
- Frame skipping mengurangi lag
- Majority voting smoothing mengurangi noise
- Confidence threshold mencegah false positive
- Fallback method memberikan robustness

---

## 📝 Catatan Teknis

- **Frame Resolution**: 640×480 (dikecilkan 50% untuk processing)
- **Processing Frequency**: Setiap 3 frame (untuk efisiensi)
- **Smoothing Window**: 3 frame terakhir
- **Hand Detection**: Max 1 tangan per frame
- **Confidence Threshold**: 50%
- **Model Features**: 63 dimensi (21 landmark × 3 axis)
- **Model Accuracy**: Ditampilkan saat training selesai

---

## 📚 Referensi & Resources

- **OpenCV**: https://docs.opencv.org/
- **MediaPipe**: https://mediapipe.dev/
- **scikit-learn**: https://scikit-learn.org/
- **Morphological Operations**: Digital Image Processing concepts
- **Finger Counting**: Hand gesture recognition field

---

## ✨ Kesimpulan

**Proyek ini DEFINIT menggunakan pengolahan citra secara ekstensif**, meliputi:
- ✅ Konversi warna (Color space)
- ✅ Filtering (Gaussian Blur)
- ✅ Thresholding & Binarization
- ✅ Morphological Operations
- ✅ Contour Detection
- ✅ Landmark-based Feature Extraction
- ✅ Real-time Video Processing
- ✅ Image Visualization & Compositing

Kombinasi dengan Machine Learning (Random Forest) membuat sistem ini **robust dan akurat** untuk deteksi jumlah jari secara real-time.

---

**Dokumentasi selesai: 16 April 2026**
