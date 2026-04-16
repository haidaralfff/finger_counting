# 📖 Panduan Cepat - Finger Counting System

**Untuk dokumentasi lengkap**, lihat [DOKUMENTASI_LENGKAP.md](DOKUMENTASI_LENGKAP.md)

---

## 🎯 Apa Itu Proyek Ini?

Sistem **deteksi & penghitung jumlah jari realtime** menggunakan:
- 📸 **Pengolahan Citra**: Color conversion, filtering, thresholding, morphology, contour detection
- 🤖 **Computer Vision**: MediaPipe Hand Detection (21 landmark points)
- 🧠 **Machine Learning**: Random Forest Classifier
- 🎙️ **Audio Output**: Text-to-Speech (Bahasa Indonesia)

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install opencv-python mediapipe scikit-learn gTTS playsound numpy
```

### 2. Prepare Dataset
```
dataset_fingers/
├── 0/ → 30+ images (0 jari/fist)
├── 1/ → 30+ images (1 jari)
├── 2/ → 30+ images (2 jari)
├── 3/ → 30+ images (3 jari)
├── 4/ → 30+ images (4 jari)
└── 5/ → 30+ images (5 jari/open hand)
```

### 3. Run Program
```bash
python main.py
# Press 'Q' to quit
```

---

## 🏗️ Arsitektur Singkat

```
Webcam Input (640×480)
        ↓
┌─────────────────────────────┐
│ IMAGE PROCESSING            │
│ ├─ Grayscale conversion     │
│ ├─ Gaussian blur (5×5)      │
│ ├─ Thresholding (Otsu)      │
│ ├─ Morphology (close+open)  │
│ └─ Contour detection        │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│ HAND DETECTION (MediaPipe)  │
│ └─ 21 landmark points       │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│ FEATURE EXTRACTION          │
│ └─ 63D vector (21×3)        │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│ ML PREDICTION               │
│ ├─ Random Forest (100 trees)│
│ └─ Confidence check (50%)   │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│ SMOOTHING & OUTPUT          │
│ ├─ Majority voting (3-frame)│
│ ├─ Visual overlay           │
│ └─ Audio (gTTS Indonesian)  │
└─────────────────────────────┘
        ↓
   Display & Sound
```

---

## 📊 Pipeline Machine Learning

### Training Phase
```
Load Images (0-5 folders)
    ↓
Extract 21 landmarks per image (63D vector)
    ↓
StandardScaler Normalization
    ↓
Train RandomForestClassifier (100 trees)
    ↓
Save: finger_model.pkl, scaler.pkl
```

### Inference Phase
```
Real-time Frame
    ↓
Hand Detection (MediaPipe)
    ↓
Extract 63D landmark vector
    ↓
Scale with StandardScaler
    ↓
Random Forest Prediction + Confidence
    ↓
Confidence >= 50%? 
├─ YES → Accept ML prediction
└─ NO → Use fallback heuristic
    ↓
Majority Voting (3-frame window)
    ↓
Output: Finger count (0-5)
```

---

## 📁 File Structure

```
FINGER COUTING-PENGOLAHAN CITRA/
├── main.py                      ← Main program
├── docs.md                      ← Quick guide (ini)
├── docs system.md               ← Old docs (deprecated)
├── DOKUMENTASI_LENGKAP.md       ← Complete docs ⭐
├── finger_model.pkl             ← Trained model (auto-created)
├── scaler.pkl                   ← Scaler weights (auto-created)
└── dataset_fingers/
    ├── 0/ (≥30 images: fist)
    ├── 1/ (≥30 images: 1 finger)
    ├── 2/ (≥30 images: 2 fingers)
    ├── 3/ (≥30 images: 3 fingers)
    ├── 4/ (≥30 images: 4 fingers)
    └── 5/ (≥30 images: open hand)
```

---

## 🖼️ Pengolahan Citra - Tahapan

| # | Tahap | Fungsi | Input | Output |
|---|-------|--------|-------|--------|
| 1 | Color Conversion | BGR→Grayscale, BGR→RGB | RGB frame | Gray image |
| 2 | Gaussian Blur | Reduce noise (5×5 kernel) | Gray image | Blurred gray |
| 3 | Thresholding | Otsu's + Adaptive | Blurred | Binary image (0/255) |
| 4 | Morphology | Close + Open (5×5) | Binary | Cleaned binary |
| 5 | Contour Detection | Find object boundaries | Binary | Contours list |
| 6 | Hand Landmarks | MediaPipe 21 points | RGB frame | 63D feature vector |
| 7 | Visualization | Draw on frame | Features | Annotated frame |

---

## 🤖 Machine Learning Details

### Model: Random Forest
```
Classifier: RandomForestClassifier(n_estimators=100, random_state=42)
Features: 63 dimensions (21 landmarks × 3 coordinates: x, y, z)
Classes: 6 (0, 1, 2, 3, 4, 5 jari)
Preprocessing: StandardScaler normalization
```

### Prediction Process
```python
# Load model & scaler
model = pickle.load('finger_model.pkl')
scaler = pickle.load('scaler.pkl')

# Preprocess
features_scaled = scaler.transform(features)

# Predict
prediction = model.predict(features_scaled)[0]      # 0-5
probabilities = model.predict_proba(features_scaled)[0]
confidence = max(probabilities)

# Check confidence
if confidence >= 0.5:
    output = prediction
else:
    output = fallback_heuristic(landmarks)
```

### Fallback Heuristic
Jika model confidence < 50%, gunakan analisis manual:
```python
def count_fingers(landmarks):
    count = 0
    # Thumb: if tip.x > base.x → extended
    if landmarks[4].x > landmarks[3].x:
        count += 1
    # Others: if tip.y < joint.y → extended  
    for tip in [8, 12, 16, 20]:
        if landmarks[tip].y < landmarks[tip-2].y:
            count += 1
    return count
```

---

## ⚙️ Konfigurasi Kunci

```python
# Image Processing
BLUR_KERNEL = (5, 5)
MORPH_KERNEL = (5, 5)
MORPH_SHAPE = cv2.MORPH_ELLIPSE

# Hand Detection
MAX_HANDS = 1

# Model
N_ESTIMATORS = 100
CONFIDENCE_THRESHOLD = 0.5

# Smoothing
SMOOTHING_WINDOW = 3          # Majority voting
FRAME_SKIP = 3                # Process every 3rd frame

# Audio
AUDIO_COOLDOWN = 2            # Seconds
TTS_LANGUAGE = 'id'           # Indonesian
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| **Q** | Quit program |
| (none) | Point hand at camera → auto-detect |

---

## 📊 Dataset Checklist

```
For each class (0-5 jari):
✅ At least 50-100 images
✅ Varied lighting (3+ conditions)
✅ Varied angles (5+ orientations)
✅ Varied distances (3+ depths)
✅ Simple background
✅ High contrast with hand
✅ Full hand visible (no occlusion)
✅ 640×480 resolution (or larger)
✅ JPG/PNG format

Expected accuracy: 85-95% (with good dataset)
```

---

## 🔍 Debugging Tips

### Model not training?
```
→ Check if dataset_fingers/ has images
→ Verify folder names: 0, 1, 2, 3, 4, 5
→ Check image formats (.jpg, .png)
```

### Webcam not working?
```
→ Check camera connection (USB / built-in)
→ Try different camera index (cv2.VideoCapture(0/1/2))
→ Check camera permissions
```

### No hand detected?
```
→ Improve lighting
→ Ensure hand is visible & in frame
→ Check MediaPipe installation
```

### Audio not playing?
```
→ pip install --upgrade playsound gTTS
→ Check system volume
→ Test with: python -c "from playsound import playsound; playsound('file.mp3')"
```

---

## 📚 dokumentation Map

| File | Konten |
|------|--------|
| **docs.md** (ini) | Quick reference, cheat sheet |
| **DOKUMENTASI_LENGKAP.md** | Everything: architecture, image processing, ML, workflow, dataset template |
| **docs system.md** | Legacy documentation (deprecated) |
| **main.py** | Source code with inline comments |

---

## 🚀 Workflow Diagram

```
START
  │
  ├─→ Load model (or train if missing)
  │
  ├─→ Open webcam
  │
  LOOP (30+ FPS):
  ├─→ Capture frame
  │
  ├─→ Process image (7 stages)
  │
  ├─→ Detect hand landmarks (every 3rd frame)
  │
  ├─→ Extract 63D features
  │
  ├─→ ML prediction + fallback
  │
  ├─→ Smooth with majority voting
  │
  ├─→ Display + Audio (if changed)
  │
  └─→ Continue until Q pressed
  │
  ├─→ Cleanup
  │
  END
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| FPS | 30-60 |
| Latency | 100-200ms |
| Accuracy | 85-95% |
| Model Size | 5-10 MB |
| Memory | 300-500 MB |

---

## 📞 Need Help?

1. **Detailed explanation**: Read [DOKUMENTASI_LENGKAP.md](DOKUMENTASI_LENGKAP.md)
2. **Code details**: Check inline comments in `main.py`
3. **Troubleshooting**: See Debugging Tips section above

---

**Last Updated**: April 2026  
**Version**: 1.0  
**Status**: ✅ Complete
