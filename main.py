import cv2
import mediapipe as mp
from gtts import gTTS
from playsound import playsound
import threading
import os
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ✅ Path dataset
DATASET_PATH = "dataset_fingers"
MODEL_PATH = "finger_model.pkl"
SCALER_PATH = "scaler.pkl"

# ✅ Fungsi load dataset dari folder
def load_dataset():
    X = []
    y = []
    
    for finger_count in range(6):  # 0-5 jari
        folder_path = os.path.join(DATASET_PATH, str(finger_count))
        
        if not os.path.exists(folder_path):
            print(f"⚠️ Folder {folder_path} tidak ditemukan!")
            continue
        
        for img_file in os.listdir(folder_path):
            if img_file.endswith(('.jpeg', '.jpg', '.png')):
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                # Ekstrak fitur dari gambar
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                
                if result.multi_hand_landmarks:
                    # Ambil landmark
                    landmarks = result.multi_hand_landmarks[0].landmark
                    features = []
                    
                    for lm in landmarks:
                        features.extend([lm.x, lm.y, lm.z])
                    
                    X.append(features)
                    y.append(finger_count)
    
    return np.array(X), np.array(y)

# ✅ Fungsi train/load model
def train_or_load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("✅ Loading model yang sudah ada...")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    
    print("🔄 Training model dari dataset...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("❌ Dataset kosong! Gunakan fallback count_fingers()")
        return None, None
    
    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Simpan model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✅ Model berhasil dibuat dengan accuracy: {model.score(X_scaled, y):.2%}")
    return model, scaler

# Fungsi suara (non-blocking + anti spam)
last_speak_time = 0
cooldown = 2  # detik

def speak(text):
    global last_speak_time

    if time.time() - last_speak_time < cooldown:
        return

    last_speak_time = time.time()

    def run():
        tts = gTTS(text=text, lang='id')
        filename = "voice.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)

    threading.Thread(target=run).start()

# ✅ Fungsi hitung jari (fallback jika model tidak ada)
def count_fingers_fallback(lm):
    fingers = []

    # Ibu jari
    if lm[4].x > lm[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Jari lainnya
    tips = [8, 12, 16, 20]
    for tip in tips:
        if lm[tip].y < lm[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

# ✅ Load/Train model
print("\n" + "="*50)
print("🚀 INISIALISASI SISTEM DETEKSI JARI")
print("="*50)

model, scaler = train_or_load_model()

# ✅ Validasi model
def validate_model(model, scaler):
    if model is None or scaler is None:
        print("Model tidak tersedia - Menggunakan fallback heuristic")
        return False
    else:
        print("Model ML berhasil dimuat")
        return True

model_available = validate_model(model, scaler)

# ✅ Fungsi preprocess frame untuk model
def preprocess_frame_for_model(hand_landmarks):
    """Preprocess hand landmarks untuk model"""
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features).reshape(1, -1)

# ✅ Fungsi run model prediction dengan confidence checking
def run_model_prediction(landmarks, model, scaler, confidence_threshold=0.5):
    """Jalankan prediksi dengan model ML + confidence threshold"""
    try:
        if model is None or scaler is None:
            raise ValueError("Model tidak tersedia")
        
        features = preprocess_frame_for_model(landmarks)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probas = model.predict_proba(features_scaled)[0]
        confidence = float(max(probas))
        
        # ✅ Cek threshold confidence
        if confidence < confidence_threshold:
            return None, confidence
        
        return int(prediction), confidence
    except Exception as e:
        print(f"❌ Error saat menjalankan model: {str(e)}")
        return None, None

# ✅ Fungsi smoothing dengan majority voting
def smooth_prediction(prediction, history, window_size=3):
    """Smooth prediksi menggunakan majority voting"""
    if prediction is not None:
        history.append(prediction)
    
    # Simpan hanya window yang dibutuhkan
    if len(history) > window_size:
        history.pop(0)
    
    # Jika ada cukup history, gunakan majority vote
    if len(history) >= window_size:
        from collections import Counter
        most_common = Counter(history).most_common(1)[0][0]
        return most_common, history
    
    # Return last valid prediction atau None
    return history[-1] if history else None, history

# ✅ Fungsi Pengolahan Citra - Thresholding & Binarization
def image_processing_threshold(frame):
    """
    Pengolahan citra dengan thresholding & binarization
    Returns: gray, binary_otsu, binary_adaptive
    """
    # Konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur untuk noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's Binarization (otomatis cari threshold optimal)
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Adaptive Thresholding (bagus untuk pencahayaan tidak merata)
    binary_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return gray, binary_otsu, binary_adaptive

# ✅ Fungsi Morphological Operations
def morphological_operations(binary_image):
    """
    Operasi morfologi untuk membersihkan binary image
    """
    # Kernel untuk operasi morfologi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Closing (dilasi + erosi) - menutup hole kecil
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    # Opening (erosi + dilasi) - menghilangkan noise kecil
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    return opened

# ✅ Fungsi Contour Detection
def detect_contours(binary_image):
    """
    Deteksi contour dari binary image
    """
    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours, hierarchy

# ✅ Fungsi Visualisasi Pengolahan Citra
def visualize_processing_steps(frame, gray, binary_otsu, binary_adaptive, morpho):
    """
    Tampilkan semua tahap pengolahan citra dalam satu window
    """
    # Extract region ukuran sama
    h, w = frame.shape[:2]
    size = (w//2, h//2)
    
    # Resize untuk ditampilkan bersamaan
    frame_resized = cv2.resize(frame, size)
    gray_bgr = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), size)
    otsu_bgr = cv2.resize(cv2.cvtColor(binary_otsu, cv2.COLOR_GRAY2BGR), size)
    adaptive_bgr = cv2.resize(cv2.cvtColor(binary_adaptive, cv2.COLOR_GRAY2BGR), size)
    morpho_bgr = cv2.resize(cv2.cvtColor(morpho, cv2.COLOR_GRAY2BGR), size)
    
    # Gabung vertikal - baris 1
    row1 = np.hstack([frame_resized, gray_bgr])
    
    # Gabung vertikal - baris 2
    row2 = np.hstack([otsu_bgr, adaptive_bgr])
    
    # Gabung horizontal
    combined = np.vstack([row1, row2])
    
    # Tambah label
    cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, "Grayscale", (w//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, "Otsu Binary", (10, h//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, "Adaptive Binary", (w//2 + 10, h//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return combined

# Webcam
print("\n📹 Membuka webcam...")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("❌ GAGAL membuka webcam!")
    exit()
else:
    print("✅ Webcam berhasil dibuka")
    print("\n📌 Kontrol: Tekan 'Q' untuk keluar\n")
    print("="*50 + "\n")

last_count = -1
frame_skip = 3
frame_count = 0
fingers = 0
confidence_score = 0.0  # ✅ Initialize confidence score
confidence_threshold = 0.5  # ✅ Threshold confidence 50%
prediction_history = []  # ✅ History untuk smoothing
smoothing_window = 3  # ✅ Gunakan 3 frame sebelumnya

# ✅ kontrol program
running = True

while running:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Resize biar ringan
    small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    # ✅ PENGOLAHAN CITRA - Thresholding & Binarization
    gray, binary_otsu, binary_adaptive = image_processing_threshold(frame)
    morpho = morphological_operations(binary_otsu)
    contours, hierarchy = detect_contours(morpho)

    # Deteksi tiap beberapa frame
    if frame_count % frame_skip == 0:
        new_fingers = None
        confidence_score = 0
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                if model_available:
                    # Gunakan model ML dengan confidence threshold
                    predicted_fingers, confidence = run_model_prediction(
                        hand_landmarks, model, scaler, confidence_threshold
                    )
                    if predicted_fingers is not None:
                        new_fingers = predicted_fingers
                        confidence_score = confidence
                    else:
                        # Fallback jika confidence terlalu rendah
                        new_fingers = count_fingers_fallback(hand_landmarks.landmark)
                        confidence_score = 0.0
                else:
                    # Gunakan fallback
                    new_fingers = count_fingers_fallback(hand_landmarks.landmark)
                    confidence_score = 0.0
        else:
            new_fingers = 0
            confidence_score = 0.0
        
        # ✅ Smoothing dengan majority voting
        fingers, prediction_history = smooth_prediction(new_fingers, prediction_history, smoothing_window)

    frame_count += 1

    # Gambar landmark
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Tampilkan jumlah jari + status model
    status_text = f"Jumlah Jari: {fingers} {'[ML]' if model_available else '[Heuristic]'}"
    cv2.putText(frame, status_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    # Tampilkan confidence score
    if model_available and confidence_score > 0:
        conf_text = f"Confidence: {confidence_score:.1%}"
        color = (0, 255, 0) if confidence_score >= confidence_threshold else (0, 0, 255)
        cv2.putText(frame, conf_text, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    # Tampilkan FPS + info frame rate
    frame_info = f"Frame: {frame_count} | Smoothing: {len(prediction_history)}/{smoothing_window}"
    cv2.putText(frame, frame_info, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

    # ✅ Suara jika berubah (dengan safeguard)
    if fingers != last_count:
        try:
            if fingers == 0:
                speak("Tidak ada jari")
            else:
                speak(f"{fingers} jari")
            
            # Log perubahan
            print(f"✓ Perubahan terdeteksi: {last_count} -> {fingers} jari")
            last_count = fingers
        except Exception as e:
            print(f"⚠️ Error saat berbicara: {str(e)}")
            last_count = fingers

    cv2.imshow("Deteksi Jumlah Jari", frame)

    # ✅ tekan Q untuk keluar
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Keluar dari program...")
        running = False

# ✅ cleanup (WAJIB)
cap.release()
cv2.destroyAllWindows()
os._exit(0)