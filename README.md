# Real-Time Facial Analysis Pipeline (YOLO11 + BoT-Sort)

Pipeline analisis wajah real-time berkinerja tinggi yang dirancang untuk melakukan deteksi wajah, pelacakan (tracking) persisten, estimasi demografi (Usia, Jenis Kelamin, Ras), dan analisis emosi wajah menggunakan **YOLO11**, **BoT-Sort**, dan **ONNX Runtime**.

## 🚀 Fitur Utama
* **Face Tracking Berbasis BoT-Sort:** Pelacakan wajah yang stabil dan persisten menggunakan kombinasi algoritma IoU matching dan Kalman Filter.
* **Analisis Demografi & Emosi Multitask:** Estimasi kelompok usia, jenis kelamin, ras (FairFace standard), serta 8 emosi wajah (FER-2013 standard) dalam satu pipeline.
* **Optimasi Frame Skipping:** Mengurangi beban komputasi dengan melakukan inferensi berat (demografi & emosi) setiap N-frame sekali untuk performa FPS yang optimal.
* **Sistem Polygon Zone Counting:** Mendukung pendefinisian area masuk/keluar (poligon) untuk menghitung wajah yang melewati zona deteksi tertentu secara dinamis.
* **HUD Monitor Kinerja:** Tampilan FPS keseluruhan, waktu inferensi masing-masing tahap (YOLO, FairFace, Emotion), serta jumlah wajah yang keluar zona secara real-time.

---

## 📁 Struktur Direktori Proyek
```text
├── model_Yolov11n_bounding_box_botsort_agentic.py  # Skrip utama pipeline real-time
├── requirements.txt                                # Daftar dependensi Python
├── .gitignore                                      # File pengecualian Git
└── README.md                                       # Dokumentasi proyek ini
```
> **Catatan:** File model berukuran besar (`*.pt`, `*.onnx`) sengaja diabaikan oleh `.gitignore` untuk menjaga ukuran repositori tetap ringan. Unduh file model tersebut secara terpisah (lihat bagian [Unduh Model](#-unduh-model)).

---

## 🛠️ Panduan Instalasi

### 1. Prasyarat
Pastikan sistem Anda sudah terinstal **Python 3.8** atau versi yang lebih baru.

### 2. Kloning Repositori
```bash
git clone <URL_REPOSITORI_ANDA>
cd deepface
```

### 3. Instal Dependensi
Instal pustaka-pustaka yang diperlukan menggunakan `pip`:
```bash
pip install -r requirements.txt
```

---

## 📦 Unduh Model (Weights)
Sebelum menjalankan skrip, pastikan Anda telah menempatkan file model berikut ke dalam direktori root proyek Anda:

1. **YOLO11 Face Detector:**
   * Nama file: `yolo11n-face.pt` (Sudah disertakan langsung di dalam repositori ini).
2. **FairFace Model (ONNX):**
   * Nama file: `fairface.onnx`
   * Digunakan untuk klasifikasi Usia, Gender, dan Ras.
   * 🔗 **[Unduh fairface.onnx dari Hugging Face (FaceFusion)](https://huggingface.co/facefusion/models-3.0.0/resolve/main/fairface.onnx)**
3. **Emotion Model (ONNX):**
   * Nama file: `emotion.onnx`
   * Digunakan untuk deteksi ekspresi/emosi wajah (Mini-Xception).
   * 🔗 **[Unduh emotion.onnx dari GitHub (neal-zhan)](https://github.com/neal-zhan/face-recognition/raw/master/emotion.onnx)**

---

## 💻 Cara Menjalankan

Jalankan skrip utama untuk memulai analisis wajah real-time (secara default menggunakan feed CCTV Malioboro):

```bash
python model_Yolov11n_bounding_box_botsort_agentic.py
```

### Kontrol Aplikasi:
* Tekan tombol **`q`** pada keyboard saat jendela video aktif untuk keluar dari aplikasi secara aman (graceful exit).

---

## ⚙️ Konfigurasi Utama
Anda dapat menyesuaikan parameter utama langsung di dalam instansiasi kelas `RealTimeFacialAnalyzer` pada blok `__main__`:

```python
analyzer = RealTimeFacialAnalyzer(
    yolo_model_path="yolo11n-face.pt",
    fairface_model_path="fairface.onnx",
    emotion_model_path="emotion.onnx",
    conf_threshold=0.1,  # Batas minimum kepercayaan deteksi wajah
    iou_threshold=0.45   # Batas NMS IoU untuk YOLO
)
```

Untuk mengubah sumber video (misalnya menggunakan webcam bawaan laptop Anda), ubah argumen `camera_source` pada method `start_webcam`:
```python
# Gunakan camera_source=0 atau 1 untuk webcam lokal
analyzer.start_webcam(camera_source=0, frame_interval=3)
```

---

## 📝 Lisensi
Proyek ini dilisensikan di bawah lisensi MIT. Bebas digunakan untuk tujuan akademik maupun komersial.
