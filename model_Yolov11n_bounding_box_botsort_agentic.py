import os
import sys
import time
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
import onnxruntime as ort

# Coba mengimpor Ultralytics YOLO untuk deteksi wajah
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


class RealTimeFacialAnalyzer:
    """
    Pipa Analisis Wajah Real-Time Siap Produksi menggunakan ONNX Runtime.
    
    Atribut:
        AGE_CLASSES (List[str]): Pemetaan label kelompok usia (standar FairFace).
        GENDER_CLASSES (List[str]): Pemetaan label jenis kelamin (standar FairFace).
        RACE_CLASSES (List[str]): Pemetaan label ras/etnis (standar FairFace).
        EMOTION_CLASSES (List[str]): Pemetaan 7 label emosi wajah (standar FER-2013).
    """

    AGE_CLASSES: List[str] = [
        '0-2', '3-9', '10-19', '20-29', '30-39', 
        '40-49', '50-59', '60-69', '70+'
    ]
    
    GENDER_CLASSES: List[str] = ['Male', 'Female']
    
    RACE_CLASSES: List[str] = [
        'White', 'Black', 'Latino_Hispanic', 'East Asian', 
        'Southeast Asian', 'Indian', 'Middle Eastern'
    ]
    
    EMOTION_CLASSES: List[str] = [
        'Neutral', 'Happiness', 'Surprise', 'Sadness',
        'Anger', 'Disgust', 'Fear', 'Contempt'
    ]

    def __init__(
        self,
        yolo_model_path: str = "yolo11n-face.pt",
        fairface_model_path: str = "fairface.onnx",
        emotion_model_path: str = "emotion.onnx",
        conf_threshold: float = 0.2,
        iou_threshold: float = 0.45,
    ) -> None:
        """
        Inisialisasi deteksi wajah dan sesi inferensi ONNX sekali di konstruktor
        untuk menghindari kebocoran memori dan overhead selama streaming real-time.

        Argumen:
            yolo_model_path (str): Jalur ke detektor wajah YOLO11 (.pt atau .onnx).
            fairface_model_path (str): Jalur ke model ONNX FairFace.
            emotion_model_path (str): Jalur ke model ONNX Emosi FER-2013.
            conf_threshold (float): Ambang batas kepercayaan minimum untuk deteksi wajah.
            iou_threshold (float): Ambang batas Intersection Over Union NMS untuk deteksi wajah.
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # 1. Inisialisasi Detektor Wajah (YOLO11)
        self.yolo_model = self._init_yolo_detector(yolo_model_path)

        # 2. Inisialisasi Sesi ONNX Demografi (FairFace)
        self.fairface_session, self.fairface_input_name = self._init_onnx_session(
            fairface_model_path, "Demographics (FairFace)"
        )

        # 3. Inisialisasi Sesi ONNX Emosi (Mini-Xception)
        self.emotion_session, self.emotion_input_name = self._init_onnx_session(
            emotion_model_path, "Emotion (Mini-Xception)"
        )

        print("[INFO] RealTimeFacialAnalyzer initialized successfully.")

    def _init_yolo_detector(self, model_path: str) -> Optional[Any]:
        """Memuat detektor wajah YOLO11 menggunakan ultralytics atau cadangan (fallback)."""
        abs_path = os.path.join(self.script_dir, model_path) if not os.path.isabs(model_path) else model_path
        
        target_path = abs_path if os.path.exists(abs_path) else model_path
        
        if HAS_ULTRALYTICS: 
            print(f"[INFO] Loading YOLO11 Face Detector from: {target_path}")
            try:
                model = YOLO(target_path)
                return model
            except Exception as e:
                print(f"[WARN] Failed to load YOLO model via Ultralytics: {e}")
                return None
        else:
            print("[WARN] 'ultralytics' package not installed. YOLO inference limited.")
            return None

    def _init_onnx_session(
        self, model_path: str, model_name: str
    ) -> Tuple[Optional[ort.InferenceSession], Optional[str]]:
        """Menginisialisasi InferenceSession ONNX Runtime dengan penyedia eksekusi CPU (CPU execution provider)."""
        abs_path = os.path.join(self.script_dir, model_path) if not os.path.isabs(model_path) else model_path

        if not os.path.exists(abs_path):
            print(f"[WARNING] ONNX model for {model_name} not found at '{abs_path}'.")
            return None, None

        try:
            session = ort.InferenceSession(
                abs_path,
                providers=['CPUExecutionProvider']
            )
            input_name = session.get_inputs()[0].name
            print(f"[INFO] Loaded ONNX session for {model_name}: {abs_path}")
            return session, input_name
        except Exception as e:
            print(f"[ERROR] Failed to load ONNX model {model_name}: {e}")
            return None, None

    def preprocess_demographics(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Pra-proses potongan wajah BGR untuk model ONNX FairFace (224x224, RGB, format NCHW).

        Argumen:
            face_crop (np.ndarray): Potongan gambar BGR dari wajah yang dipotong.

        Pengembalian:
            np.ndarray: Tensor float32 hasil pra-proses dengan bentuk [1, 3, 224, 224].
        """
        # BGR -> RGB
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        # Ubah ukuran ke 224x224
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        # Normalisasi ke [0, 1]
        norm = resized.astype(np.float32) / 255.0
        # Standarisasi (mean/std ImageNet)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        standardized = (norm - mean) / std
        # HWC -> CHW -> NCHW [1, 3, 224, 224]
        tensor = np.transpose(standardized, (2, 0, 1))
        input_tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
        return input_tensor

    def preprocess_emotion(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Pra-proses potongan wajah BGR untuk model ONNX Emosi FERPlus (64x64, Grayscale, format NCHW).

        Argumen:
            face_crop (np.ndarray): Potongan gambar BGR dari wajah yang dipotong.

        Pengembalian:
            np.ndarray: Tensor float32 hasil pra-proses dengan bentuk [1, 1, 64, 64].
        """
        # BGR -> Grayscale
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        # Ubah ukuran ke 64x64
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        # Model FERPlus mengharapkan nilai piksel mentah dalam rentang [0, 255], jangan dinormalisasi ke [0, 1]
        img_float = resized.astype(np.float32)
        # Format [1, 1, 64, 64] (NCHW)
        input_tensor = img_float[np.newaxis, np.newaxis, :, :]
        return input_tensor

    def predict_demographics(
        self, face_crop: np.ndarray
    ) -> Tuple[str, str, str, float, float, float]:
        """
        Menjalankan inferensi ONNX FairFace untuk memprediksi Usia, Jenis Kelamin, dan Ras
        beserta confidence score masing-masing (dari softmax probability).

        Argumen:
            face_crop (np.ndarray): Gambar wajah BGR yang dipotong.

        Pengembalian:
            Tuple[str, str, str, float, float, float]:
                (age_label, gender_label, race_label,
                 age_conf, gender_conf, race_conf)
        """
        if self.fairface_session is None:
            return ("Unknown", "Unknown", "Unknown", 0.0, 0.0, 0.0)

        def _softmax(x: np.ndarray) -> np.ndarray:
            e = np.exp(x - np.max(x))
            return e / e.sum()

        try:
            input_tensor = self.preprocess_demographics(face_crop)
            outputs = self.fairface_session.run(None, {self.fairface_input_name: input_tensor})

            if len(outputs) >= 3:
                race_out, gender_out, age_out = outputs[0], outputs[1], outputs[2]

                if race_out.ndim == 1:
                    # Output langsung berupa class index — tidak ada logits untuk softmax
                    race_idx   = int(race_out[0])
                    gender_idx = int(gender_out[0])
                    age_idx    = int(age_out[0])
                    race_conf = gender_conf = age_conf = 1.0   # tidak tersedia
                else:
                    race_probs   = _softmax(race_out[0])
                    gender_probs = _softmax(gender_out[0])
                    age_probs    = _softmax(age_out[0])
                    race_idx   = int(np.argmax(race_probs))
                    gender_idx = int(np.argmax(gender_probs))
                    age_idx    = int(np.argmax(age_probs))
                    race_conf   = float(race_probs[race_idx])
                    gender_conf = float(gender_probs[gender_idx])
                    age_conf    = float(age_probs[age_idx])

            elif len(outputs) == 1:
                logits     = outputs[0][0]
                race_probs   = _softmax(logits[:7])
                gender_probs = _softmax(logits[7:9])
                age_probs    = _softmax(logits[9:18])
                race_idx   = int(np.argmax(race_probs))
                gender_idx = int(np.argmax(gender_probs))
                age_idx    = int(np.argmax(age_probs))
                race_conf   = float(race_probs[race_idx])
                gender_conf = float(gender_probs[gender_idx])
                age_conf    = float(age_probs[age_idx])
            else:
                return ("Unknown", "Unknown", "Unknown", 0.0, 0.0, 0.0)

            race   = self.RACE_CLASSES[race_idx   % len(self.RACE_CLASSES)]
            gender = self.GENDER_CLASSES[gender_idx % len(self.GENDER_CLASSES)]
            age    = self.AGE_CLASSES[age_idx    % len(self.AGE_CLASSES)]

            return age, gender, race, age_conf, gender_conf, race_conf
        except Exception as e:
            print(f"[ERR] Demographics inference error: {e}")
            return ("Unknown", "Unknown", "Unknown", 0.0, 0.0, 0.0)

    def predict_emotion(self, face_crop: np.ndarray) -> Tuple[str, float]:
        """
        Menjalankan inferensi ONNX Mini-Xception untuk memprediksi emosi wajah
        beserta confidence score (dari softmax probability).

        Argumen:
            face_crop (np.ndarray): Gambar wajah BGR yang dipotong.

        Pengembalian:
            Tuple[str, float]: (emotion_label, confidence)
        """
        if self.emotion_session is None:
            return "Unknown", 0.0

        try:
            input_tensor = self.preprocess_emotion(face_crop)
            outputs = self.emotion_session.run(None, {self.emotion_input_name: input_tensor})

            emotion_logits = outputs[0][0]
            # Softmax untuk mendapatkan probability
            e = np.exp(emotion_logits - np.max(emotion_logits))
            emotion_probs = e / e.sum()

            emotion_idx  = int(np.argmax(emotion_probs))
            emotion_conf = float(emotion_probs[emotion_idx])

            return self.EMOTION_CLASSES[emotion_idx % len(self.EMOTION_CLASSES)], emotion_conf
        except Exception as e:
            print(f"[ERR] Emotion inference error: {e}")
            return "Unknown", 0.0

    def track_faces(
        self, frame: np.ndarray, offset: Tuple[int, int] = (0, 0)
    ) -> List[Tuple[int, int, int, int, float, int]]:
        """
        Mendeteksi DAN melacak wajah menggunakan YOLO11 + BoT-Sort.

        Argumen:
            frame (np.ndarray): Frame webcam penuh dalam format BGR.
            offset (Tuple[int, int]): Offset (x, y) untuk menggeser koordinat kembali ke frame global.

        Pengembalian:
            List[Tuple[int, int, int, int, float, int]]:
                Daftar (x1, y1, x2, y2, confidence, track_id).
                track_id adalah ID unik persisten yang diberikan oleh BoT-Sort.
        """
        tracked = []
        if self.yolo_model is None:
            return tracked

        try:
            results = self.yolo_model.track(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                tracker="botsort.yaml",
                persist=True,               
                verbose=False,

            )
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    if box.id is None:
                        continue   # Lewati deteksi tanpa track ID (belum confirmed)
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    conf  = float(box.conf[0].cpu().numpy())
                    tid   = int(box.id[0].cpu().numpy())
                    x1, y1, x2, y2 = coords[:4]
                    x1 += offset[0]
                    y1 += offset[1]
                    x2 += offset[0]
                    y2 += offset[1]
                    tracked.append((x1, y1, x2, y2, conf, tid))
        except Exception as e:
            print(f"[ERR] BoT-Sort tracking error: {e}")

        return tracked

    def start_webcam(self, camera_source: Any = 1, frame_interval: int = 3) -> None:
        """
        Menjalankan loop pemrosesan webcam real-time dengan optimasi lompatan frame (frame skipping).

        Argumen:
            camera_source (Any): Indeks perangkat kamera OpenCV (int) atau URL aliran/stream (str).
            frame_interval (int): Memproses inferensi mendalam setiap frame ke-N untuk meningkatkan FPS.
        """
        print(f"[INFO] Opening video source: {camera_source}...")
        if isinstance(camera_source, int):
            cap = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_source)
        else:
            cap = cv2.VideoCapture(camera_source)

        if not cap.isOpened():
            print(f"[ERROR] Could not access webcam at index {camera_source}.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        window_name = """
Pipa Analisis Wajah Real-Time (Webcam) — Edisi BoT-Sort
-------------------------------------------------------
Pipa berkinerja tinggi dan ringan untuk deteksi wajah, estimasi demografi
(Usia, Jenis Kelamin, Ras), dan analisis emosi menggunakan YOLO11 + BoT-Sort + ONNX Runtime.

Pelacakan: BoT-Sort (bawaan Ultralytics) — berbasis IoU + filter Kalman
Detektor Wajah: YOLO11n-Face (zjykzj/YOLO11Face)
Penulis: Senior Computer Vision Engineer
"""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("[INFO] Webcam feed started. Press 'q' to exit gracefully.")

        frame_count = 0
        cached_results: List[Dict[str, Any]] = []

        prev_time = time.perf_counter()
        overall_fps = 0.0

        # Metrik Performa Setiap Tahapan
        yolo_ms = 0.0
        fairface_ms = 0.0
        emotion_ms = 0.0
        yolo_fps = 0.0
        fairface_fps = 0.0
        emotion_fps = 0.0

        # --- BoT-Sort Zone Counter State ---
        # Hanya menyimpan zona terakhir per track_id (BoT-Sort yang urus ID matching)
        zone_state: Dict[int, str] = {}     # {track_id: 'inside' | 'outside'}
        zone_local_ids: Dict[int, int] = {} # {track_id: local_id} untuk wajah di dalam zona
        exited_count: int = 0
        on_box_now: int = 0

        # --- Polygon Zone Definitions (persentase dari resolusi frame) ---
        _COUNT_ZONE_PCT = np.array([
            [0.700, 0.400],   
            [0.950, 0.400],   
            [0.950, 0.650],   
            [0.500, 0.650],   
        ], dtype=np.float32)

        _DETECT_ZONE_PCT = np.array([
            [0.775, 0.200],   
            [0.978, 0.200],   
            [0.978, 0.780],   
            [0.283, 0.780],   
        ], dtype=np.float32)

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[ERROR] Failed to capture frame from camera.")
                    break

                frame_count += 1
                h_img, w_img = frame.shape[:2]

                COUNT_ZONE_PTS  = (_COUNT_ZONE_PCT  * [w_img, h_img]).astype(np.int32)
                DETECT_ZONE_PTS = (_DETECT_ZONE_PCT * [w_img, h_img]).astype(np.int32)
                _db = cv2.boundingRect(DETECT_ZONE_PTS)
                if frame_count % frame_interval == 0 or len(cached_results) == 0:
                    current_results = []

                    # Area Deteksi — bounding rect dari poligon DETECT_ZONE_PTS
                    roi_x1 = max(0, _db[0])
                    roi_y1 = max(0, _db[1])
                    roi_x2 = min(w_img, _db[0] + _db[2])
                    roi_y2 = min(h_img, _db[1] + _db[3])
                    roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]

                    # Tahap 1: Deteksi Wajah + BoT-Sort
                    t0_yolo = time.perf_counter()
                    tracked_boxes = self.track_faces(roi_frame, offset=(roi_x1, roi_y1))
                    t1_yolo = time.perf_counter()

                    curr_yolo_ms = max(0.1, (t1_yolo - t0_yolo) * 1000.0)
                    yolo_ms = 0.85 * yolo_ms + 0.15 * curr_yolo_ms if yolo_ms > 0 else curr_yolo_ms
                    yolo_fps = 1000.0 / yolo_ms

                    curr_fairface_ms = 0.0
                    curr_emotion_ms  = 0.0
                    faces_processed  = 0

                    current_results = []
                    for (x1, y1, x2, y2, conf, tid) in tracked_boxes:
                        # Penanganan Kasus Batas: Pemotongan crop di luar batas gambar
                        x1_c = max(0, min(x1, w_img - 1))
                        y1_c = max(0, min(y1, h_img - 1))
                        x2_c = max(x1_c + 1, min(x2, w_img))
                        y2_c = max(y1_c + 1, min(y2, h_img))

                        if (x2_c - x1_c) <= 5 or (y2_c - y1_c) <= 5:
                            continue

                        # Filter ketat: buang wajah yang centroid-nya di luar DETECT_ZONE polygon
                        cx_check = float((x1_c + x2_c) // 2)
                        cy_check = float((y1_c + y2_c) // 2)
                        if cv2.pointPolygonTest(DETECT_ZONE_PTS, (cx_check, cy_check), False) < 0:
                            continue

                        face_crop = frame[y1_c:y2_c, x1_c:x2_c]
                        faces_processed += 1

                        # Tahap 2: Demografi (FairFace) + tingkat kepercayaan
                        t0_ff = time.perf_counter()
                        age, gender, race, age_conf, gender_conf, race_conf = self.predict_demographics(face_crop)
                        t1_ff = time.perf_counter()
                        curr_fairface_ms += max(0.001, (t1_ff - t0_ff) * 1000.0)

                        # Tahap 3: Emosi (Mini-Xception) + tingkat kepercayaan
                        t0_em = time.perf_counter()
                        emotion, emotion_conf = self.predict_emotion(face_crop)
                        t1_em = time.perf_counter()
                        curr_emotion_ms += max(0.001, (t1_em - t0_em) * 1000.0)

                        current_results.append({
                            'bbox': (x1_c, y1_c, x2_c, y2_c),
                            'conf': conf,          
                            'track_id': tid,
                            'age': age,             'age_conf': age_conf,
                            'gender': gender,       'gender_conf': gender_conf,
                            'race': race,           'race_conf': race_conf,
                            'emotion': emotion,     'emotion_conf': emotion_conf,
                        })

                    if faces_processed > 0:
                        fairface_ms = 0.85 * fairface_ms + 0.15 * curr_fairface_ms if fairface_ms > 0 else curr_fairface_ms
                        fairface_fps = 1000.0 / max(0.1, fairface_ms)
                        emotion_ms = 0.85 * emotion_ms + 0.15 * curr_emotion_ms if emotion_ms > 0 else curr_emotion_ms
                        emotion_fps = 1000.0 / max(0.1, emotion_ms)

                    cached_results = current_results

                    # --- Deteksi Keluar Zona BoT-Sort (Poligon) ---
                    active_ids: set = set()
                    on_box_now = 0

                    for res in cached_results:
                        (bx1, by1, bx2, by2) = res['bbox']
                        tid = res['track_id']
                        cx  = (bx1 + bx2) // 2
                        cy  = (by1 + by2) // 2

                        in_zone = cv2.pointPolygonTest(COUNT_ZONE_PTS, (float(cx), float(cy)), False) >= 0
                        new_zone = 'inside' if in_zone else 'outside'

                        if in_zone:
                            on_box_now += 1

                        # Deteksi exit: track_id yang sebelumnya 'inside' sekarang 'outside'
                        if tid in zone_state:
                            if zone_state[tid] == 'inside' and new_zone == 'outside':
                                exited_count += 1
                                print(f"[BoT-Sort] EXIT! track_id={tid}  Total={exited_count}")

                        zone_state[tid] = new_zone
                        active_ids.add(tid)

                    # Hapus track_id yang sudah tidak aktif dari zone_state
                    stale = [tid for tid in zone_state if tid not in active_ids]
                    for tid in stale:
                        del zone_state[tid]

                # --- Render Poligon Zona Deteksi ---
                total_exited = exited_count

                # Warna box: kuning-oranye normal, merah saat ada wajah di dalam
                box_color = (0, 200, 255) if on_box_now == 0 else (0, 80, 255)

                # Isian semi-transparan untuk zona penghitungan (count zone)
                zone_overlay = frame.copy()
                cv2.fillPoly(zone_overlay, [COUNT_ZONE_PTS], box_color)
                cv2.addWeighted(zone_overlay, 0.08, frame, 0.92, 0, frame)

                # Batas zona penghitungan (count zone border)
                cv2.polylines(frame, [COUNT_ZONE_PTS], True, box_color, 1,lineType=cv2.LINE_AA)

                # Batas area deteksi (detection area border)
                dl_x = DETECT_ZONE_PTS[0][0] - 10
                dl_y = DETECT_ZONE_PTS[0][1] - 10

                cv2.polylines(frame, [DETECT_ZONE_PTS], True, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                cv2.rectangle(frame, (dl_x+9,dl_y-7),(dl_x+135,dl_y+9),(255,255,255), -1)
                cv2.putText(frame, "Detection Area", (dl_x+10, dl_y+7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

                # Label count di sudut kiri atas box
                count_label = f"Counting Area"
                cl_font = cv2.FONT_HERSHEY_SIMPLEX
                cl_scale = 0.5
                cl_thick = 1
                (cl_w, cl_h), _ = cv2.getTextSize(count_label, cl_font, cl_scale, cl_thick)
                cl_x = COUNT_ZONE_PTS[0][0]          # titik kiri atas polygon count zone
                cl_y = max(cl_h + 8, COUNT_ZONE_PTS[0][1] - 6)
                # Latar belakang berbentuk pill (pill background)
                cv2.rectangle(frame, (cl_x - 1, cl_y - cl_h - 6), (cl_x + cl_w + 8, cl_y + 5),
                              (box_color), -1)
                cv2.rectangle(frame, (cl_x - 1, cl_y - cl_h - 6), (cl_x + cl_w + 8, cl_y + 4),
                              box_color, 1)
                cv2.putText(frame, count_label, (cl_x, cl_y),
                            cl_font, cl_scale, (255, 255, 255), cl_thick, cv2.LINE_AA)

                # --- Buat mapping local ID untuk wajah yang ADA di dalam zona ---
                # Stable: track_id yang sudah punya local_id tetap sama, yang baru dapat nomor berikutnya
                in_zone_tids = [
                    res['track_id'] for res in cached_results
                    if cv2.pointPolygonTest(
                        COUNT_ZONE_PTS,
                        (float((res['bbox'][0] + res['bbox'][2]) // 2),
                         float((res['bbox'][1] + res['bbox'][3]) // 2)),
                        False
                    ) >= 0
                ]
                # Hapus track_id yang sudah tidak di zona dari zone_local_ids
                for old_tid in list(zone_local_ids.keys()):
                    if old_tid not in in_zone_tids:
                        del zone_local_ids[old_tid]
                # Assign local ID baru untuk track_id yang belum terdaftar
                for ztid in in_zone_tids:
                    if ztid not in zone_local_ids:
                        next_local = max(zone_local_ids.values(), default=0) + 1
                        zone_local_ids[ztid] = next_local

                # Render Bounding Box & Hamparan Teks (Text Overlay) untuk SEMUA frame
                for res in cached_results:
                    (bx1, by1, bx2, by2) = res['bbox']
                    det_conf    = res['conf']
                    tid         = res['track_id']
                    age         = res['age']
                    gender      = res['gender']
                    race        = res['race']
                    emotion     = res['emotion']
                    age_conf    = res.get('age_conf', 0.0)
                    gender_conf = res.get('gender_conf', 0.0)
                    race_conf   = res.get('race_conf', 0.0)
                    emotion_conf = res.get('emotion_conf', 0.0)

                    # Bounding box color: merah jika centroid di dalam zone box, hijau jika di luar
                    cx_face = (bx1 + bx2) // 2
                    cy_face = (by1 + by2) // 2
                    is_in_zone = cv2.pointPolygonTest(COUNT_ZONE_PTS, (float(cx_face), float(cy_face)), False) >= 0
                    bbox_color = (0, 80, 255) if is_in_zone else (0, 255, 127)

                    # 1. Bounding box
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), bbox_color, 2)

                    # 2. Local ID di sudut kiri atas bbox — hanya untuk wajah di dalam zona
                    if is_in_zone and tid in zone_local_ids:
                        local_id = zone_local_ids[tid]
                        tid_label = f"#{local_id}"
                        cv2.putText(frame, tid_label, (bx1 + 4, by1 + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                    # 3. Text overlay lines — conf dirata-rata dari 3 model
                    overall_conf = (emotion_conf)

                    text_line1 = f"{gender} | {age}"
                    text_line2 = f"{race} | {emotion}"
                    text_line3 = f"Conf: {overall_conf:.0%}"

                    font       = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.42
                    thickness  = 1

                    (w1, h1), _ = cv2.getTextSize(text_line1, font, font_scale, thickness)
                    (w2, h2), _ = cv2.getTextSize(text_line2, font, font_scale, thickness)
                    (w3, h3), _ = cv2.getTextSize(text_line3, font, font_scale, thickness)
                    box_w = max(w1, w2, w3) + 12
                    box_h = h1 + h2 + h3 + 22

                    text_bg_y1 = max(0, by1 - box_h - 4)
                    text_bg_y2 = max(box_h + 4, by1)
                    text_bg_x1 = max(0, bx1)
                    text_bg_x2 = min(w_img, bx1 + box_w)

                    # Panel latar belakang semi-transparan
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

                    # Baris 1: Gender & Age + confidence
                    cv2.putText(frame, text_line1,
                                (text_bg_x1 + 5, text_bg_y1 + h1 + 4),
                                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    # Baris 2: Race + confidence
                    cv2.putText(frame, text_line2,
                                (text_bg_x1 + 5, text_bg_y1 + h1 + h2 + 10),
                                font, font_scale, (100, 220, 255), thickness, cv2.LINE_AA)
                    # Baris 3: Emotion + confidence + YOLO det conf
                    cv2.putText(frame, text_line3,
                                (text_bg_x1 + 5, text_bg_y1 + h1 + h2 + h3 + 18),
                                font, font_scale, (80, 255, 160), thickness, cv2.LINE_AA)

                # Hitung FPS Video Keseluruhan
                curr_time = time.perf_counter()
                time_diff = max(0.001, curr_time - prev_time)
                prev_time = curr_time
                overall_fps = 0.9 * overall_fps + 0.1 * (1.0 / time_diff)

                # Gambar Panel HUD Pemantauan Kinerja (Sudut Kiri Atas)
                hud_overlay = frame.copy()
                cv2.rectangle(hud_overlay, (10, 10), (220, 105), (10, 10, 10), -1)
                cv2.addWeighted(hud_overlay, 0.7, frame, 0.3, 0, frame)
                cv2.rectangle(frame, (10, 10), (220, 105), (200, 200, 200), 1)

                hud_font = cv2.FONT_HERSHEY_SIMPLEX
                hud_scale = 0.3
                hud_thick = 1

                lines = [
                    (f"Overall Video : {overall_fps:.1f} FPS", (0, 255, 0)),
                    (f"Botsort+YOLO: {yolo_fps:.1f} FPS ({yolo_ms:.1f} ms)", (255, 255, 0)),
                    (f"FairFace      : {fairface_fps:.1f} FPS ({fairface_ms:.1f} ms)", (255, 180, 50)),
                    (f"Emotion       : {emotion_fps:.1f} FPS ({emotion_ms:.1f} ms)", (0, 255, 255)),
                    (f"Keluar Zona   : {exited_count} face(s)", (0, 200, 255)),
                ]

                for idx, (line_str, color) in enumerate(lines):
                    y_pos = 25 + idx * 18
                    cv2.putText(frame, line_str, (20, y_pos),
                                hud_font, hud_scale, color, hud_thick, cv2.LINE_AA)

                # Tampilkan aliran video langsung (live feed)
                cv2.imshow(window_name, frame)

                # Pemeriksaan tombol keluar yang aman ('q')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] 'q' pressed. Shutting down pipeline...")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Webcam released and windows closed cleanly.")


if __name__ == "__main__":
    # Titik masuk skrip untuk pengujian produksi
    analyzer = RealTimeFacialAnalyzer(
        yolo_model_path="yolo11n-face.pt",
        fairface_model_path="fairface.onnx",
        emotion_model_path="emotion.onnx",
        conf_threshold=0.1,
        iou_threshold=0.45
    )

    # Atur sumber kamera ke URL CCTV Malioboro
    cctv_url = "https://cctvjss.jogjakota.go.id/malioboro/Malioboro_25_Utara_Mall.stream/chunklist_w1892315291.m3u8"
    analyzer.start_webcam(camera_source=cctv_url, frame_interval=3)
