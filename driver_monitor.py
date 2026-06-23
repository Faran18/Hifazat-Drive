"""
driver_monitor.py
Driver monitoring using fine-tuned YOLO ONNX model + MediaPipe face landmarks.
Detects: drowsiness (EAR), yawning (MAR), phone use, eating, drinking, distracted gaze (head pose).
"""

import cv2
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math
from collections import deque

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[DriverMonitor] MediaPipe not available.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[DriverMonitor] YOLO not available.")

try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[DriverMonitor] onnxruntime not available.")

# ─ Custom Model Paths ──────────────────────────────────────────────────────────
CUSTOM_YOLO_DMS_PATH = r"yolo_dms.onnx"  # Fine-tuned driver behavior detector
CUSTOM_YOLO_CLASSES_PATH = r"yolo_classes.json"  # Classes: cigarette, drinking, eating, phone, seatbelt, face
FACE_LANDMARKER_TASK_PATH = r"face_landmarker.task"  # MediaPipe face landmarker (auto-download if missing)

# ──────────────────────────────────────────────────────────────────────────────
# THRESHOLDS & CONFIG
# ──────────────────────────────────────────────────────────────────────────────
BUFFER_WINDOW = 60
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.35
YAW_THRESHOLD = 25
PITCH_THRESHOLD = 35
EMA_ALPHA = 0.30

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 308, 13, 14]

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, 330.0, -65.0),         # Chin
    (-225.0, -170.0, -135.0),    # Left eye
    (225.0, -170.0, -135.0),     # Right eye
    (-150.0, 150.0, -125.0),     # Left mouth
    (150.0, 150.0, -125.0),      # Right mouth
], dtype=np.float32)

# ── Driver states ────────────────────────────────────────────────────────────
DRIVER_STATES = {
    "NORMAL":           {"color": (0, 200, 50),   "label": "✓ NORMAL",           "priority": 0},
    "DROWSY":           {"color": (0, 165, 255),  "label": "⚠ DROWSY",           "priority": 3},
    "TEXTING":          {"color": (0, 0, 255),    "label": "🚨 TEXTING & DRIVE", "priority": 5},
    "PHONE":            {"color": (0, 165, 255),  "label": "📵 PHONE DISTRACT",  "priority": 4},
    "INGESTION":        {"color": (0, 165, 255),  "label": "🍽 EATING/DRINKING",  "priority": 2},
    "GAZE_DISTRACTED":  {"color": (0, 255, 255), "label": "👀 GAZE DISTRACTED",  "priority": 2},
    "YAWNING":          {"color": (255, 255, 0), "label": "😴 FATIGUE YAWN",      "priority": 1},
}

# EAR thresholds
EAR_DROWSY_THRESHOLD  = 0.22
EAR_SLEEP_THRESHOLD   = 0.18
EAR_CONSEC_FRAMES_DROWSY = 15
EAR_CONSEC_FRAMES_SLEEP  = 30

# Head pose thresholds (degrees)
HEAD_YAW_THRESHOLD   = 30
HEAD_PITCH_THRESHOLD = 20

# Landmark indices for eyes (MediaPipe 468-point mesh)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Nose tip + chin for head pose
NOSE_TIP  = 1
CHIN      = 199
LEFT_EAR_IDX  = 234
RIGHT_EAR_IDX = 454


@dataclass
class DriverState:
    state: str = "NORMAL"
    ear: float = 0.0
    mar: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    phone_detected: bool = False
    ingestion_detected: bool = False
    alert_message: Optional[str] = None


# ─ Global EMA state for head pose smoothing ─────────────────────────────────
_ema_pitch = None
_ema_yaw = None


def _ensure_face_landmarker_downloaded():
    """Auto-download face_landmarker.task if missing."""
    import os
    if os.path.exists(FACE_LANDMARKER_TASK_PATH):
        return FACE_LANDMARKER_TASK_PATH
    print(f"[DriverMonitor] Downloading {FACE_LANDMARKER_TASK_PATH}...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    try:
        urllib.request.urlretrieve(url, FACE_LANDMARKER_TASK_PATH)
        print(f"[DriverMonitor] Downloaded to {FACE_LANDMARKER_TASK_PATH}")
        return FACE_LANDMARKER_TASK_PATH
    except Exception as e:
        print(f"[DriverMonitor] Failed to download: {e}")
        return None


def _calculate_ear(landmarks, eye_indices, img_w, img_h) -> float:
    """Eye Aspect Ratio from MediaPipe landmarks."""
    pts = [np.array([landmarks[i].x * img_w, landmarks[i].y * img_h]) for i in eye_indices]
    p2_p6 = np.linalg.norm(pts[1] - pts[5])
    p3_p5 = np.linalg.norm(pts[2] - pts[4])
    p1_p4 = np.linalg.norm(pts[0] - pts[3])
    return (p2_p6 + p3_p5) / (2.0 * p1_p4) if p1_p4 > 0 else 0.0


def _calculate_mar(landmarks, mouth_indices, img_w, img_h) -> float:
    """Mouth Aspect Ratio from MediaPipe landmarks."""
    p_left = np.array([landmarks[mouth_indices[0]].x * img_w, landmarks[mouth_indices[0]].y * img_h])
    p_right = np.array([landmarks[mouth_indices[1]].x * img_w, landmarks[mouth_indices[1]].y * img_h])
    p_top = np.array([landmarks[mouth_indices[2]].x * img_w, landmarks[mouth_indices[2]].y * img_h])
    p_bottom = np.array([landmarks[mouth_indices[3]].x * img_w, landmarks[mouth_indices[3]].y * img_h])
    horizontal = np.linalg.norm(p_left - p_right)
    vertical = np.linalg.norm(p_top - p_bottom)
    return vertical / horizontal if horizontal > 0 else 0.0


def _estimate_head_pose(landmarks, img_w, img_h) -> Tuple[float, float]:
    """Estimate head pose (pitch, yaw) using PnP with EMA smoothing."""
    global _ema_pitch, _ema_yaw
    
    image_points = np.array([
        (landmarks[1].x * img_w, landmarks[1].y * img_h),        # Nose
        (landmarks[152].x * img_w, landmarks[152].y * img_h),    # Chin
        (landmarks[33].x * img_w, landmarks[33].y * img_h),      # Left eye
        (landmarks[263].x * img_w, landmarks[263].y * img_h),    # Right eye
        (landmarks[61].x * img_w, landmarks[61].y * img_h),      # Left mouth
        (landmarks[291].x * img_w, landmarks[291].y * img_h),    # Right mouth
    ], dtype=np.float32)

    focal_length = float(img_w)
    cam_matrix = np.array([[focal_length, 0, img_w / 2.0],
                           [0, focal_length, img_h / 2.0],
                           [0, 0, 1.0]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    success, rot_vec, trans_vec = cv2.solvePnP(MODEL_POINTS, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
    if not success:
        return (_ema_pitch if _ema_pitch is not None else 0.0, _ema_yaw if _ema_yaw is not None else 0.0)

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    proj_matrix = np.hstack((rot_mat, trans_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    raw_pitch = float(euler_angles[0][0])
    raw_yaw = -float(euler_angles[1][0])

    # EMA smoothing
    if _ema_pitch is None:
        _ema_pitch, _ema_yaw = raw_pitch, raw_yaw
    else:
        _ema_pitch = EMA_ALPHA * raw_pitch + (1 - EMA_ALPHA) * _ema_pitch
        _ema_yaw = EMA_ALPHA * raw_yaw + (1 - EMA_ALPHA) * _ema_yaw

    return _ema_pitch, _ema_yaw


class DriverMonitor:
    """
    Fine-tuned driver monitoring using YOLO ONNX (behavior detection) + MediaPipe (face landmarks).
    Detects: drowsiness, yawning, phone use, eating, drinking, gaze distraction.
    """

    def __init__(self):
        self.yolo_dms = None
        self.face_detector = None
        self.class_names = {}
        self._init_models()

        # Buffers for smoothing detections
        self.ear_buffer = deque(maxlen=BUFFER_WINDOW)
        self.mar_buffer = deque(maxlen=BUFFER_WINDOW)
        self.gaze_buffer = deque(maxlen=BUFFER_WINDOW)
        self.phone_buffer = deque(maxlen=BUFFER_WINDOW)
        self.ingestion_buffer = deque(maxlen=BUFFER_WINDOW)

    def _init_models(self):
        """Initialize YOLO ONNX and MediaPipe face landmarker."""
        # Load YOLO ONNX model for behavior detection
        if ONNX_AVAILABLE:
            try:
                self.yolo_dms = onnxruntime.InferenceSession(CUSTOM_YOLO_DMS_PATH)
                print(f"[DriverMonitor] ONNX DMS model loaded: {CUSTOM_YOLO_DMS_PATH}")
            except Exception as e:
                print(f"[DriverMonitor] Failed to load ONNX DMS: {e}")
                self.yolo_dms = None
        elif YOLO_AVAILABLE:
            # Fallback to YOLO if ONNX not available (for .pt format)
            try:
                self.yolo_dms = YOLO(CUSTOM_YOLO_DMS_PATH, task='detect')
                print(f"[DriverMonitor] YOLO DMS model loaded: {CUSTOM_YOLO_DMS_PATH}")
            except Exception as e:
                print(f"[DriverMonitor] Failed to load YOLO DMS: {e}")
                self.yolo_dms = None

        # Load class names
        try:
            import os
            if os.path.exists(CUSTOM_YOLO_CLASSES_PATH):
                with open(CUSTOM_YOLO_CLASSES_PATH, 'r') as f:
                    self.class_names = json.load(f)
                print(f"[DriverMonitor] Classes loaded: {list(self.class_names.values())}")
        except Exception as e:
            print(f"[DriverMonitor] Failed to load classes: {e}")

        # Load MediaPipe face landmarker
        if MP_AVAILABLE:
            try:
                task_path = _ensure_face_landmarker_downloaded()
                if task_path:
                    base_options = python.BaseOptions(model_asset_path=task_path)
                    options = vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=False,
                        num_faces=1
                    )
                    self.face_detector = vision.FaceLandmarker.create_from_options(options)
                    print("[DriverMonitor] MediaPipe face landmarker initialized")
            except Exception as e:
                print(f"[DriverMonitor] Failed to init face landmarker: {e}")
                self.face_detector = None

    def process(self, frame: np.ndarray) -> DriverState:
        """Process frame and return driver state."""
        if self.face_detector is None or self.yolo_dms is None:
            return DriverState(state="NORMAL", alert_message="Models not loaded")

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.face_detector.detect(mp_image)

        is_drowsy = False
        is_yawning = False
        is_distracted_gaze = False
        current_pitch = 0.0
        current_yaw = 0.0
        avg_ear = 0.0
        current_mar = 0.0

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            # Calculate EAR
            left_ear = _calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = _calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            self.ear_buffer.append(avg_ear < EAR_THRESHOLD)

            # Calculate MAR
            current_mar = _calculate_mar(landmarks, MOUTH, w, h)
            self.mar_buffer.append(current_mar > MAR_THRESHOLD)

            # Estimate head pose
            current_pitch, current_yaw = _estimate_head_pose(landmarks, w, h)

            # Thresholds
            if sum(self.ear_buffer) >= int(0.8 * BUFFER_WINDOW):
                is_drowsy = True
            if sum(self.mar_buffer) >= int(0.6 * BUFFER_WINDOW):
                is_yawning = True

            gaze_distracted = (abs(current_yaw) > YAW_THRESHOLD or current_pitch > PITCH_THRESHOLD)
            self.gaze_buffer.append(gaze_distracted)
            if sum(self.gaze_buffer) >= int(0.7 * BUFFER_WINDOW):
                is_distracted_gaze = True
        else:
            # No face detected
            self.ear_buffer.append(True)
            self.gaze_buffer.append(True)
            if sum(self.ear_buffer) >= int(0.8 * BUFFER_WINDOW):
                is_drowsy = True
            if sum(self.gaze_buffer) >= int(0.7 * BUFFER_WINDOW):
                is_distracted_gaze = True

        # YOLO behavior detection (phone, eating, drinking, etc.)
        frame_has_phone = False
        frame_has_ingestion = False

        if isinstance(self.yolo_dms, onnxruntime.InferenceSession):
            # ONNX inference
            try:
                frame_input = cv2.resize(frame, (640, 640))
                frame_blob = frame_input.astype(np.float32) / 255.0
                frame_blob = np.transpose(frame_blob, (2, 0, 1))
                frame_blob = np.expand_dims(frame_blob, 0)

                inputs = {self.yolo_dms.get_inputs()[0].name: frame_blob}
                outputs = self.yolo_dms.run(None, inputs)
                predictions = outputs[0][0]

                for pred in predictions:
                    if len(pred) < 6:
                        continue
                    conf = pred[4]
                    if conf < 0.55:
                        continue
                    cls_scores = pred[5:]
                    if len(cls_scores) > 0:
                        cls_id = int(np.argmax(cls_scores))
                        cls_name = self.class_names.get(str(cls_id), "unknown")
                        if cls_name == "phone":
                            frame_has_phone = True
                        elif cls_name in ("eating", "drinking", "cigarette"):
                            frame_has_ingestion = True
            except Exception as e:
                print(f"[DriverMonitor] ONNX prediction error: {e}")
        elif self.yolo_dms is not None:
            try:
                yolo_results = self.yolo_dms.predict(source=frame, conf=0.55, verbose=False)
                for box in yolo_results[0].boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names.get(str(cls_id), "unknown")
                    if cls_name == "phone":
                        frame_has_phone = True
                    elif cls_name in ("eating", "drinking", "cigarette"):
                        frame_has_ingestion = True
            except Exception as e:
                print(f"[DriverMonitor] YOLO prediction error: {e}")

        self.phone_buffer.append(frame_has_phone)
        self.ingestion_buffer.append(frame_has_ingestion)

        detected_phone = sum(self.phone_buffer) >= int(0.3 * BUFFER_WINDOW)
        detected_ingestion = sum(self.ingestion_buffer) >= int(0.3 * BUFFER_WINDOW)

        # Determine state
        state = "NORMAL"
        if is_drowsy:
            state = "DROWSY"
        elif detected_phone:
            state = "TEXTING" if current_pitch < -5 else "PHONE"
        elif is_distracted_gaze:
            state = "GAZE_DISTRACTED"
        elif detected_ingestion:
            state = "INGESTION"
        elif is_yawning:
            state = "YAWNING"

        return DriverState(
            state=state,
            ear=avg_ear,
            mar=current_mar,
            yaw=current_yaw,
            pitch=current_pitch,
            phone_detected=detected_phone,
            ingestion_detected=detected_ingestion,
            alert_message=self._state_to_alert(state)
        )

    @staticmethod
    def _state_to_alert(state: str) -> Optional[str]:
        alerts = {
            "DROWSY":           "⚠️ DROWSINESS DETECTED - STAY ALERT",
            "TEXTING":          "🚨 TEXTING & DRIVING - CRITICAL DANGER",
            "PHONE":            "📵 PHONE DISTRACTION DETECTED",
            "INGESTION":        "🍽️ EATING/DRINKING WHILE DRIVING",
            "GAZE_DISTRACTED":  "👀 DRIVER GAZE DISTRACTED",
            "YAWNING":          "😴 FATIGUE - YAWNING DETECTED",
        }
        return alerts.get(state)

    def draw_overlay(self, frame: np.ndarray, state: DriverState) -> np.ndarray:
        """Draw driver state overlay on frame."""
        out = frame.copy()
        info = DRIVER_STATES.get(state.state, DRIVER_STATES["NORMAL"])
        color = info["color"]
        label = info["label"]

        # Status box top-left
        h, w = out.shape[:2]
        overlay = out.copy()
        cv2.rectangle(overlay, (8, 8), (320, 110), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

        cv2.putText(out, "DRIVER MONITOR", (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(out, label, (16, 65),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)

        metrics = f"EAR:{state.ear:.2f} Yaw:{state.yaw:.0f}° Pitch:{state.pitch:.0f}°"
        cv2.putText(out, metrics, (16, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        # Alert banner for critical states
        if state.state in ("DROWSY", "TEXTING", "PHONE", "INGESTION"):
            banner_h = 50
            overlay2 = out.copy()
            cv2.rectangle(overlay2, (0, h - banner_h), (w, h), color, -1)
            cv2.addWeighted(overlay2, 0.7, out, 0.3, 0, out)
            msg = state.alert_message or label
            cv2.putText(out, msg, (16, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return out
