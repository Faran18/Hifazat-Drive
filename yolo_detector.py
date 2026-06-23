"""
yolo_detector.py
Vehicle detection and distance estimation using YOLO.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time

# Try to import ultralytics, fall back to a mock for environments without GPU
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    distance_category: str  # "VERY_CLOSE", "MEDIUM", "FAR"
    distance_label: str
    center: Tuple[int, int]
    area_ratio: float  # bbox area / frame area


VEHICLE_CLASSES = {
    "car", "truck", "bus", "motorcycle", "bicycle",
    "motorbike", "van", "boat", "train"
}

# Color map per distance
DISTANCE_COLORS = {
    "VERY_CLOSE": (0, 0, 255),    # Red
    "MEDIUM":     (0, 165, 255),  # Orange
    "FAR":        (0, 255, 0),    # Green
}

# Thresholds: fraction of frame area occupied by bbox
CLOSE_THRESHOLD  = 0.08   # > 8% → VERY CLOSE
MEDIUM_THRESHOLD = 0.02   # 2–8% → MEDIUM
# < 2% → FAR


class YOLODetector:
    def __init__(self, model_path: str, confidence: float = 0.4):
        if not model_path:
            raise ValueError("model_path is required. Provide path to your custom YOLO model.")
        self.confidence = confidence
        self.model = None
        self._load_model(model_path)
        self.frame_count = 0
        self.last_detections: List[Detection] = []

    def _load_model(self, model_path: str):
        if not YOLO_AVAILABLE:
            raise ImportError("[YOLO] ultralytics not available. Install with: pip install ultralytics")
        try:
            self.model = YOLO(model_path)
            print(f"[YOLO] Vehicle model loaded: {model_path}")
        except Exception as e:
            raise RuntimeError(f"[YOLO] Failed to load vehicle model from {model_path}: {e}")

    def _estimate_distance(self, bbox: Tuple[int, int, int, int],
                           frame_h: int, frame_w: int) -> Tuple[str, str, float]:
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_h * frame_w
        ratio = bbox_area / frame_area if frame_area > 0 else 0

        if ratio > CLOSE_THRESHOLD:
            return "VERY_CLOSE", "⚠ VERY CLOSE", ratio
        elif ratio > MEDIUM_THRESHOLD:
            return "MEDIUM", "~ MEDIUM", ratio
        else:
            return "FAR", "· FAR", ratio

    def detect(self, frame: np.ndarray, skip_frames: int = 1) -> List[Detection]:
        self.frame_count += 1
        if self.frame_count % max(skip_frames, 1) != 0:
            return self.last_detections

        h, w = frame.shape[:2]
        detections: List[Detection] = []

        if self.model is None:
            # Mock: return a single fake detection for demo purposes
            mock_bbox = (int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7))
            dist_cat, dist_label, ratio = self._estimate_distance(mock_bbox, h, w)
            detections.append(Detection(
                class_name="car",
                confidence=0.85,
                bbox=mock_bbox,
                distance_category=dist_cat,
                distance_label=dist_label,
                center=(int(w * 0.5), int(h * 0.5)),
                area_ratio=ratio
            ))
            self.last_detections = detections
            return detections

        try:
            results = self.model(frame, conf=self.confidence, verbose=False)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = results.names[cls_id].lower()
                if cls_name not in VEHICLE_CLASSES:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dist_cat, dist_label, ratio = self._estimate_distance((x1, y1, x2, y2), h, w)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                detections.append(Detection(
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    distance_category=dist_cat,
                    distance_label=dist_label,
                    center=(cx, cy),
                    area_ratio=ratio
                ))
        except Exception as e:
            print(f"[YOLO] Inference error: {e}")

        self.last_detections = detections
        return detections

    def draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = DISTANCE_COLORS[det.distance_category]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name.upper()} {det.confidence:.0%} | {det.distance_label}"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - lh - baseline - 4), (x1 + lw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return out

    def get_worst_alert(self, detections: List[Detection]) -> Optional[str]:
        """Return the highest-priority alert string for current detections."""
        for det in detections:
            if det.distance_category == "VERY_CLOSE":
                return f"Alert! {det.class_name.capitalize()} is very close ahead!"
        for det in detections:
            if det.distance_category == "MEDIUM":
                return f"Warning! Vehicle detected at medium distance!"
        return None
