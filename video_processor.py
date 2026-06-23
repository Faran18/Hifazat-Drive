"""
video_processor.py
Orchestrates YOLO + MediaPipe + AlertManager to process a video file frame-by-frame.
Yields annotated frames + per-frame event data.
"""

# ── Custom Model Paths ───────────────────────────────────────────────────────
# Vehicle Detection Model (Fine-tuned YOLO)
CUSTOM_YOLO_VEHICLE_PATH = r"CWS.pt"  # Fine-tuned vehicle detector

# Driver Monitoring Model (ONNX + Classes)
CUSTOM_YOLO_DMS_PATH = r"yolo_dms.onnx"  # Fine-tuned driver behavior detector
CUSTOM_YOLO_CLASSES_PATH = r"yolo_classes.json"  # Driver behavior classes

import cv2
import numpy as np
import time
import os
import sys
from dataclasses import dataclass, field
from typing import Iterator, List, Dict, Optional, Tuple

from yolo_detector import YOLODetector, Detection
from driver_monitor import DriverMonitor, DriverState, DRIVER_STATES
from Voice_Agent.alert_manager import AlertManager, Alert, build_driver_alerts, build_vehicle_alert


@dataclass
class FrameResult:
    frame_number: int
    annotated_frame: np.ndarray
    detections: List[Detection]
    driver_state: Optional[DriverState]
    alerts: List[Alert]
    fps: float


@dataclass
class ProcessingConfig:
    mode: str = "combined"          # "vehicle", "driver", "combined"
    skip_frames: int = 2            # process every N frames (1 = all)
    confidence: float = 0.4
    enable_voice: bool = False
    yolo_custom_path: str = ""     # deprecated; use CUSTOM_YOLO_MODEL_PATH constant
    sensitivity: float = 0.5        # 0-1 slider; affects thresholds


class VideoProcessor:
    def __init__(self, config: ProcessingConfig):
        self.cfg = config

        self.yolo: Optional[YOLODetector] = None
        self.driver: Optional[DriverMonitor] = None
        self.alert_mgr = AlertManager()
        # Voice alerts now handled via Web Speech API in browser

        if config.mode in ("vehicle", "combined"):
            self.yolo = YOLODetector(
                model_path=CUSTOM_YOLO_VEHICLE_PATH,
                confidence=config.confidence
            )

        if config.mode in ("driver", "combined"):
            self.driver = DriverMonitor()

        # Stats accumulators
        self._total_vehicles: int = 0
        self._driver_state_counts: Dict[str, int] = {}
        self._frame_log: List[dict] = []

    def process_video(self, video_path: str) -> Iterator[Tuple[FrameResult, dict]]:
        """
        Generator: yields (FrameResult, stats_snapshot) for each processed frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_num    = 0
        t_prev       = time.time()

        self.alert_mgr.reset()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Frame skipping: still read but skip heavy inference
            process_this = (frame_num % max(self.cfg.skip_frames, 1) == 0)

            detections: List[Detection] = []
            driver_st: Optional[DriverState] = None
            frame_alerts: List[Alert] = []

            if process_this:
                annotated = frame.copy()

                # ── YOLO ──────────────────────────────────────────────────
                if self.yolo:
                    detections = self.yolo.detect(frame, skip_frames=1)
                    annotated  = self.yolo.draw(annotated, detections)
                    self._total_vehicles += len(detections)

                    for det in detections:
                        a = build_vehicle_alert(det.distance_category, det.class_name,
                                                self.alert_mgr, frame_num)
                        if a:
                            frame_alerts.append(a)

                # ── Driver Monitor ────────────────────────────────────────
                if self.driver:
                    driver_st = self.driver.process(frame)
                    annotated = self.driver.draw_overlay(annotated, driver_st)

                    s = driver_st.state
                    self._driver_state_counts[s] = self._driver_state_counts.get(s, 0) + 1

                    a = build_driver_alerts(s, self.alert_mgr, frame_num)
                    if a:
                        frame_alerts.append(a)

                # ── Voice (handled by Web Speech API in browser) ───────────────────
                # No backend voice processing needed

                # ── FPS ───────────────────────────────────────────────────
                t_now  = time.time()
                fps    = 1.0 / max(t_now - t_prev, 1e-6)
                t_prev = t_now

                # ── HUD: frame counter + progress ─────────────────────────
                progress_pct = int(frame_num / max(total_frames, 1) * 100)
                cv2.putText(annotated,
                            f"Frame {frame_num}/{total_frames}  {progress_pct}%  {fps:.1f}fps",
                            (annotated.shape[1] - 300, annotated.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                # Log
                self._frame_log.append({
                    "frame": frame_num,
                    "vehicles": len(detections),
                    "driver_state": driver_st.state if driver_st else "N/A",
                    "alerts": [a.alert_type for a in frame_alerts],
                })

                result = FrameResult(
                    frame_number=frame_num,
                    annotated_frame=annotated,
                    detections=detections,
                    driver_state=driver_st,
                    alerts=frame_alerts,
                    fps=fps,
                )

                stats = self._build_stats(frame_num, total_frames)
                yield result, stats

        cap.release()

    def _build_stats(self, current_frame: int, total_frames: int) -> dict:
        total_st = sum(self._driver_state_counts.values()) or 1
        return {
            "total_vehicles":   self._total_vehicles,
            "critical_alerts":  self.alert_mgr.get_critical_count(),
            "warning_alerts":   self.alert_mgr.get_warning_count(),
            "driver_states":    dict(self._driver_state_counts),
            "alert_counts":     self.alert_mgr.get_counts(),
            "progress":         current_frame / max(total_frames, 1),
            "drowsy_pct":       self._driver_state_counts.get("DROWSY", 0) / total_st * 100,
            "distracted_pct":   self._driver_state_counts.get("DISTRACTED", 0) / total_st * 100,
            "sleeping_pct":     self._driver_state_counts.get("SLEEPING", 0) / total_st * 100,
            "phone_pct":        self._driver_state_counts.get("USING_PHONE", 0) / total_st * 100,
            "frame_log":        list(self._frame_log[-200:]),  # last 200 entries for display
        }

    def get_full_log(self):
        return list(self._frame_log)

    def stop(self):
        # Cleanup (voice agent no longer used)
        pass