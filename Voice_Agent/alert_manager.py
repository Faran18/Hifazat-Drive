"""
alert_manager.py
Priority-based alert system with cooldown to prevent alert spam.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import defaultdict


# Alert levels
ALERT_CRITICAL = "CRITICAL"
ALERT_WARNING  = "WARNING"
ALERT_INFO     = "INFO"

# Alert types with priority (higher = more urgent) and cooldown (seconds)
ALERT_CONFIG: Dict[str, dict] = {
    "SLEEPING":      {"priority": 100, "cooldown": 4.0,  "level": ALERT_CRITICAL, "color": "#FF2D2D"},
    "USING_PHONE":   {"priority": 90,  "cooldown": 5.0,  "level": ALERT_CRITICAL, "color": "#FF5500"},
    "VERY_CLOSE":    {"priority": 80,  "cooldown": 3.0,  "level": ALERT_CRITICAL, "color": "#FF4500"},
    "DROWSY":        {"priority": 60,  "cooldown": 5.0,  "level": ALERT_WARNING,  "color": "#FFA500"},
    "DISTRACTED":    {"priority": 40,  "cooldown": 6.0,  "level": ALERT_WARNING,  "color": "#FFD700"},
    "MEDIUM":        {"priority": 20,  "cooldown": 4.0,  "level": ALERT_WARNING,  "color": "#FFFF00"},
}


@dataclass
class Alert:
    alert_type: str
    message: str
    level: str
    color: str
    timestamp: float = field(default_factory=time.time)
    frame_number: int = 0


class AlertManager:
    def __init__(self):
        self._last_trigger: Dict[str, float] = defaultdict(float)
        self._alert_log: List[Alert] = []
        self._pending_voice: Optional[Alert] = None
        self._counts: Dict[str, int] = defaultdict(int)

    def check(self, alert_type: str, message: str, frame_number: int = 0) -> Optional[Alert]:
        """
        Submit a potential alert. Returns an Alert object if it should fire now,
        None if it's on cooldown or a higher-priority alert was just queued.
        """
        if alert_type not in ALERT_CONFIG:
            return None

        cfg = ALERT_CONFIG[alert_type]
        now = time.time()
        last = self._last_trigger[alert_type]

        if now - last < cfg["cooldown"]:
            return None  # Still cooling down

        self._last_trigger[alert_type] = now
        alert = Alert(
            alert_type=alert_type,
            message=message,
            level=cfg["level"],
            color=cfg["color"],
            timestamp=now,
            frame_number=frame_number,
        )
        self._alert_log.append(alert)
        self._counts[alert_type] += 1

        # Set as pending voice if higher priority than current pending
        if (self._pending_voice is None or
                cfg["priority"] > ALERT_CONFIG.get(self._pending_voice.alert_type, {}).get("priority", 0)):
            self._pending_voice = alert

        return alert

    def consume_voice_alert(self) -> Optional[Alert]:
        """Pop the highest-priority pending voice alert."""
        a = self._pending_voice
        self._pending_voice = None
        return a

    def get_log(self) -> List[Alert]:
        return list(self._alert_log)

    def get_counts(self) -> Dict[str, int]:
        return dict(self._counts)

    def get_critical_count(self) -> int:
        return sum(v for k, v in self._counts.items()
                   if ALERT_CONFIG.get(k, {}).get("level") == ALERT_CRITICAL)

    def get_warning_count(self) -> int:
        return sum(v for k, v in self._counts.items()
                   if ALERT_CONFIG.get(k, {}).get("level") == ALERT_WARNING)

    def reset(self):
        self._last_trigger.clear()
        self._alert_log.clear()
        self._pending_voice = None
        self._counts.clear()


def build_driver_alerts(driver_state_str: str, alert_mgr: AlertManager,
                        frame_number: int = 0) -> Optional[Alert]:
    """Map driver state string → alert manager call."""
    messages = {
        "SLEEPING":    "Warning! Driver is sleeping! Please pull over!",
        "USING_PHONE": "Danger! Do not use mobile phone while driving!",
        "DROWSY":      "Alert! Driver appears drowsy.",
        "DISTRACTED":  "Pay attention! You are distracted.",
    }
    if driver_state_str in messages:
        return alert_mgr.check(driver_state_str, messages[driver_state_str], frame_number)
    return None


def build_vehicle_alert(distance_category: str, class_name: str,
                        alert_mgr: AlertManager, frame_number: int = 0) -> Optional[Alert]:
    """Map vehicle distance → alert manager call."""
    if distance_category == "VERY_CLOSE":
        msg = f"Alert! {class_name.capitalize()} is very close ahead!"
        return alert_mgr.check("VERY_CLOSE", msg, frame_number)
    elif distance_category == "MEDIUM":
        msg = f"Warning! Vehicle detected at medium distance!"
        return alert_mgr.check("MEDIUM", msg, frame_number)
    return None
