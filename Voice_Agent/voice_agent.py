"""
voice_agent.py
TTS voice alert engine using gTTS (online) with pygame for playback.
Falls back gracefully if dependencies are missing.
"""

import os
import io
import time
import threading
import queue
from typing import Optional


# ── Dependency flags ──────────────────────────────────────────────────────────
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


class VoiceAgent:
    """
    Async TTS voice agent.
    - Maintains a priority queue of messages.
    - Background thread dequeues and speaks.
    - Per-message cooldown to prevent spam.
    """

    def __init__(self, enabled: bool = True, lang: str = "en"):
        self.enabled = enabled
        self.lang = lang
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._spoken: dict[str, float] = {}
        self._cooldown = 4.0  # seconds between identical messages
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pyttsx_engine = None

        if enabled:
            self._init_engine()
            self._start()

    def _init_engine(self):
        """Prefer pyttsx3 (offline), fallback to gTTS (online)."""
        if PYTTSX3_AVAILABLE:
            try:
                self._pyttsx_engine = pyttsx3.init()
                self._pyttsx_engine.setProperty("rate", 165)
                self._pyttsx_engine.setProperty("volume", 1.0)
                print("[VoiceAgent] Using pyttsx3 (offline TTS)")
                return
            except Exception as e:
                print(f"[VoiceAgent] pyttsx3 init failed: {e}")
                self._pyttsx_engine = None
        
        if GTTS_AVAILABLE and PYGAME_AVAILABLE:
            print("[VoiceAgent] Using gTTS + pygame (online TTS)")
        else:
            print("[VoiceAgent] TTS unavailable - alerts will print to console")

    def _start(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, message: str, priority: int = 50):
        """Enqueue a message. Lower priority number = higher urgency."""
        if not self.enabled:
            return
        now = time.time()
        last = self._spoken.get(message, 0)
        if now - last < self._cooldown:
            return
        self._spoken[message] = now
        self._queue.put((priority, message))

    def _worker(self):
        while self._running:
            try:
                priority, message = self._queue.get(timeout=0.5)
                self._play(message)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VoiceAgent] Worker error: {e}")

    def _play(self, message: str):
        """Try pyttsx3 first (offline), then gTTS (online), then console."""
        if self._pyttsx_engine is not None:
            self._play_pyttsx(message)
        elif GTTS_AVAILABLE and PYGAME_AVAILABLE:
            self._play_gtts(message)
        else:
            print(f"[VoiceAgent] 🔊 ALERT: {message}")

    def _play_gtts(self, message: str):
        try:
            print(f"[VoiceAgent] Speaking via gTTS: {message}")
            tts = gTTS(text=message, lang=self.lang, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            pygame.mixer.music.load(buf)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
        except Exception as e:
            print(f"[VoiceAgent] gTTS error: {e}")

    def _play_pyttsx(self, message: str):
        try:
            print(f"[VoiceAgent] Speaking: {message}")
            self._pyttsx_engine.say(message)
            self._pyttsx_engine.runAndWait()
        except Exception as e:
            print(f"[VoiceAgent] pyttsx3 error: {e}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if enabled and not self._running:
            self._start()
