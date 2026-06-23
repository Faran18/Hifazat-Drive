"""
app.py  –  AI Driving Safety Analysis System
Streamlit frontend for dual-AI driving safety analysis.
"""

import os
import sys
import io
import time
import tempfile
import threading

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from video_processor import VideoProcessor, ProcessingConfig
from Voice_Agent.alert_manager import ALERT_CONFIG

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hifazat Drive",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS theme ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark dashboard feel */
  .stApp { background: #0d1117; color: #e6edf3; }
  section[data-testid="stSidebar"] { background: #161b22; }
  section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

  /* Cards */
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    margin-bottom: 8px;
  }
  .metric-card .val { font-size: 2.4rem; font-weight: 700; line-height: 1; }
  .metric-card .lbl { font-size: 0.78rem; color: #8b949e; margin-top: 4px; }
  .metric-card.red   { border-color: #FF2D2D; }
  .metric-card.orange{ border-color: #FF5500; }
  .metric-card.blue  { border-color: #58a6ff; }
  .metric-card.green { border-color: #3fb950; }

  /* Alert banner */
  .alert-banner {
    border-radius: 8px;
    padding: 10px 16px;
    font-weight: 600;
    font-size: 1rem;
    margin: 4px 0;
  }
  .alert-critical { background: #3d0000; border-left: 4px solid #FF2D2D; color: #ff8080; }
  .alert-warning  { background: #2d2000; border-left: 4px solid #FFA500; color: #ffd080; }

  /* Section headers */
  .section-title {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b949e;
    margin: 20px 0 10px;
    padding-bottom: 4px;
    border-bottom: 1px solid #30363d;
  }

  /* Progress bar */
  .stProgress > div > div { background: #58a6ff !important; }

  /* Video frame border */
  .video-frame img { border-radius: 8px; border: 1px solid #30363d; }

  /* Scrollable log */
  .log-scroll {
    max-height: 300px;
    overflow-y: auto;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px;
    font-family: monospace;
    font-size: 0.75rem;
  }

  /* Hide default Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Web Speech API Component ─────────────────────────────────────────────────
def create_speech_component(alerts_text: str):
    """Create HTML component that reads alerts using Web Speech API."""
    html_code = f"""
    <script>
    function speakAlert(text) {{
        if ('speechSynthesis' in window && text.trim()) {{
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.2;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            window.speechSynthesis.speak(utterance);
        }}
    }}
    // Auto-speak when component loads
    speakAlert(`{alerts_text}`);
    </script>
    """
    st.components.v1.html(html_code, height=0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def metric_card(value, label, color="blue"):
    return f"""
    <div class="metric-card {color}">
      <div class="val">{value}</div>
      <div class="lbl">{label}</div>
    </div>"""


def alert_html(message, level="WARNING"):
    cls = "alert-critical" if level == "CRITICAL" else "alert-warning"
    icon = "🚨" if level == "CRITICAL" else "⚠️"
    return f'<div class="alert-banner {cls}">{icon} {message}</div>'


def bgr_to_rgb_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown("## 🚗 Hifazat Drive")
        st.markdown('<div class="section-title">Analysis Mode</div>', unsafe_allow_html=True)

        mode = st.selectbox(
            "Detection Mode",
            options=["combined", "vehicle", "driver"],
            format_func=lambda m: {
                "combined": "🔀 Combined (Vehicle + Driver)",
                "vehicle":  "🚙 Vehicle Detection Only",
                "driver":   "👁 Driver Monitoring Only",
            }[m],
            key="mode",
        )

        st.markdown('<div class="section-title">Upload</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload Driving Video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a dashcam or driving video (MP4 recommended)",
        )

        st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)

        skip = st.slider("Frame Skip (higher = faster)", 1, 6, 2,
                         help="Process every Nth frame to speed up analysis")
        conf = st.slider("Detection Confidence", 0.2, 0.9, 0.4, 0.05)
        sensitivity = st.slider("Alert Sensitivity", 0.1, 1.0, 0.5, 0.1)

        voice = st.toggle("🔊 Enable Voice Alerts", value=True,
                          help="Requires pyttsx3 (offline) or gTTS + pygame")

        st.markdown('<div class="section-title">YOLO Model</div>', unsafe_allow_html=True)
        st.info("Using custom YOLO model defined in code.")

        return {
            "mode": mode,
            "uploaded": uploaded,
            "skip": skip,
            "conf": conf,
            "sensitivity": sensitivity,
            "voice": voice,
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = sidebar()

    # ── Hero header ──────────────────────────────────────────────────────────
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("# 🚗 Hifazat Drive")
        st.markdown('<span style="color:#8b949e;font-size:0.9rem;">Dual-AI Driving Safety Analysis System — Vehicle Detection + Driver Monitoring</span>',
                    unsafe_allow_html=True)
    with col_h2:
        mode_badges = {
            "combined": "🔀 Combined Mode",
            "vehicle":  "🚙 Vehicle Only",
            "driver":   "👁 Driver Only",
        }
        st.markdown(f'<div style="text-align:right;padding-top:8px;color:#58a6ff;font-weight:600;">{mode_badges[cfg["mode"]]}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── Upload preview ───────────────────────────────────────────────────────
    if cfg["uploaded"] is None:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;border:1px dashed #30363d;border-radius:12px;background:#161b22;">
          <div style="font-size:3rem;">📁</div>
          <div style="font-size:1.3rem;font-weight:600;margin:12px 0 6px;">Upload a Driving Video</div>
          <div style="color:#8b949e;font-size:0.9rem;">Supports MP4, AVI, MOV, MKV — Use the sidebar uploader to begin</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Layout: video | live stats ────────────────────────────────────────────
    col_vid, col_stats = st.columns([3, 2], gap="medium")

    with col_vid:
        st.markdown('<div class="section-title">Live Analysis Feed</div>', unsafe_allow_html=True)
        video_placeholder = st.empty()
        progress_bar      = st.progress(0)
        status_txt        = st.empty()
        alert_placeholder = st.empty()

    with col_stats:
        st.markdown('<div class="section-title">Real-Time Metrics</div>', unsafe_allow_html=True)

        metric_row = st.empty()

        st.markdown('<div class="section-title">Driver Behavior</div>', unsafe_allow_html=True)
        behavior_placeholder = st.empty()

        st.markdown('<div class="section-title">Alert Log</div>', unsafe_allow_html=True)
        log_placeholder = st.empty()

    # ── Dashboard (below video) ───────────────────────────────────────────────
    st.markdown('<div class="section-title">Analytics Dashboard</div>', unsafe_allow_html=True)
    dash_cols = st.columns(4)

    # ── Start button ─────────────────────────────────────────────────────────
    st.markdown("---")
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        start = st.button("▶ Start Analysis", type="primary", use_container_width=True)
        stop  = st.button("⏹ Stop", use_container_width=True)

    if "processing" not in st.session_state:
        st.session_state.processing = False
    if stop:
        st.session_state.processing = False
    if start:
        st.session_state.processing = True

    # ── Show video preview before processing ─────────────────────────────────
    if not st.session_state.processing:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(cfg["uploaded"].read())
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        ret, preview = cap.read()
        cap.release()
        if ret:
            with col_vid:
                video_placeholder.image(bgr_to_rgb_pil(preview),
                                        caption="Preview (first frame)", use_container_width=True)
        os.unlink(tmp_path)
        return

    # ── Processing loop ───────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(cfg["uploaded"].getvalue())
        tmp_path = tmp.name

    proc_config = ProcessingConfig(
        mode=cfg["mode"],
        skip_frames=cfg["skip"],
        confidence=cfg["conf"],
        enable_voice=cfg["voice"],
        sensitivity=cfg["sensitivity"],
    )

    processor = VideoProcessor(proc_config)
    alert_log_display: list = []
    last_stats: dict = {}
    final_frame_log: list = []

    try:
        for result, stats in processor.process_video(tmp_path):
            if not st.session_state.processing:
                break

            last_stats = stats

            # ── Video frame ──────────────────────────────────────────────
            with col_vid:
                video_placeholder.image(
                    bgr_to_rgb_pil(result.annotated_frame),
                    use_container_width=True,
                )
                progress_bar.progress(min(stats["progress"], 1.0))
                status_txt.markdown(
                    f'<span style="color:#8b949e;font-size:0.8rem;">Frame {result.frame_number} | '
                    f'{result.fps:.1f} fps</span>',
                    unsafe_allow_html=True,
                )

            # ── Current alerts ────────────────────────────────────────────
            if result.alerts:
                html_alerts = "".join(
                    alert_html(a.message, a.level) for a in result.alerts
                )
                alert_placeholder.markdown(html_alerts, unsafe_allow_html=True)
            else:
                alert_placeholder.empty()

            # ── Metrics ───────────────────────────────────────────────────
            with col_stats:
                metric_row.markdown(
                    metric_card(stats["critical_alerts"], "Critical Alerts",   "red")  +
                    metric_card(stats["warning_alerts"],  "Warnings",          "orange"),
                    unsafe_allow_html=True,
                )

                # Behavior bars
                def pct_bar(label, pct, color):
                    return (
                        f'<div style="margin:6px 0;">'
                        f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;margin-bottom:2px;">'
                        f'<span>{label}</span><span style="color:{color};">{pct:.1f}%</span></div>'
                        f'<div style="background:#21262d;border-radius:4px;height:6px;">'
                        f'<div style="width:{min(pct,100):.1f}%;background:{color};'
                        f'border-radius:4px;height:6px;transition:width 0.3s;"></div></div></div>'
                    )

                behavior_html = (
                    pct_bar("😴 Drowsy",       stats["drowsy_pct"],    "#FFA500") +
                    pct_bar("😵 Sleeping",      stats["sleeping_pct"],  "#FF2D2D") +
                    pct_bar("📱 Phone Use",     stats["phone_pct"],     "#FF5500") +
                    pct_bar("👀 Distracted",    stats["distracted_pct"],"#FFD700")
                )
                behavior_placeholder.markdown(behavior_html, unsafe_allow_html=True)

            # ── Voice Alert (Web Speech API) ─────────────────────────────
            if result.alerts and cfg["voice"]:
                alert_msgs = " | ".join([a.message for a in result.alerts])
                create_speech_component(alert_msgs)

            # ── Alert log
                for a in result.alerts:
                    alert_log_display.insert(0, {
                        "Frame": a.frame_number,
                        "Type":  a.alert_type,
                        "Msg":   a.message[:55],
                    })
                alert_log_display = alert_log_display[:40]

                if alert_log_display:
                    log_html = "".join(
                        f'<div style="padding:3px 0;border-bottom:1px solid #21262d;font-size:0.73rem;">'
                        f'<span style="color:#58a6ff;">#{r["Frame"]}</span> '
                        f'<span style="color:#FF6B6B;">[{r["Type"]}]</span> {r["Msg"]}</div>'
                        for r in alert_log_display[:15]
                    )
                    log_placeholder.markdown(
                        f'<div class="log-scroll">{log_html}</div>',
                        unsafe_allow_html=True
                    )

            # small yield to keep UI responsive
            time.sleep(0.01)

        # ── Processing complete ───────────────────────────────────────────
        final_frame_log = processor.get_full_log()
        st.session_state.processing = False
        processor.stop()

    except Exception as e:
        st.error(f"Processing error: {e}")
        st.session_state.processing = False
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # ── Post-run dashboard ────────────────────────────────────────────────────
    if last_stats:
        st.markdown("---")
        st.markdown("## 📊 Analysis Complete")

        kpi_cols = st.columns(4)
        kpis = [
            #(last_stats.get("total_vehicles", 0),  "Total Vehicles",  "blue",   "🚗"),
            (last_stats.get("critical_alerts", 0), "Critical Alerts", "red",    "🚨"),
            (last_stats.get("warning_alerts", 0),  "Warnings",        "orange", "⚠️"),
            (f"{last_stats.get('drowsy_pct', 0):.1f}%", "Time Drowsy", "green", "😴"),
        ]
        for col, (v, l, c, icon) in zip(kpi_cols, kpis):
            with col:
                st.markdown(metric_card(f"{icon} {v}", l, c), unsafe_allow_html=True)

        # ── Alert breakdown table ─────────────────────────────────────────
        st.markdown('<div class="section-title">Alert Breakdown</div>', unsafe_allow_html=True)
        ac = last_stats.get("alert_counts", {})
        if ac:
            df = pd.DataFrame([
                {"Alert Type": k, "Count": v,
                 "Level": ALERT_CONFIG.get(k, {}).get("level", "?"),
                 "Priority": ALERT_CONFIG.get(k, {}).get("priority", 0)}
                for k, v in ac.items()
            ]).sort_values("Priority", ascending=False)
            st.dataframe(df.drop(columns=["Priority"]), use_container_width=True, hide_index=True)

        # ── Frame log / CSV export ────────────────────────────────────────
        st.markdown('<div class="section-title">Event Timeline</div>', unsafe_allow_html=True)
        if final_frame_log:
            log_df = pd.DataFrame(final_frame_log)
            log_df["alerts"] = log_df["alerts"].apply(lambda x: ", ".join(x) if x else "—")
            st.dataframe(log_df.tail(100), use_container_width=True, hide_index=True)

            csv_bytes = log_df.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download Full Event Log (CSV)",
                data=csv_bytes,
                file_name="drivesafe_event_log.csv",
                mime="text/csv",
            )

        # Driver behaviour pie (text summary)
        st.markdown('<div class="section-title">Driver Behaviour Summary</div>', unsafe_allow_html=True)
        ds = last_stats.get("driver_states", {})
        total = sum(ds.values()) or 1
        summary_cols = st.columns(len(ds) or 1)
        state_colors = {"NORMAL": "#3fb950", "DROWSY": "#FFA500", "SLEEPING": "#FF2D2D",
                        "USING_PHONE": "#FF5500", "DISTRACTED": "#FFD700"}
        for col, (state, count) in zip(summary_cols, ds.items()):
            pct = count / total * 100
            col.markdown(
                f'<div style="text-align:center;padding:12px;background:#161b22;'
                f'border:1px solid {state_colors.get(state,"#30363d")};border-radius:8px;">'
                f'<div style="font-size:1.5rem;font-weight:700;color:{state_colors.get(state,"#fff")};">'
                f'{pct:.1f}%</div>'
                f'<div style="font-size:0.75rem;color:#8b949e;margin-top:4px;">{state}</div></div>',
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()