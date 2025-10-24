# dashboard_app.py
import streamlit as st
import cv2
import threading
import time
from face_gaze_headpose import FaceGazeHeadpose
from phone_detector import PhoneDetector
from attention_engine import AttentionEngine
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide", page_title="Exam Focus Monitor")

# Sidebar config
st.sidebar.title("Settings")
fps_target = st.sidebar.slider("Processing FPS", 1, 20, 8)
show_boxes = st.sidebar.checkbox("Show phone boxes", True)

st.title("Focus & Device Detection — Live Monitor")
col1, col2 = st.columns([2,1])

# placeholders
video_placeholder = col1.empty()
info_placeholder = col2.empty()
log_placeholder = st.empty()

# Initialize detectors (singletons)
face_detector = FaceGazeHeadpose()
phone_detector = PhoneDetector(model_name="yolov8n")  # change to None for fallback
engine = AttentionEngine()

# Capture device
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

stop_flag = False
frame_lock = threading.Lock()
last_frame = None

def capture_loop():
    global last_frame, stop_flag
    interval = 1.0 / fps_target
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        with frame_lock:
            last_frame = frame.copy()
        time.sleep(interval)

t = threading.Thread(target=capture_loop, daemon=True)
t.start()

# Main processing loop inside Streamlit
try:
    while True:
        start = time.time()
        with frame_lock:
            if last_frame is None:
                time.sleep(0.05)
                continue
            frame = last_frame.copy()
        # Flip horizontally for user-friendly view
        frame = cv2.flip(frame, 1)

        # Process face/gaze/headpose
        fg = face_detector.process(frame)
        # Detect phone
        dets, phone_present = phone_detector.detect(frame) if phone_detector.model is not None else ([], False)

        obs = {}
        obs.update(fg)
        obs["phone_present"] = phone_present

        result = engine.update(obs)

        # Draw overlays
        overlay = frame.copy()
        h,w = frame.shape[:2]
        if fg.get("face_present", False):
            # draw head angles text
            cv2.putText(overlay, f"Yaw:{fg['head_yaw']:.1f} Pitch:{fg['head_pitch']:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            # draw iris centers
            lx,ly = int(fg["left_iris"][0]*w), int(fg["left_iris"][1]*h)
            rx,ry = int(fg["right_iris"][0]*w), int(fg["right_iris"][1]*h)
            cv2.circle(overlay, (lx,ly), 3, (0,255,0), -1)
            cv2.circle(overlay, (rx,ry), 3, (0,255,0), -1)

        # draw phone boxes
        if show_boxes and dets:
            for d in dets:
                x1,y1,x2,y2 = d["box"]
                clsname = d["class"]
                conf = d["conf"]
                color = (0,0,255) if "phone" in clsname.lower() or "cell" in clsname.lower() else (255,0,0)
                cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2)
                cv2.putText(overlay, f"{clsname}:{conf:.2f}", (x1,y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # draw score box
        score_text = f"Score: {result['score']:.1f} ({result['label']})"
        cv2.rectangle(overlay, (5,h-60), (360,h-5), (0,0,0), -1)
        cv2.putText(overlay, score_text, (10,h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        if result["alerts"]:
            y0 = 10
            for a in result["alerts"]:
                cv2.putText(overlay, f"ALERT: {a}", (10, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                y0 += 30

        # convert to RGB for streamlit
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        video_placeholder.image(overlay_rgb, channels="RGB", use_column_width=True)

        # info panel
        info_str = f"**Live info**\n\n- Score: {result['score']:.1f}\n- Label: {result['label']}\n- Alerts: {', '.join(result['alerts']) if result['alerts'] else 'None'}\n- Recent events: {len(engine.events)}"
        info_placeholder.markdown(info_str)

        # Log area (last few events)
        recent = list(engine.events)[-10:][::-1]
        lines = ["### Event Log (recent)"]
        for ts,etype,detail in recent:
            lines.append(f"- {datetime.fromtimestamp(ts).strftime('%H:%M:%S')} — {etype} — {detail}")
        log_placeholder.markdown("\n".join(lines))

        # control framerate
        elapsed = time.time() - start
        sleep = max(0.01, (1.0/fps_target) - elapsed)
        time.sleep(sleep)

except KeyboardInterrupt:
    stop_flag = True
    cap.release()
    st.write("Stopped")

