# attention_engine.py
import time
import numpy as np
from collections import deque

class AttentionEngine:
    def __init__(self,
                 w_face=0.15, w_head=0.35, w_gaze=0.40, w_phone=0.6,
                 ema_alpha=0.3):
        self.w_face = w_face
        self.w_head = w_head
        self.w_gaze = w_gaze
        self.w_phone = w_phone
        self.ema_alpha = ema_alpha
        self.ema_score = None
        # state counters
        self.face_missing_count = 0
        self.events = deque(maxlen=1000)  # store (ts, type, details)

    def head_angle_score(self, yaw, pitch):
        # Map yaw (deg) and pitch (deg) to 0..1. 0 deg => 1. Values > 25 deg degrade.
        yaw_abs = abs(yaw); pitch_abs = abs(pitch)
        yaw_score = max(0.0, 1.0 - yaw_abs / 40.0)
        pitch_score = max(0.0, 1.0 - pitch_abs / 30.0)
        # combine
        return np.clip((yaw_score*0.6 + pitch_score*0.4), 0.0, 1.0)

    def gaze_score_from_dxdy(self, dx, dy):
        # dx,dy roughly normalized -1..1; near 0 -> looking at screen
        mag = np.linalg.norm([dx, dy])
        score = max(0.0, 1.0 - mag)  # if mag 0 => score 1. mag >1 => 0.
        return float(np.clip(score, 0.0, 1.0))

    def update(self, obs):
        """
        obs: dict from face_gaze_headpose and phone_detector results e.g.
          {'face_present':True, 'head_yaw':..., 'head_pitch':..., 'gaze_dx':..., 'gaze_dy':..., 'phone_present':bool}
        returns dict: {'score':0..100, 'label':'Focused'/'Distracted'/'Away', 'alerts':[...]}
        """
        now = time.time()
        if not obs.get("face_present", False):
            self.face_missing_count += 1
        else:
            self.face_missing_count = 0

        face_present = 1.0 if obs.get("face_present", False) else 0.0
        head_s = 0.0
        gaze_s = 0.0
        if obs.get("face_present", False):
            head_s = self.head_angle_score(obs.get("head_yaw", 0.0), obs.get("head_pitch", 0.0))
            gaze_s = self.gaze_score_from_dxdy(obs.get("gaze_dx", 0.0), obs.get("gaze_dy", 0.0))

        phone_pen = 1.0 if obs.get("phone_present", False) else 0.0
        lost_pen = 1.0 if self.face_missing_count >= 5 else 0.0  # if face lost for 5 frames (≈few seconds), penalize

        raw = (self.w_face*face_present) + (self.w_head*head_s) + (self.w_gaze*gaze_s) - (self.w_phone*phone_pen) - (0.4*lost_pen)
        raw = np.clip(raw, 0.0, 1.0)
        score = float(raw * 100.0)

        # EMA smoothing
        if self.ema_score is None:
            self.ema_score = score
        else:
            self.ema_score = (self.ema_alpha * score) + (1.0 - self.ema_alpha) * self.ema_score

        # label thresholds
        if self.ema_score >= 80:
            label = "Focused"
        elif self.ema_score >= 50:
            label = "Distracted"
        else:
            label = "Away/High Risk"

        alerts = []
        if phone_pen > 0:
            alerts.append("Phone detected in frame")
            self.events.append((now, "phone", {"ema_score":self.ema_score}))
        if lost_pen > 0:
            alerts.append("Face lost for several frames")
            self.events.append((now, "face_lost", {"count":self.face_missing_count}))
        if label != "Focused":
            # record distraction events
            self.events.append((now, "attention_"+label.lower(), {"score":self.ema_score}))

        out = {
            "score": float(self.ema_score),
            "raw_score": float(score),
            "label": label,
            "alerts": alerts,
            "events_recent": list(self.events)[-10:]
        }
        return out
