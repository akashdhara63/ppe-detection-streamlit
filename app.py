
import base64
import os
import threading
import time
from collections import deque
from datetime import datetime

import av
import cv2
import streamlit as st
from huggingface_hub import hf_hub_download
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="PPE Guardian AI",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(ROOT, "assets")
MODEL_DIR = os.path.join(ROOT, "models")
VIOLATION_DIR = os.path.join(ROOT, "static", "violations")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIOLATION_DIR, exist_ok=True)

HERO_IMAGE = os.path.join(ASSET_DIR, "hero.png")
LOCAL_MODEL = os.path.join(MODEL_DIR, "best.pt")

# This is a PPE-specific YOLOv8 model. The original project used
# yolov8n.pt (COCO), which cannot detect PPE classes such as hardhat,
# mask, gloves or safety vest.
HF_REPO = "ayushgupta7777/safetyvision-yolov8"
HF_FILENAME = "v2/best.pt"

REQUIRED_PPE = {
    "Hardhat": ("Hardhat", "NO-Hardhat"),
    "Mask": ("Mask", "NO-Mask"),
    "Gloves": ("Gloves", "NO-Gloves"),
    "Goggles": ("Goggles", "NO-Goggles"),
    "Safety Vest": ("Safety Vest", "NO-Safety Vest"),
}

VIOLATION_LABELS = {
    "NO-Hardhat": "No Hardhat",
    "NO-Mask": "No Mask",
    "NO-Gloves": "No Gloves",
    "NO-Goggles": "No Goggles",
    "NO-Safety Vest": "No Safety Vest",
    "No_Harness": "No Harness",
    "Fall-Detected": "Fall Detected",
}

POSITIVE_LABELS = {
    "Hardhat",
    "Mask",
    "Gloves",
    "Goggles",
    "Safety Vest",
}

# -----------------------------
# Styling
# -----------------------------
def image_to_data_uri(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


hero_uri = image_to_data_uri(HERO_IMAGE)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 85% 5%, rgba(37,99,235,.18), transparent 26%),
            radial-gradient(circle at 10% 35%, rgba(14,165,233,.10), transparent 30%),
            #07111f;
        color: #e5eefb;
    }}

    [data-testid="stSidebar"] {{
        background: #091525;
        border-right: 1px solid rgba(255,255,255,.08);
    }}

    .block-container {{
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        min-height: 350px;
        border-radius: 26px;
        border: 1px solid rgba(255,255,255,.12);
        background-image:
            linear-gradient(90deg, rgba(5,15,28,.96) 0%, rgba(5,15,28,.78) 42%, rgba(5,15,28,.20) 100%),
            url("{hero_uri}");
        background-size: cover;
        background-position: center;
        box-shadow: 0 25px 70px rgba(0,0,0,.35);
        display: flex;
        align-items: center;
        padding: 44px;
        margin-bottom: 22px;
    }}

    .hero-copy {{
        max-width: 640px;
    }}

    .eyebrow {{
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(37,99,235,.20);
        border: 1px solid rgba(96,165,250,.35);
        color: #93c5fd;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }}

    .hero h1 {{
        font-size: clamp(2.2rem, 5vw, 4.1rem);
        line-height: 1.02;
        margin: 16px 0 12px;
        color: white;
        font-weight: 800;
        letter-spacing: -.04em;
    }}

    .hero p {{
        color: #c7d6ea;
        font-size: 1.03rem;
        line-height: 1.7;
        margin: 0;
    }}

    .glass {{
        background: rgba(13, 27, 45, .72);
        border: 1px solid rgba(255,255,255,.09);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 12px 35px rgba(0,0,0,.18);
        backdrop-filter: blur(14px);
    }}

    .section-title {{
        font-size: 1.25rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 4px 0 12px;
    }}

    .status {{
        padding: 14px 16px;
        border-radius: 14px;
        font-weight: 800;
        margin: 8px 0;
    }}

    .status-ok {{
        background: rgba(16,185,129,.13);
        color: #6ee7b7;
        border: 1px solid rgba(16,185,129,.25);
    }}

    .status-bad {{
        background: rgba(239,68,68,.13);
        color: #fca5a5;
        border: 1px solid rgba(239,68,68,.25);
    }}

    .status-warn {{
        background: rgba(245,158,11,.13);
        color: #fcd34d;
        border: 1px solid rgba(245,158,11,.25);
    }}

    .ppe-row {{
        display: flex;
        justify-content: space-between;
        gap: 10px;
        align-items: center;
        padding: 10px 12px;
        margin: 6px 0;
        border-radius: 12px;
        background: rgba(255,255,255,.035);
        border: 1px solid rgba(255,255,255,.06);
    }}

    .ppe-name {{
        color: #dbeafe;
        font-weight: 600;
    }}

    .badge {{
        border-radius: 999px;
        padding: 4px 9px;
        font-size: 11px;
        font-weight: 800;
    }}

    .badge-green {{ background: rgba(16,185,129,.15); color: #6ee7b7; }}
    .badge-red {{ background: rgba(239,68,68,.15); color: #fca5a5; }}
    .badge-yellow {{ background: rgba(245,158,11,.15); color: #fcd34d; }}

    .small {{
        color: #8ea4bf;
        font-size: 12px;
        line-height: 1.6;
    }}

    .metric-card {{
        background: rgba(13,27,45,.72);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 16px;
        padding: 16px;
    }}

    .metric-label {{ color: #8ea4bf; font-size: 12px; }}
    .metric-value {{ color: white; font-size: 26px; font-weight: 800; margin-top: 4px; }}

    .footer-note {{
        color: #7187a3;
        font-size: 11px;
        text-align: center;
        padding: 18px 0 0;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: rgba(255,255,255,.08);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Model
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(LOCAL_MODEL) and os.path.getsize(LOCAL_MODEL) > 1_000_000:
        return YOLO(LOCAL_MODEL), "local"

    weights = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILENAME,
        local_dir=MODEL_DIR,
    )
    return YOLO(weights), "Hugging Face"


with st.spinner("Loading PPE detection model..."):
    try:
        model, model_source = load_model()
    except Exception as exc:
        st.error(
            "The PPE model could not be loaded. Make sure you have internet access "
            "for the first run, or place a compatible `best.pt` file in the `models` folder."
        )
        st.exception(exc)
        st.stop()


# -----------------------------
# PPE video processor
# -----------------------------
class PPEVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.confidence = 0.35
        self.imgsz = 640
        self.save_cooldown = 3.0
        self.last_saved = 0.0
        self.lock = threading.Lock()
        self.last_result = {
            "status": "WAITING",
            "score": 0,
            "fps": 0,
            "detections": 0,
            "violations": [],
            "ppe": {name: "Not detected" for name in REQUIRED_PPE},
        }
        self.events = deque(maxlen=20)
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0

    def _label_for_class(self, class_id):
        return str(self.model.names[class_id])

    def _build_status(self, detected_names):
        ppe_state = {}
        violations = []

        for display_name, (positive, negative) in REQUIRED_PPE.items():
            if negative in detected_names:
                ppe_state[display_name] = "Missing"
                violations.append(VIOLATION_LABELS.get(negative, negative))
            elif positive in detected_names:
                ppe_state[display_name] = "Detected"
            else:
                ppe_state[display_name] = "Not detected"

        # Only call a person fully compliant when all required PPE is actually
        # detected. "Not detected" is kept separate from "Missing" to reduce
        # false violation claims when an item is occluded or too small.
        if any(v == "Missing" for v in ppe_state.values()):
            status = "NOT COMPLIANT"
        elif all(v == "Detected" for v in ppe_state.values()):
            status = "COMPLIANT"
        else:
            status = "CHECKING"

        confirmed = sum(v == "Detected" for v in ppe_state.values())
        score = int((confirmed / len(REQUIRED_PPE)) * 100)

        return status, score, ppe_state, sorted(set(violations))

    def _save_violation(self, frame, violations):
        now = time.time()
        if not violations or now - self.last_saved < self.save_cooldown:
            return

        stamp = datetime.now()
        filename = f"violation_{stamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        filepath = os.path.join(VIOLATION_DIR, filename)

        if cv2.imwrite(filepath, frame):
            self.events.appendleft(
                {
                    "timestamp": stamp.strftime("%d %b %Y, %H:%M:%S"),
                    "type": ", ".join(violations),
                    "path": filepath,
                }
            )
            self.last_saved = now

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1

        results = self.model.predict(
            image,
            conf=self.confidence,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        detected_names = []
        boxes_for_drawing = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self._label_for_class(class_id)
            confidence = float(box.conf[0])

            if confidence < self.confidence:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_names.append(class_name)
            boxes_for_drawing.append(
                (x1, y1, x2, y2, class_name, confidence)
            )

        status, score, ppe_state, violations = self._build_status(
            set(detected_names)
        )

        # Draw detections.
        for x1, y1, x2, y2, class_name, confidence in boxes_for_drawing:
            is_violation = class_name in VIOLATION_LABELS or class_name == "No_Harness"
            box_color = (40, 70, 230) if is_violation else (35, 190, 130)

            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
            text = f"{class_name}  {confidence:.0%}"
            text_y = max(24, y1 - 8)

            cv2.rectangle(
                image,
                (x1, max(0, text_y - 22)),
                (x1 + max(120, len(text) * 9), text_y + 2),
                box_color,
                -1,
            )
            cv2.putText(
                image,
                text,
                (x1 + 5, text_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Header status banner.
        if status == "NOT COMPLIANT":
            banner_color = (40, 60, 220)
        elif status == "COMPLIANT":
            banner_color = (35, 170, 110)
        else:
            banner_color = (30, 145, 210)

        cv2.rectangle(image, (12, 12), (405, 64), banner_color, -1)
        cv2.putText(
            image,
            f"PPE: {status}",
            (26, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if violations:
            self._save_violation(image, violations)

        # FPS calculation.
        self.fps_counter += 1
        elapsed = time.time() - self.last_fps_time
        if elapsed >= 1.0:
            fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.last_fps_time = time.time()
        else:
            fps = self.last_result.get("fps", 0)

        with self.lock:
            self.last_result = {
                "status": status,
                "score": score,
                "fps": round(fps, 1),
                "detections": len(boxes_for_drawing),
                "violations": violations,
                "ppe": ppe_state,
            }

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def get_snapshot(self):
        with self.lock:
            return dict(self.last_result)

    def get_events(self):
        with self.lock:
            return list(self.events)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🦺 PPE Guardian AI")
    st.caption("Real-time workplace safety monitoring")

    st.markdown("### Detection settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.35,
        step=0.05,
        help="Lower values detect more objects but can increase false positives.",
    )
    imgsz = st.select_slider(
        "Inference size",
        options=[416, 512, 640],
        value=640,
        help="416 is faster on CPU. 640 generally gives better small-object detection.",
    )

    st.markdown("### Required PPE")
    for item in REQUIRED_PPE:
        st.checkbox(item, value=True, disabled=True)

    st.divider()
    st.markdown("### Model")
    st.success("PPE-specific model loaded")
    st.caption(f"Source: {model_source}")
    st.caption("Classes: helmet, mask, gloves, goggles, safety vest + violation classes")

    st.divider()
    st.markdown("### Camera tips")
    st.markdown(
        """
        <div class="small">
        • Allow camera permission in your browser.<br>
        • Stand 1–3 metres from the camera.<br>
        • Keep your face and upper body visible.<br>
        • Use good front lighting.<br>
        • Avoid fast movement and heavy occlusion.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Hero
# -----------------------------
st.markdown(
    """
    <section class="hero">
      <div class="hero-copy">
        <span class="eyebrow">AI Workplace Safety • Live Vision</span>
        <h1>PPE Guardian AI</h1>
        <p>
          Real-time computer vision that checks hardhat, mask, gloves,
          goggles and safety-vest compliance directly from your laptop webcam.
        </p>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Live camera + dashboard
# -----------------------------
left, right = st.columns([1.65, 1], gap="large")

with left:
    st.markdown('<div class="section-title">📹 Live Camera Monitor</div>', unsafe_allow_html=True)

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="ppe-live-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
            "audio": False,
        },
        video_processor_factory=PPEVideoProcessor,
        async_processing=True,
    )

    st.markdown(
        '<div class="small">The camera runs in your browser. '
        'This avoids the old <code>cv2.VideoCapture(0)</code> problem where '
        'a deployed Streamlit server tried to open its own camera.</div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="section-title">🛡️ Safety Dashboard</div>', unsafe_allow_html=True)

    status_box = st.empty()
    metric_cols = st.columns(2)
    score_box = metric_cols[0].empty()
    violation_box = metric_cols[1].empty()
    ppe_box = st.empty()

    if ctx.video_processor:
        ctx.video_processor.confidence = confidence
        ctx.video_processor.imgsz = imgsz

    if ctx.state.playing:
        while ctx.state.playing:
            if ctx.video_processor:
                snapshot = ctx.video_processor.get_snapshot()
                events = ctx.video_processor.get_events()

                status = snapshot["status"]
                if status == "COMPLIANT":
                    status_box.markdown(
                        '<div class="status status-ok">✓ PPE COMPLIANT</div>',
                        unsafe_allow_html=True,
                    )
                elif status == "NOT COMPLIANT":
                    status_box.markdown(
                        '<div class="status status-bad">⚠ PPE VIOLATION DETECTED</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    status_box.markdown(
                        '<div class="status status-warn">◌ ANALYZING PPE...</div>',
                        unsafe_allow_html=True,
                    )

                score_box.markdown(
                    f'<div class="metric-card"><div class="metric-label">Compliance score</div>'
                    f'<div class="metric-value">{snapshot["score"]}%</div></div>',
                    unsafe_allow_html=True,
                )
                violation_box.markdown(
                    f'<div class="metric-card"><div class="metric-label">Live detections</div>'
                    f'<div class="metric-value">{snapshot["detections"]}</div></div>',
                    unsafe_allow_html=True,
                )

                html = ""
                for name, state in snapshot["ppe"].items():
                    if state == "Detected":
                        badge = '<span class="badge badge-green">DETECTED</span>'
                    elif state == "Missing":
                        badge = '<span class="badge badge-red">MISSING</span>'
                    else:
                        badge = '<span class="badge badge-yellow">NOT DETECTED</span>'

                    html += (
                        f'<div class="ppe-row"><span class="ppe-name">{name}</span>{badge}</div>'
                    )

                ppe_box.markdown(html, unsafe_allow_html=True)

            time.sleep(0.5)
    else:
        status_box.markdown(
            '<div class="status status-warn">▶ Click START above the camera to begin</div>',
            unsafe_allow_html=True,
        )
        score_box.markdown(
            '<div class="metric-card"><div class="metric-label">Compliance score</div>'
            '<div class="metric-value">—</div></div>',
            unsafe_allow_html=True,
        )
        violation_box.markdown(
            '<div class="metric-card"><div class="metric-label">Live detections</div>'
            '<div class="metric-value">0</div></div>',
            unsafe_allow_html=True,
        )


# -----------------------------
# Event history
# -----------------------------
st.markdown("---")
st.markdown('<div class="section-title">🚨 Recent Safety Events</div>', unsafe_allow_html=True)

if ctx.video_processor:
    events = ctx.video_processor.get_events()
else:
    events = []

if events:
    cols = st.columns(min(3, len(events)))
    for idx, event in enumerate(events[:6]):
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="glass">
                    <div style="font-weight:800;color:#fca5a5;">
                        {event["type"]}
                    </div>
                    <div class="small">{event["timestamp"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.info("No violations have been captured yet.")

st.markdown(
    '<div class="footer-note">PPE Guardian AI is a monitoring aid, not a substitute for qualified safety personnel.</div>',
    unsafe_allow_html=True,
)
