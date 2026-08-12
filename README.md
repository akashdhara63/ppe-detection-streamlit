# 🦺 PPE Guardian AI — Streamlit Webcam PPE Detection

A polished Streamlit PPE monitoring application with **real browser webcam support** and a PPE-specific YOLOv8 model.

## What was fixed

The original project had two important problems:

1. It loaded `yolov8n.pt`, which is a general COCO model and does **not** know classes such as `Hardhat`, `Mask`, `Gloves`, or `Safety Vest`.
2. It used `cv2.VideoCapture(0)`. In a Streamlit deployment, that opens a camera on the machine running the Python process (the server), not the user's laptop.

This version uses:

- `streamlit-webrtc` for the **laptop/browser webcam**
- A PPE-specific YOLOv8 model with 13 PPE/safety classes
- Live bounding boxes
- Compliance score
- Missing-PPE warnings
- Violation snapshots
- A redesigned dashboard
- Your supplied city/PPE illustration as the hero image

## PPE classes

The included model supports:

- Hardhat / NO-Hardhat
- Mask / NO-Mask
- Gloves / NO-Gloves
- Goggles / NO-Goggles
- Safety Vest / NO-Safety Vest
- Person
- No_Harness
- Fall-Detected

The model is SafetyVision YOLOv8 v2 by Ayush Gupta and is downloaded automatically from Hugging Face on first run if `models/best.pt` is not already present.

## Run locally

### 1. Open the project folder

```bash
cd ppe_detection_streamlit
```

### 2. Create a virtual environment

Recommended: Python 3.12.

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

If PowerShell blocks activation, use:

```bash
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m streamlit run app.py
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the application

```bash
python -m streamlit run app.py
```

Open the local URL shown by Streamlit, normally:

```text
http://localhost:8501
```

### 5. Allow the camera

When you click **START** in the webcam component:

1. Browser asks for camera permission.
2. Click **Allow**.
3. Your laptop webcam starts inside the page.
4. The frames are sent through WebRTC to the Streamlit app for inference.

## Streamlit Cloud / GitHub deployment

Upload the complete project to GitHub, including:

- `app.py`
- `requirements.txt`
- `assets/hero.png`
- `models/` if you have supplied weights
- `.streamlit/config.toml`
- `README.md`

Then create a Streamlit Cloud app and select:

```text
app.py
```

For a deployed webcam application, use the HTTPS Streamlit URL and allow browser camera permission.

> Important: `cv2.VideoCapture(0)` is intentionally not used. Browser WebRTC is required for a remote Streamlit app to access the user's camera.

## If the model does not download

The first run needs internet access. You can manually place a compatible PPE YOLO `.pt` model at:

```text
models/best.pt
```

The app will use that local file instead of downloading the model.

## Performance

For a CPU laptop:

- Start with inference size `416`
- Use confidence around `0.35–0.45`
- Keep one person reasonably close to the camera
- Use good lighting

For better accuracy, use `640` inference size.

## Important limitation

This is a computer-vision assistance/demo system. A detection of "Not detected" is deliberately different from "Missing" to reduce false violation claims when PPE is occluded or too small. Safety decisions should be verified by a qualified human.

## Model attribution

SafetyVision YOLOv8 v2:
- Author: Ayush Gupta
- Repository/model: `ayushgupta7777/safetyvision-yolov8`
- License: AGPL-3.0

Ultralytics YOLOv8 is also AGPL-3.0. Review the applicable licenses before commercial redistribution.

