import os
import bz2
import requests
import cv2
import imutils
import numpy as np
from flask import Flask, render_template, Response
from dds_core.detection import DrowsinessDetector

# =========================================
# 🔧 AUTO-DOWNLOAD MODEL FILE (if missing)
# =========================================
MODEL_DIR = "Files"
MODEL_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("🔽 Downloading dlib facial landmark model (~100MB compressed)...")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    r = requests.get(url, stream=True)
    with open(os.path.join(MODEL_DIR, "temp.dat.bz2"), "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("🧩 Decompressing model...")
    with bz2.BZ2File(os.path.join(MODEL_DIR, "temp.dat.bz2")) as fr, open(MODEL_PATH, "wb") as fw:
        fw.write(fr.read())
    os.remove(os.path.join(MODEL_DIR, "temp.dat.bz2"))
    print("✅ Model ready at:", MODEL_PATH)

# =========================================
# 🚗 FLASK APP INITIALIZATION
# =========================================
app = Flask(__name__)

# Initialize the detector
detector = DrowsinessDetector(MODEL_PATH)

# =========================================
# 🧠 VIDEO FEED HANDLER
# =========================================
def gen_frames():
    cap = cv2.VideoCapture(0)  # Use default webcam
    if not cap.isOpened():
        raise RuntimeError("⚠️ Could not access webcam. Check permissions.")

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Resize and process the frame
            frame = imutils.resize(frame, width=640)
            vis, state = detector.process_frame(frame)

            # Encode the processed frame to send to frontend
            _, buffer = cv2.imencode('.jpg', vis)
            frame = buffer.tobytes()

            # Stream the frame to the browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# =========================================
# 🌐 FLASK ROUTES
# =========================================
@app.route('/')
def index():
    return render_template('index.html')  # HTML page with video element

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# =========================================
# 🚀 RUN LOCALLY OR ON RENDER
# =========================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
