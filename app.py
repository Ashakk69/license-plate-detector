import cv2
import sqlite3
from flask import Flask, render_template, Response, request
import threading
import time
import numpy as np
import re
import os
import requests

# --- AUTO-DOWNLOAD MODEL FILE ---
MODEL_FILE = 'best.pt'
MODEL_URL = 'https://huggingface.co/is-a-dev/yolov8-license-plate-detector/resolve/main/best.pt'

if not os.path.exists(MODEL_FILE):
    print(f"'{MODEL_FILE}' not found. Downloading...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(MODEL_URL, headers=headers, stream=True)
        response.raise_for_status()
        with open(MODEL_FILE, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading the model file: {e}")
        exit()

# --- CONFIGURATION ---
use_yolo = True
use_easyocr = True # Set to True for multi-language support

# --- INITIALIZATION ---
app = Flask(__name__)

# Load detection models
if use_yolo:
    from ultralytics import YOLO
    model = YOLO(MODEL_FILE)
else:
    # Fallback to Haar Cascade if YOLO is disabled
    plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Load OCR models
if use_easyocr:
    import easyocr
    # Add language codes for the regions you want to support
    print("Loading EasyOCR models... (this may take a moment)")
    reader = easyocr.Reader(['en', 'ru']) # English and Russian
    print("EasyOCR loaded.")
else:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('vehicles.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS entries (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 plate TEXT,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                 )''')
    conn.commit()
    conn.close()

init_db()

# --- SHARED VARIABLES & BACKGROUND THREAD ---
plate_to_log = None
last_logged_plate = None
last_log_time = 0
lock = threading.Lock()

def log_plate_worker():
    global plate_to_log, last_logged_plate, last_log_time
    while True:
        time.sleep(0.1)
        with lock:
            plate = plate_to_log
            plate_to_log = None
        if plate:
            current_time = time.time()
            if plate != last_logged_plate or (current_time - last_log_time) > 5:
                conn = sqlite3.connect('vehicles.db', check_same_thread=False)
                c = conn.cursor()
                c.execute('INSERT INTO entries (plate) VALUES (?)', (plate,))
                conn.commit()
                conn.close()
                last_logged_plate = plate
                last_log_time = current_time
                print(f"✅ Logged plate: {plate}")

threading.Thread(target=log_plate_worker, daemon=True).start()

# --- CAMERA SETUP ---
camera_sources = {
    "0": {"name": "Default Webcam", "source": 0},
    "1": {"name": "Secondary Webcam", "source": 1},
    "2": {"name": "Phone IP Camera", "source": "http://192.168.0.101:8080/video"}
}
current_source_key = "0"
cap = cv2.VideoCapture(camera_sources[current_source_key]["source"])

# --- HELPER FUNCTIONS ---
def correct_perspective(frame, box_coords):
    x, y, w, h = box_coords
    src_pts = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    width, height = 300, 100
    dst_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(frame, matrix, (width, height))
    return result

def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

# --- MAIN VIDEO PROCESSING FUNCTION ---
def generate_frames():
    global plate_to_log
    global cap 
    
    while True:
        success, frame = cap.read()
        if not success:
            print(f"⚠️ Failed to read frame from camera {current_source_key}. Reconnecting...")
            time.sleep(1)
            if cap: cap.release()
            cap = cv2.VideoCapture(camera_sources[current_source_key]["source"])
            continue

        frame_copy = frame.copy()
        frame = cv2.resize(frame, (640, 480))
        
        detected_plates = []
        if use_yolo:
            results = model(frame)[0]
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, _ = result
                if score > 0.5:
                    detected_plates.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
        else: # Fallback to Haar
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plates = plate_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in plates:
                detected_plates.append((x, y, w, h))

        for (x, y, w, h) in detected_plates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            original_h, original_w, _ = frame_copy.shape
            scale_x, scale_y = original_w / 640, original_h / 480
            original_box = (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))
            straightened_plate = correct_perspective(frame_copy, original_box)
            processed_plate = preprocess_for_ocr(straightened_plate)
            
            plate_text = ""
            try:
                if use_easyocr:
                    result = reader.readtext(processed_plate, detail=0, paragraph=False)
                    if result:
                        plate_text = "".join(result).strip().upper().replace(" ", "")
                else: # Tesseract
                    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    plate_text = pytesseract.image_to_string(processed_plate, config=custom_config).strip().upper().replace(" ", "").replace("\n", "")
            except Exception as e:
                print(f"OCR Error: {e}")
                continue

            # Universal Filter Logic
            is_good_length = 4 <= len(plate_text) <= 10
            has_letter = any(c.isalpha() for c in plate_text)
            has_digit = any(c.isdigit() for c in plate_text)

            if is_good_length and has_letter and has_digit:
                cv2.putText(frame, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                with lock:
                    plate_to_log = plate_text
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html', cameras=camera_sources, current_camera=current_source_key)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/switch_camera', methods=['POST'])
def switch_camera():
    global cap, current_source_key
    new_source_key = request.form.get('source')
    if new_source_key in camera_sources:
        current_source_key = new_source_key
        if cap.isOpened(): cap.release()
        cap = cv2.VideoCapture(camera_sources[new_source_key]["source"])
        print(f"Switched to camera: {camera_sources[new_source_key]['name']}")
        return "OK", 200
    return "Camera not found", 404

@app.route('/logs')
def logs():
    conn = sqlite3.connect('vehicles.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('SELECT plate, timestamp FROM entries ORDER BY timestamp DESC LIMIT 100')
    entries = c.fetchall()
    conn.close()
    return render_template('logs.html', entries=entries)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)