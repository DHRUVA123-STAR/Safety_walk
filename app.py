from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from datetime import datetime, timedelta
import math
import cv2
import numpy as np
import base64
import os
import json
import urllib.parse
import urllib.request
import threading
import ssl
import certifi
import uuid
import smtplib
from email.message import EmailMessage
from werkzeug.utils import secure_filename
import cloudinary
import cloudinary.uploader
import firebase_admin
from firebase_admin import credentials, firestore, messaging

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

app = Flask(__name__)

CITY_LAT = 16.5062
CITY_LON = 80.6480
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIRESTORE_TIMEOUT_SEC = float(os.getenv("FIRESTORE_TIMEOUT_SEC", "8"))
ANALYZE_MAX_DIM = int(os.getenv("ANALYZE_MAX_DIM", "640"))

# Lightweight .env loader
def load_local_env():
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

load_local_env()

# Feedback Gmail account
# For Railway deploys, set these in Railway Variables.
FEEDBACK_GMAIL_ADDRESS = os.getenv("FEEDBACK_GMAIL_ADDRESS", "").strip()
FEEDBACK_GMAIL_APP_PASSWORD = os.getenv("FEEDBACK_GMAIL_APP_PASSWORD", "")

def get_feedback_email_config_errors():
    missing = []
    if not FEEDBACK_GMAIL_ADDRESS:
        missing.append("FEEDBACK_GMAIL_ADDRESS")
    if not FEEDBACK_GMAIL_APP_PASSWORD:
        missing.append("FEEDBACK_GMAIL_APP_PASSWORD")
    return missing

def send_feedback_email(feedback_record):
    missing = get_feedback_email_config_errors()
    if missing:
        raise RuntimeError(
            "Feedback email is not configured. Missing Railway variables: " + ", ".join(missing)
        )

    message = EmailMessage()
    message["Subject"] = (
        f"Community Safety App Feedback - {feedback_record['rating']}/5"
    )
    message["From"] = FEEDBACK_GMAIL_ADDRESS
    message["To"] = FEEDBACK_GMAIL_ADDRESS
    message["Reply-To"] = FEEDBACK_GMAIL_ADDRESS

    submitted_at = feedback_record["created_at"]
    comment = feedback_record["comment"] or "(No comment provided)"
    remote_ip = feedback_record.get("remote_ip") or "Unknown"
    user_agent = feedback_record.get("user_agent") or "Unknown"
    message.set_content(
        "\n".join(
            [
                "A new feedback submission was received.",
                "",
                f"Rating: {feedback_record['rating']}/5",
                f"Comment: {comment}",
                f"User Token: {feedback_record['user_token']}",
                f"Submitted At (UTC): {submitted_at}",
                f"Remote IP: {remote_ip}",
                f"User Agent: {user_agent}",
            ]
        )
    )

    ssl_context = ssl.create_default_context(cafile=certifi.where())

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as smtp:
        smtp.ehlo()
        smtp.starttls(context=ssl_context)
        smtp.ehlo()
        smtp.login(FEEDBACK_GMAIL_ADDRESS, FEEDBACK_GMAIL_APP_PASSWORD)
        smtp.send_message(message)

MODEL_DIR = os.path.join(BASE_DIR, "models")
PROTO_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
CAFFE_MODEL_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")
PROTO_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt"
CAFFE_MODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"

# Cloudinary Config
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Firebase Config
firebase_creds_env = os.getenv("FIREBASE_CREDENTIALS")
if firebase_creds_env:
    # Production: Read from Railway Environment Variable
    cred_dict = json.loads(firebase_creds_env)
    cred = credentials.Certificate(cred_dict)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
else:
    # Local Development: Read from physical file
    cred_path = os.path.join(BASE_DIR, "firebase-adminsdk.json")
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
    else:
        db = None
        print("WARNING: firebase-adminsdk.json not found and FIREBASE_CREDENTIALS not set!")

MOBILENET_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
VEHICLE_CLASSES = {"bicycle", "bus", "car", "motorbike", "train"}

_vehicle_net = None
_vehicle_net_lock = threading.Lock()
_hog_detector = None
_yolo_model = None
_yolo_model_lock = threading.Lock()
_yolo_disabled = False
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolov8n.pt")

def log_service_status():
    print(f"TOMTOM_API_KEY set: {bool(os.getenv('TOMTOM_API_KEY'))}")
    print(f"YOLO available: {YOLO is not None} ({YOLO_MODEL_NAME})")
    model_candidate_paths = []
    if os.path.isabs(YOLO_MODEL_NAME):
        model_candidate_paths.append(YOLO_MODEL_NAME)
    else:
        model_candidate_paths.append(os.path.join(BASE_DIR, YOLO_MODEL_NAME))
        model_candidate_paths.append(os.path.abspath(YOLO_MODEL_NAME))

    local_model_path = next((p for p in model_candidate_paths if os.path.exists(p)), None)
    if local_model_path:
        print(f"YOLO model status: local file found at {local_model_path}")
    else:
        print(
            "YOLO model status: local file not found. "
            "Ultralytics will auto-download model on first detection request."
        )

def is_urban_area(lat, lon):
    distance = math.sqrt((lat - CITY_LAT) ** 2 + (lon - CITY_LON) ** 2)
    return distance < 0.1

def calculate_score(area, lighting, traffic, crowd, vehicle_count=0):
    score = 100
    reasons = []

    if area == "Urban":
        score -= 15
        reasons.append("Urban area has higher movement density.")
    else:
        score -= 5
        reasons.append("Rural area has lower movement density.")

    if lighting == "Poor":
        score -= 20
        reasons.append("Poor lighting reduces visibility.")

    if traffic == "High":
        score -= 20
        reasons.append("High traffic increases accident risk.")

    if crowd == "Dense":
        score -= 15
        reasons.append("Dense crowd may cause congestion.")

    if vehicle_count >= 3:
        score -= 10
        reasons.append("High vehicle count detected by camera AI.")

    current_hour = datetime.now().hour
    if current_hour < 6 or current_hour > 18:
        score -= 10
        reasons.append("Night time reduces safety.")

    score = max(0, min(score, 100))

    if score >= 75:
        level = "Safe"
    elif score >= 50:
        level = "Moderate"
    else:
        level = "Risky"

    return score, level, reasons

def _download_file(url, output_path):
    urllib.request.urlretrieve(url, output_path)

def ensure_vehicle_model_files():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(PROTO_PATH):
        _download_file(PROTO_URL, PROTO_PATH)
    if not os.path.exists(CAFFE_MODEL_PATH):
        _download_file(CAFFE_MODEL_URL, CAFFE_MODEL_PATH)

def get_vehicle_detector():
    global _vehicle_net
    if _vehicle_net is not None:
        return _vehicle_net

    with _vehicle_net_lock:
        if _vehicle_net is None:
            ensure_vehicle_model_files()
            _vehicle_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, CAFFE_MODEL_PATH)
    return _vehicle_net

def get_hog_people_detector():
    global _hog_detector
    if _hog_detector is None:
        _hog_detector = cv2.HOGDescriptor()
        _hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _hog_detector

def detect_lighting(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    lighting = "Good" if brightness > 100 else "Poor"
    return lighting, round(brightness, 2)

def detect_vehicles_dnn(img, confidence_threshold=0.45):
    try:
        net = get_vehicle_detector()
        resized = cv2.resize(img, (300, 300))
        blob = cv2.dnn.blobFromImage(resized, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
    except Exception:
        # Fallback: if model download/load fails on hosting, do not break full scene analysis.
        return 0, {}

    counts = {}
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < confidence_threshold:
            continue

        class_id = int(detections[0, 0, i, 1])
        if class_id < 0 or class_id >= len(MOBILENET_CLASSES):
            continue

        class_name = MOBILENET_CLASSES[class_id]
        if class_name in VEHICLE_CLASSES:
            counts[class_name] = counts.get(class_name, 0) + 1

    total_vehicles = sum(counts.values())
    return total_vehicles, counts

def detect_crowd_opencv(img):
    hog = get_hog_people_detector()
    boxes, _ = hog.detectMultiScale(
        img,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )
    people_count = len(boxes)
    crowd_label = "Dense" if people_count >= 4 else "Less"
    return people_count, crowd_label

def merge_people_counts(image_bgr, people_count, mode="autosync"):
    faces = detect_faces_haar(image_bgr)
    face_count = len(faces)
    
    # Always take the highest count between YOLO/HOG full-body detection and Haar face detection
    merged_people_count = max(people_count, face_count)

    dense_threshold = 3 if mode in {"crowd", "mobile_detect"} else 4
    crowd_label = "Dense" if merged_people_count >= dense_threshold else "Less"
    return merged_people_count, crowd_label, face_count

def get_yolo_detector():
    global _yolo_model, _yolo_disabled
    if _yolo_model is not None:
        return _yolo_model
    if _yolo_disabled:
        raise RuntimeError("YOLO disabled after previous load failure")
    if YOLO is None:
        raise RuntimeError("ultralytics not installed")
    with _yolo_model_lock:
        if _yolo_model is None:
            try:
                _yolo_model = YOLO(YOLO_MODEL_NAME)
            except Exception:
                _yolo_disabled = True
                raise
    return _yolo_model

def preprocess_for_detection(img):
    # Fast path for bright scenes to reduce latency on low-CPU cloud instances.
    base_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base_brightness = float(np.mean(base_gray))
    if base_brightness >= 95:
        return img

    # CLAHE on luminance channel for better shadow detail.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))

    # Mild gamma boost for dark scenes.
    gamma = 0.95 if brightness >= 95 else (0.85 if brightness >= 55 else 0.72)
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    out = cv2.LUT(out, lut)

    # Light sharpening to improve small-object edges.
    kernel = np.array([[0, -1, 0], [-1, 5.15, -1], [0, -1, 0]], dtype=np.float32)
    out = cv2.filter2D(out, -1, kernel)

    return out

def resize_for_analysis(img, max_dim=ANALYZE_MAX_DIM):
    height, width = img.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_dim:
        return img

    scale = max_dim / float(longest_side)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    return cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

def yolo_conf_threshold(brightness):
    if brightness < 40:
        return 0.25
    if brightness < 75:
        return 0.30
    return 0.35

def detect_scene_yolo(img, brightness, allow_fast_empty=True, enable_full_pass=True):
    detector = get_yolo_detector()
    conf = yolo_conf_threshold(brightness)
    # COCO ids: person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7
    target_classes = [0, 1, 2, 3, 5, 7]
    class_map = {
        "car": "car",
        "bus": "bus",
        "truck": "truck",
        "bicycle": "bicycle",
        "motorcycle": "motorbike",
        "motorbike": "motorbike",
        "person": "person"
    }

    def parse_results(results):
        vehicles_by_type = {}
        people_count = 0
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue
            for c in boxes.cls:
                cls_id = int(c.item())
                cls_name = names.get(cls_id, "").lower()
                mapped = class_map.get(cls_name)
                if mapped == "person":
                    people_count += 1
                elif mapped:
                    vehicles_by_type[mapped] = vehicles_by_type.get(mapped, 0) + 1
        vehicle_count = sum(vehicles_by_type.values())
        crowd = "Dense" if people_count >= 4 else "Less"
        return vehicle_count, vehicles_by_type, people_count, crowd

    # Quick pass for empty-scene fast return.
    quick_conf = min(0.55, max(0.35, conf + 0.10))
    quick_results = detector.predict(
        source=img,
        conf=quick_conf,
        iou=0.5,
        verbose=False,
        imgsz=256,
        classes=target_classes,
        max_det=12
    )
    quick_vehicle_count, quick_vehicles_by_type, quick_people_count, quick_crowd = parse_results(quick_results)
    if allow_fast_empty and quick_vehicle_count == 0 and quick_people_count == 0:
        return quick_vehicle_count, quick_vehicles_by_type, quick_people_count, quick_crowd, "yolov8n-fast-empty"
    if not enable_full_pass:
        return quick_vehicle_count, quick_vehicles_by_type, quick_people_count, quick_crowd, "yolov8n-quick"

    # Full pass only when quick pass detects likely objects.
    full_results = detector.predict(
        source=img,
        conf=conf,
        iou=0.5,
        verbose=False,
        imgsz=384,
        classes=target_classes,
        max_det=32
    )
    vehicle_count, vehicles_by_type, people_count, crowd = parse_results(full_results)
    return vehicle_count, vehicles_by_type, people_count, crowd, "yolov8n-full"

def merge_traffic_signals(camera_traffic, traffic_api):
    if not traffic_api.get("available"):
        return camera_traffic, "camera_only"

    tomtom_traffic = traffic_api.get("traffic") or "Low"
    speed_ratio = float(traffic_api.get("speed_ratio") or 1.0)

    # Conservative merge: raise risk if either source indicates pressure.
    merged = "High" if (camera_traffic == "High" or tomtom_traffic == "High" or speed_ratio < 0.75) else "Low"
    source = "merged(camera+tomtom)"
    return merged, source

def estimate_traffic_from_vehicles(vehicle_count):
    return "High" if vehicle_count >= 3 else "Low"

def fetch_external_traffic(lat, lon, timeout_sec=8):
    api_key = os.getenv("TOMTOM_API_KEY")
    if not api_key:
        return {
            "available": False,
            "provider": "TomTom",
            "traffic": None,
            "message": "TOMTOM_API_KEY is not set."
        }

    endpoint = (
        "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?"
        + urllib.parse.urlencode({"point": f"{lat},{lon}", "key": api_key})
    )

    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(endpoint, timeout=timeout_sec, context=ssl_context) as response:
            payload = json.loads(response.read().decode("utf-8"))

        flow_data = payload.get("flowSegmentData", {})
        current_speed = float(flow_data.get("currentSpeed", 0))
        free_flow_speed = float(flow_data.get("freeFlowSpeed", 0))

        ratio = (current_speed / free_flow_speed) if free_flow_speed > 0 else 0
        traffic = "High" if (ratio < 0.6 or current_speed < 20) else "Low"

        return {
            "available": True,
            "provider": "TomTom",
            "traffic": traffic,
            "current_speed": round(current_speed, 2),
            "free_flow_speed": round(free_flow_speed, 2),
            "speed_ratio": round(ratio, 2),
            "message": "Traffic fetched from TomTom flow API."
        }
    except Exception as exc:
        allow_insecure = os.getenv("ALLOW_INSECURE_SSL_FOR_DEV", "false").lower() == "true"
        if allow_insecure:
            try:
                insecure_context = ssl._create_unverified_context()
                with urllib.request.urlopen(endpoint, timeout=timeout_sec, context=insecure_context) as response:
                    payload = json.loads(response.read().decode("utf-8"))

                flow_data = payload.get("flowSegmentData", {})
                current_speed = float(flow_data.get("currentSpeed", 0))
                free_flow_speed = float(flow_data.get("freeFlowSpeed", 0))
                ratio = (current_speed / free_flow_speed) if free_flow_speed > 0 else 0
                traffic = "High" if (ratio < 0.6 or current_speed < 20) else "Low"

                return {
                    "available": True,
                    "provider": "TomTom",
                    "traffic": traffic,
                    "current_speed": round(current_speed, 2),
                    "free_flow_speed": round(free_flow_speed, 2),
                    "speed_ratio": round(ratio, 2),
                    "message": "Traffic fetched from TomTom flow API with insecure SSL fallback."
                }
            except Exception as insecure_exc:
                return {
                    "available": False,
                    "provider": "TomTom",
                    "traffic": None,
                    "message": f"Traffic API SSL/Network error: {str(insecure_exc)}"
                }

        return {
            "available": False,
            "provider": "TomTom",
            "traffic": None,
            "message": f"Traffic API SSL/Network error: {str(exc)}. Set ALLOW_INSECURE_SSL_FOR_DEV=true for local testing."
        }

def decode_data_url_image(image_data_url):
    if not image_data_url or "," not in image_data_url:
        raise ValueError("Invalid image payload.")
    image_bytes = base64.b64decode(image_data_url.split(",", 1)[1])
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image.")
    return image

def detect_faces_haar(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        return []
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(24, 24)
    )
    return faces

def detect_document_like_image(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = gray.shape[:2]
    img_area = float(img_h * img_w)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < img_area * 0.18:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        rect_area_ratio = (w * h) / img_area
        if rect_area_ratio < 0.22:
            continue

        aspect = w / max(h, 1)
        if 0.62 <= aspect <= 1.9:
            return True
    return False

def estimate_text_block_ratio(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)
    _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
    merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    merged = cv2.medianBlur(merged, 3)
    return float(np.mean(merged > 0))

def detect_screen_like_capture(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    img_area = float(img_h * img_w)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect_ratio = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < img_area * 0.30:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * peri, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        rect_ratio = (w * h) / max(img_area, 1.0)
        aspect = w / max(h, 1)
        if rect_ratio > best_rect_ratio and 0.5 <= aspect <= 2.2:
            best_rect_ratio = rect_ratio

    text_ratio = estimate_text_block_ratio(image_bgr)
    return best_rect_ratio >= 0.35 and text_ratio >= 0.05

def detect_large_centered_rect(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    img_area = float(img_h * img_w)
    img_center_x = img_w / 2.0
    img_center_y = img_h / 2.0
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < img_area * 0.20:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        rect_ratio = (w * h) / max(img_area, 1.0)
        if rect_ratio < 0.26:
            continue

        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        if abs(cx - img_center_x) <= img_w * 0.18 and abs(cy - img_center_y) <= img_h * 0.18:
            aspect = w / max(h, 1)
            if 0.5 <= aspect <= 2.1:
                return True
    return False

def detect_portrait_like_image(image_bgr, faces):
    if faces is None or len(faces) == 0:
        return False

    img_h, img_w = image_bgr.shape[:2]
    img_area = float(img_h * img_w)
    largest_face_area = max((w * h) for (_, _, w, h) in faces)
    face_ratio = largest_face_area / max(img_area, 1.0)
    return face_ratio >= 0.08

def road_scene_score(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_half = hsv[hsv.shape[0] // 2 :, :]
    if lower_half.size == 0:
        return 0.0

    saturation = lower_half[:, :, 1]
    value = lower_half[:, :, 2]
    grayish_ratio = float(np.mean(saturation < 65))
    visible_ratio = float(np.mean(value > 45))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lower_gray = gray[gray.shape[0] // 2 :, :]
    edges = cv2.Canny(lower_gray, 70, 180)
    edge_ratio = float(np.mean(edges > 0))

    return grayish_ratio * 0.45 + visible_ratio * 0.25 + min(edge_ratio / 0.18, 1.0) * 0.30

def outdoor_scene_score(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    blue_sky_ratio = float(np.mean(((h >= 90) & (h <= 130) & (s >= 35) & (v >= 70))))
    green_ratio = float(np.mean(((h >= 35) & (h <= 95) & (s >= 35) & (v >= 45))))
    bright_ratio = float(np.mean(v >= 85))
    return blue_sky_ratio * 0.35 + green_ratio * 0.25 + bright_ratio * 0.40

def water_scene_score(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_half = hsv[hsv.shape[0] // 2 :, :]
    if lower_half.size == 0:
        return 0.0

    h = lower_half[:, :, 0]
    s = lower_half[:, :, 1]
    v = lower_half[:, :, 2]
    reflective_ratio = float(np.mean((s < 70) & (v > 80)))
    muddy_ratio = float(np.mean(((h >= 8) & (h <= 28) & (s >= 35) & (v >= 35))))
    dark_pool_ratio = float(np.mean((v >= 35) & (v <= 120) & (s < 90)))
    return reflective_ratio * 0.45 + muddy_ratio * 0.30 + dark_pool_ratio * 0.25

def damaged_road_visual_score(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lower_gray = gray[gray.shape[0] // 2 :, :]
    if lower_gray.size == 0:
        return 0.0

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_hsv = hsv[hsv.shape[0] // 2 :, :]
    saturation = lower_hsv[:, :, 1]
    value = lower_hsv[:, :, 2]

    edges = cv2.Canny(lower_gray, 45, 150)
    edge_ratio = float(np.mean(edges > 0))

    # Dark, low-saturation regions often correlate with cracks, potholes,
    # and broken asphalt patches on road-surface photos.
    damage_ratio = float(np.mean((value < 125) & (saturation < 110)))
    texture_strength = min(float(np.std(lower_gray)) / 52.0, 1.0)

    return (
        min(edge_ratio / 0.12, 1.0) * 0.42
        + min(damage_ratio / 0.38, 1.0) * 0.38
        + texture_strength * 0.20
    )

def verify_alert_content(image_bgr, tag):
    processed = preprocess_for_detection(image_bgr)
    lighting, brightness = detect_lighting(processed)
    tag_lower = tag.lower()

    if brightness < 10:
        return False, "Image is too dark to verify."

    try:
        vehicle_count, _, people_count, _, _ = detect_scene_yolo(processed, brightness, allow_fast_empty=False)
    except Exception:
        vehicle_count, _ = detect_vehicles_dnn(processed)
        people_count, _ = detect_crowd_opencv(processed)

    faces = detect_faces_haar(processed)
    text_ratio = estimate_text_block_ratio(processed)
    centered_rect = detect_large_centered_rect(processed)
    document_like = detect_document_like_image(processed)
    screen_like = detect_screen_like_capture(processed)
    road_score = road_scene_score(processed)
    outdoor_score = outdoor_scene_score(processed)
    water_score = water_scene_score(processed)
    road_damage_score = damaged_road_visual_score(processed)
    strong_damaged_road_candidate = (
        "damaged road" in tag_lower
        and road_score >= 0.44
        and outdoor_score >= 0.02
        and road_damage_score >= 0.38
    )

    if document_like:
        return False, "This looks like a document or ID card. Please upload a real road, traffic, crowd, or hazard photo."
    if screen_like and not strong_damaged_road_candidate:
        return False, "This looks like a screenshot or screen photo. Please upload a real outdoor alert photo."
    if centered_rect and text_ratio >= 0.035 and not strong_damaged_road_candidate:
        return False, "This looks like a close-up card or document photo, not a community alert."
    if detect_portrait_like_image(processed, faces) and vehicle_count == 0 and people_count < 3:
        return False, "This looks like a portrait or passport-style photo, not a community alert."

    has_scene_evidence = vehicle_count > 0 or people_count > 0 or road_score >= 0.42 or outdoor_score >= 0.18
    if not has_scene_evidence:
        return False, "This image does not look like a real outdoor road, traffic, crowd, or hazard scene."

    traffic_match = vehicle_count >= 2 and outdoor_score >= 0.08 and text_ratio < 0.05 and not centered_rect
    crowd_match = people_count >= 3 and outdoor_score >= 0.08 and text_ratio < 0.05 and not centered_rect
    damaged_road_match = (
        road_score >= 0.42
        and outdoor_score >= 0.02
        and water_score < 0.48
        and text_ratio < 0.09
        and (not centered_rect or strong_damaged_road_candidate)
        and vehicle_count <= 2
        and people_count <= 2
        and (
            road_damage_score >= 0.36
            or (road_score >= 0.54 and outdoor_score >= 0.06)
        )
    )
    water_logging_match = (
        road_score >= 0.44
        and outdoor_score >= 0.12
        and water_score >= 0.34
        and text_ratio < 0.045
        and not centered_rect
        and people_count <= 2
    )
    other_match = (
        road_score >= 0.50
        and outdoor_score >= 0.16
        and text_ratio < 0.04
        and not centered_rect
        and faces is not None
        and len(faces) == 0
    )

    if "traffic" in tag_lower and not traffic_match:
        return False, f"This image does not match High Traffic. Detected vehicles: {vehicle_count}. Please upload a busier road scene."
    elif "crowd" in tag_lower and not crowd_match:
        return False, f"This image does not match Crowd. Detected people: {people_count}. Please upload a clearer crowd scene."
    elif "damaged road" in tag_lower and not damaged_road_match:
        if document_like or (centered_rect and text_ratio >= 0.06) or (screen_like and text_ratio >= 0.12):
            return False, "ID card or document detected. Please upload a real road photo."
        return False, "Damaged road not detected clearly. Please upload a clearer road photo."
    elif "water logging" in tag_lower and not water_logging_match:
        return False, "This image does not look like water logging on a road. Please upload a clearer outdoor water-logging photo."
    elif "other" in tag_lower and not other_match:
        return False, "This image does not match a valid road or hazard alert for this app."

    return True, "Verified"

def save_uploaded_community_image(image_data_url):
    # Upload directly to Cloudinary
    response = cloudinary.uploader.upload(image_data_url, folder="safety_app_alerts")
    return response.get("secure_url")

def delete_cloudinary_image(image_url):
    if not image_url or "cloudinary.com" not in image_url or "/upload/" not in image_url:
        return
    try:
        after_upload = image_url.split("/upload/")[1]
        if after_upload.startswith("v") and "/" in after_upload:
            after_upload = after_upload.split("/", 1)[1]
        public_id = after_upload.rsplit(".", 1)[0]
        cloudinary.uploader.destroy(public_id)
    except Exception as e:
        print(f"Failed to delete from Cloudinary: {e}")

def cleanup_old_alerts():
    """Deletes posts and images older than 7 days to keep the feed relevant."""
    if not db:
        return
    try:
        seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        # Query posts older than 7 days
        old_posts = db.collection('community_posts').where('created_at', '<', seven_days_ago).stream(timeout=FIRESTORE_TIMEOUT_SEC)
        
        for doc in old_posts:
            data = doc.to_dict()
            image_url = data.get("image_url", "")
            
            # Delete physical image from Cloudinary
            delete_cloudinary_image(image_url)
            
            # Delete record from Firebase Firestore
            doc.reference.delete()
    except Exception as e:
        print(f"Cleanup error: {e}")

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

def fetch_posts_with_reactions(timeout_sec=None):
    if not db:
        return []
    timeout_sec = timeout_sec or FIRESTORE_TIMEOUT_SEC
    try:
        docs = db.collection('community_posts').order_by(
            'created_at',
            direction=firestore.Query.DESCENDING
        ).limit(50).stream(timeout=timeout_sec)
        posts = []
        for doc in docs:
            data = doc.to_dict()
            reactions_data = data.get('reactions', {})
            avoid_count = sum(1 for r in reactions_data.values() if r == 'Avoid')
            careful_count = sum(1 for r in reactions_data.values() if r == 'Careful')
            safe_now_count = sum(1 for r in reactions_data.values() if r == 'Safe now')
            
            posts.append({
                "id": doc.id,
                "image_url": data.get("image_url"),
                "lat": float(data.get("lat", 0)),
                "lon": float(data.get("lon", 0)),
                "tag": data.get("tag"),
                "note": data.get("note", ""),
                "created_at": data.get("created_at"),
                "reactions": {
                    "Avoid": avoid_count,
                    "Careful": careful_count,
                    "Safe now": safe_now_count
                }
            })
        return posts
    except Exception as e:
        print(f"Community posts fetch error: {e}")
        return []

def count_fcm_users(max_docs=1000, timeout_sec=None):
    if not db:
        return 0
    timeout_sec = timeout_sec or FIRESTORE_TIMEOUT_SEC
    try:
        docs = db.collection("fcm_tokens").limit(max_docs + 1).stream(timeout=timeout_sec)
        count = 0
        for _ in docs:
            count += 1
            if count > max_docs:
                return f"{max_docs}+"
        return count
    except Exception as e:
        print(f"FCM token count error: {e}")
        return 0

def compute_community_penalty(lat, lon):
    if lat is None or lon is None:
        return 0, []

    now = datetime.utcnow()
    posts = fetch_posts_with_reactions()
    nearby = []
    score = 0.0

    for post in posts:
        distance = haversine_km(lat, lon, float(post["lat"]), float(post["lon"]))
        if distance > 2.0:
            continue

        created = datetime.fromisoformat(post["created_at"])
        age_hours = max(0.0, (now - created).total_seconds() / 3600.0)
        recency_factor = max(0.2, 1.0 - (age_hours / 168.0))
        distance_factor = 1.0 if distance <= 0.5 else 0.5

        reactions = post["reactions"]
        reaction_weight = (
            reactions["Avoid"] * 2.0
            + reactions["Careful"] * 1.0
            - reactions["Safe now"] * 1.2
        )

        base = 1.2 if post["tag"] in {"Damaged Road", "Water Logging"} else 1.0
        post_risk = max(0.0, (base + reaction_weight * 0.4) * recency_factor * distance_factor)
        score += post_risk
        nearby.append((post, distance))

    penalty = min(20, int(round(score)))
    reasons = []
    if nearby:
        reasons.append(f"Community alerts nearby: {len(nearby)} post(s) within 2 km.")
    if penalty > 0:
        reasons.append(f"Community risk impact applied: -{penalty} points.")
    return penalty, reasons

@app.route("/", methods=["GET", "POST"])
def home():
    score = None
    safety_level = None
    reasons = None
    voice_trigger = None

    if request.method == "POST":
        area = request.form.get("area")
        lighting = request.form.get("lighting")
        traffic = request.form.get("traffic")
        crowd = request.form.get("crowd")
        voice_trigger = request.form.get("voice_trigger")

        score, safety_level, reasons = calculate_score(area, lighting, traffic, crowd, 0)

    return render_template(
        "index.html",
        score=score,
        safety_level=safety_level,
        reasons=reasons,
        voice_trigger=voice_trigger
    )

@app.route("/detect-area", methods=["POST"])
def detect_area():
    data = request.json
    lat = float(data["lat"])
    lon = float(data["lon"])
    area = "Urban" if is_urban_area(lat, lon) else "Rural"
    return jsonify({"area": area})

@app.route("/calculate-score", methods=["POST"])
def calculate_score_api():
    data = request.json
    area = data.get("area")
    lighting = data.get("lighting")
    traffic = data.get("traffic")
    crowd = data.get("crowd")
    vehicle_count = int(data.get("vehicle_count", 0))
    score, safety_level, reasons = calculate_score(area, lighting, traffic, crowd, vehicle_count)

    return jsonify({
        "score": score,
        "safety_level": safety_level,
        "reasons": reasons,
        "community_penalty": 0
    })

@app.route("/external-traffic", methods=["POST"])
def external_traffic():
    data = request.json or {}
    lat = data.get("lat")
    lon = data.get("lon")
    if lat is None or lon is None:
        return jsonify({"available": False, "message": "lat/lon required"}), 400
    return jsonify(fetch_external_traffic(float(lat), float(lon)))


@app.route("/manifest.json", methods=["GET"])
def manifest_file():
    return send_from_directory(BASE_DIR, "manifest.json", mimetype="application/manifest+json")

@app.route("/service-worker.js", methods=["GET"])
def service_worker_file():
    response = send_from_directory(BASE_DIR, "service-worker.js")
    response.headers["Service-Worker-Allowed"] = "/"
    return response

@app.route("/icon-192.png", methods=["GET"])
def icon_192_file():
    return send_from_directory(BASE_DIR, "icon-192.png")

@app.route("/icon-512.png", methods=["GET"])
def icon_512_file():
    return send_from_directory(BASE_DIR, "icon-512.png")

@app.route("/subscribe", methods=["POST"])
def subscribe():
    data = request.json or {}
    token = data.get("token")
    user_token = data.get("user_token")
    if token and user_token and db:
        db.collection("fcm_tokens").document(user_token).set({"token": token}, merge=True)
    return jsonify({"ok": True})

def trigger_push_notifications(title, body):
    if not db: return
    try:
        tokens_query = db.collection("fcm_tokens").stream()
        tokens = [doc.to_dict().get("token") for doc in tokens_query if doc.to_dict().get("token")]
        if not tokens: return
        
        message = messaging.MulticastMessage(notification=messaging.Notification(title=title, body=body), tokens=tokens[:500])
        messaging.send_multicast(message)
    except Exception as e:
        print(f"FCM Push error: {e}")

@app.route("/community-posts", methods=["GET", "POST", "DELETE"])
def community_posts():
    # Auto-cleanup old data whenever the feed is accessed or updated
    cleanup_old_alerts()

    if request.method == "GET":
        return jsonify({"posts": fetch_posts_with_reactions()})

    if request.method == "DELETE":
        if db:
            for doc in db.collection('community_posts').stream():
                data = doc.to_dict()
                image_url = data.get("image_url", "")
                delete_cloudinary_image(image_url)
                doc.reference.delete()
        return jsonify({"ok": True, "message": "All community posts deleted successfully."})

    data = request.json or {}
    image = data.get("image")
    lat = data.get("lat")
    lon = data.get("lon")
    tag = (data.get("tag") or "Other").strip()
    note = (data.get("note") or "").strip()

    if not image or lat is None or lon is None:
        return jsonify({"ok": False, "message": "image, lat, lon required"}), 400

    try:
        # Verify alert content using AI before saving
        image_bgr = decode_data_url_image(image)
        is_real, reason = verify_alert_content(image_bgr, tag)
        if not is_real:
            return jsonify({"ok": False, "message": f"Rejected by AI: {reason}"}), 400

        image_url = save_uploaded_community_image(image)
        created_at = datetime.utcnow().isoformat()
        if db:
            db.collection('community_posts').add({
                'image_url': image_url,
                'lat': float(lat),
                'lon': float(lon),
                'tag': tag,
                'note': note,
                'created_at': created_at,
                'reactions': {},
                'helpful_feedback': {}
            })
            
            # Trigger push notification in background
            threading.Thread(target=trigger_push_notifications, args=(f"Community Alert: {tag}", note if note else "A safety alert was posted near your location.")).start()

    except Exception as exc:
        return jsonify({"ok": False, "message": f"Upload processing failed: {str(exc)}"}), 500

    return jsonify({"ok": True, "message": "Community post uploaded."})

@app.route("/community-posts/<post_id>/react", methods=["POST"])
def react_to_post(post_id):
    data = request.json or {}
    reaction_type = data.get("reaction_type")
    user_token = (data.get("user_token") or "").strip()
    valid_reactions = {"Avoid", "Careful", "Safe now"}

    if reaction_type not in valid_reactions or not user_token:
        return jsonify({"ok": False, "message": "reaction_type and user_token required"}), 400

    if db:
        db.collection('community_posts').document(post_id).set({
            'reactions': { user_token: reaction_type }
        }, merge=True)

    return jsonify({"ok": True, "message": "Reaction updated."})

@app.route("/community-posts/<post_id>", methods=["DELETE"])
def delete_community_post(post_id):
    if db:
        doc_ref = db.collection('community_posts').document(post_id)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            delete_cloudinary_image(data.get("image_url", ""))
            doc_ref.delete()
    return jsonify({"ok": True, "message": "Post deleted successfully."})

@app.route("/community-posts/<post_id>/helpful", methods=["POST"])
def mark_helpful(post_id):
    data = request.json or {}
    helpful = data.get("helpful")
    user_token = (data.get("user_token") or "").strip()

    if helpful is None or not user_token:
        return jsonify({"ok": False, "message": "helpful (bool) and user_token required"}), 400

    if db:
        db.collection('community_posts').document(post_id).set({
            'helpful_feedback': { user_token: True if helpful else False }
        }, merge=True)

    return jsonify({"ok": True})

@app.route("/feedback", methods=["POST"])
def submit_app_feedback():
    data = request.json or {}
    rating = data.get("rating")
    comment = (data.get("comment") or "").strip()
    user_token = (data.get("user_token") or "").strip()

    if not isinstance(rating, int) or rating < 1 or rating > 5 or not user_token:
        return jsonify({"ok": False, "message": "rating (1-5) and user_token required"}), 400

    feedback_record = {
        'user_token': user_token,
        'rating': rating,
        'comment': comment,
        'created_at': datetime.utcnow().isoformat(),
        'remote_ip': request.headers.get("X-Forwarded-For", request.remote_addr),
        'user_agent': request.headers.get("User-Agent", "")
    }

    if db:
        try:
            db.collection('app_feedback').add(feedback_record)
        except Exception:
            app.logger.exception("Failed to store feedback in Firestore.")

    try:
        send_feedback_email(feedback_record)
    except RuntimeError as exc:
        app.logger.warning(str(exc))
        return jsonify({"ok": False, "message": str(exc)}), 503
    except Exception:
        app.logger.exception("Failed to send feedback email.")
        return jsonify({
            "ok": False,
            "message": "Feedback was saved, but the Gmail notification email could not be sent. Please verify the Gmail settings."
        }), 503

    return jsonify({"ok": True, "message": "Feedback received and emailed successfully. Thank you!"})

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/analyze-scene", methods=["POST"])
def analyze_scene():
    try:
        data = request.json or {}
        image_raw = data.get("image", "")
        if not image_raw or "," not in image_raw:
            return jsonify({"ok": False, "message": "Invalid camera frame. Please keep camera open and try again."}), 400

        image_bytes = base64.b64decode(image_raw.split(",", 1)[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"ok": False, "message": "Unable to decode image frame on server."}), 400

        mode = (data.get("mode") or "autosync").lower()
        resized_img = resize_for_analysis(img)
        processed_img = preprocess_for_detection(resized_img)
        lighting, brightness = detect_lighting(processed_img)
        detector_used = "fallback"
        try:
            use_fast_empty = mode != "mobile_detect"
            enable_full_pass = mode == "autosync"
            vehicle_count, vehicles_by_type, people_count, crowd, yolo_mode = detect_scene_yolo(
                processed_img, brightness, allow_fast_empty=use_fast_empty, enable_full_pass=enable_full_pass
            )
            if mode in ["crowd", "autosync", "mobile_detect"]:
                people_count, crowd, face_count = merge_people_counts(processed_img, people_count, mode=mode)
            else:
                face_count = 0
            detector_used = yolo_mode
        except Exception:
            if mode in ["vehicle", "autosync", "mobile_detect"]:
                vehicle_count, vehicles_by_type = detect_vehicles_dnn(processed_img, confidence_threshold=0.30)
            else:
                vehicle_count, vehicles_by_type = 0, {}
                
            if mode in ["crowd", "autosync", "mobile_detect"]:
                people_count, crowd = detect_crowd_opencv(processed_img)
                people_count, crowd, face_count = merge_people_counts(processed_img, people_count, mode=mode)
            else:
                people_count, crowd, face_count = 0, "Less", 0
            detector_used = "mobilenet_hog"

        camera_traffic = estimate_traffic_from_vehicles(vehicle_count)
        lat = data.get("lat")
        lon = data.get("lon")

        traffic_api = {"available": False, "message": "Traffic API skipped for quick camera mode."}
        if mode == "autosync":
            traffic_api = {"available": False, "message": "Location not provided."}
            if lat is not None and lon is not None:
                # Keep Auto Sync responsive on mobile/network by limiting wait time.
                traffic_api = fetch_external_traffic(float(lat), float(lon), timeout_sec=2.5)

        final_traffic, traffic_source = merge_traffic_signals(camera_traffic, traffic_api)
        low_light_warning = (
            "Very low light detected - detection may be limited. Try in brighter conditions if possible."
            if brightness < 38 else ""
        )

        return jsonify({
            "ok": True,
            "lighting": lighting,
            "brightness": brightness,
            "vehicle_count": vehicle_count,
            "vehicles_by_type": vehicles_by_type,
            "crowd_count": people_count,
            "face_count": face_count,
            "crowd": crowd,
            "camera_traffic": camera_traffic,
            "traffic": final_traffic,
            "traffic_api": traffic_api,
            "traffic_source": traffic_source,
            "detector": detector_used,
            "low_light_warning": low_light_warning
        })
    except Exception as exc:
        return jsonify({"ok": False, "message": f"Analyze failed: {str(exc)}"}), 500

@app.route("/analyze-lighting", methods=["POST"])
def analyze_lighting():
    data = request.json
    image_bytes = base64.b64decode(data["image"].split(',')[1])
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    lighting, brightness = detect_lighting(img)
    return jsonify({"lighting": lighting, "brightness": brightness})

if __name__ == "__main__":
    log_service_status()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
