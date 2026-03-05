from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime
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
import sqlite3
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)

CITY_LAT = 16.5062
CITY_LON = 80.6480
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "community.db")
PROTO_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
CAFFE_MODEL_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")
PROTO_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt"
CAFFE_MODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"

MOBILENET_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
VEHICLE_CLASSES = {"bicycle", "bus", "car", "motorbike", "train"}

_vehicle_net = None
_vehicle_net_lock = threading.Lock()
_hog_detector = None

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

if not os.getenv("TOMTOM_API_KEY"):
    os.environ["TOMTOM_API_KEY"] = "1KJRcaiieEGOw93WYtLK20Vep8rHO8DR"

def log_service_status():
    print(f"TOMTOM_API_KEY set: {bool(os.getenv('TOMTOM_API_KEY'))}")

def is_urban_area(lat, lon):
    distance = math.sqrt((lat - CITY_LAT) ** 2 + (lon - CITY_LON) ** 2)
    return distance < 0.1

def calculate_score(area, lighting, traffic, crowd, vehicle_count=0, community_penalty=0, community_reasons=None):
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

    if community_penalty > 0:
        score -= community_penalty
        if community_reasons:
            reasons.extend(community_reasons)

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
    net = get_vehicle_detector()
    resized = cv2.resize(img, (300, 300))
    blob = cv2.dnn.blobFromImage(resized, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

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

def estimate_traffic_from_vehicles(vehicle_count):
    return "High" if vehicle_count >= 3 else "Low"

def fetch_external_traffic(lat, lon):
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
        with urllib.request.urlopen(endpoint, timeout=8, context=ssl_context) as response:
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
                with urllib.request.urlopen(endpoint, timeout=8, context=insecure_context) as response:
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

def init_storage():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_url TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                tag TEXT NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER NOT NULL,
                user_token TEXT NOT NULL,
                reaction_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(post_id, user_token),
                FOREIGN KEY(post_id) REFERENCES posts(id)
            )
            """
        )
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ New tables for feedback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        conn.execute("""
            CREATE TABLE IF NOT EXISTS helpful_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER NOT NULL,
                user_token TEXT NOT NULL,
                helpful BOOLEAN NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(post_id, user_token),
                FOREIGN KEY(post_id) REFERENCES posts(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_token TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                comment TEXT,
                created_at TEXT NOT NULL
            )
        """)
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        conn.commit()

def decode_data_url_image(image_data_url):
    image_bytes = base64.b64decode(image_data_url.split(",")[1])
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def save_uploaded_community_image(image_data_url):
    image = decode_data_url_image(image_data_url)
    filename = secure_filename(f"{uuid.uuid4().hex}.jpg")
    file_path = os.path.join(UPLOAD_DIR, filename)
    cv2.imwrite(file_path, image)
    return f"/uploads/{filename}"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

def fetch_posts_with_reactions():
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                p.id, p.image_url, p.lat, p.lon, p.tag, p.note, p.created_at,
                SUM(CASE WHEN r.reaction_type='Avoid' THEN 1 ELSE 0 END) AS avoid_count,
                SUM(CASE WHEN r.reaction_type='Careful' THEN 1 ELSE 0 END) AS careful_count,
                SUM(CASE WHEN r.reaction_type='Safe now' THEN 1 ELSE 0 END) AS safe_now_count
            FROM posts p
            LEFT JOIN reactions r ON r.post_id = p.id
            GROUP BY p.id
            ORDER BY p.id DESC
            """
        ).fetchall()

    posts = []
    for row in rows:
        posts.append({
            "id": row["id"],
            "image_url": row["image_url"],
            "lat": row["lat"],
            "lon": row["lon"],
            "tag": row["tag"],
            "note": row["note"] or "",
            "created_at": row["created_at"],
            "reactions": {
                "Avoid": int(row["avoid_count"] or 0),
                "Careful": int(row["careful_count"] or 0),
                "Safe now": int(row["safe_now_count"] or 0)
            }
        })
    return posts

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
    lat = data.get("lat")
    lon = data.get("lon")
    lat = float(lat) if lat is not None else None
    lon = float(lon) if lon is not None else None
    community_penalty, community_reasons = compute_community_penalty(lat, lon)

    score, safety_level, reasons = calculate_score(
        area, lighting, traffic, crowd, vehicle_count,
        community_penalty=community_penalty,
        community_reasons=community_reasons
    )

    return jsonify({
        "score": score,
        "safety_level": safety_level,
        "reasons": reasons,
        "community_penalty": community_penalty
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
@app.route("/community-posts", methods=["GET", "POST"])
def community_posts():
    if request.method == "GET":
        return jsonify({"posts": fetch_posts_with_reactions()})

    data = request.json or {}
    image = data.get("image")
    lat = data.get("lat")
    lon = data.get("lon")
    tag = (data.get("tag") or "Other").strip()
    note = (data.get("note") or "").strip()

    if not image or lat is None or lon is None:
        return jsonify({"ok": False, "message": "image, lat, lon required"}), 400

    image_url = save_uploaded_community_image(image)
    created_at = datetime.utcnow().isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO posts (image_url, lat, lon, tag, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (image_url, float(lat), float(lon), tag, note, created_at)
        )
        conn.commit()

    return jsonify({"ok": True, "message": "Community post uploaded."})

@app.route("/community-posts/<int:post_id>/react", methods=["POST"])
def react_to_post(post_id):
    data = request.json or {}
    reaction_type = data.get("reaction_type")
    user_token = (data.get("user_token") or "").strip()
    valid_reactions = {"Avoid", "Careful", "Safe now"}

    if reaction_type not in valid_reactions or not user_token:
        return jsonify({"ok": False, "message": "reaction_type and user_token required"}), 400

    created_at = datetime.utcnow().isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO reactions (post_id, user_token, reaction_type, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(post_id, user_token) DO UPDATE SET
                reaction_type=excluded.reaction_type,
                created_at=excluded.created_at
            """,
            (post_id, user_token, reaction_type, created_at)
        )
        conn.commit()

    return jsonify({"ok": True, "message": "Reaction updated."})

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ New feedback endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/community-posts/<int:post_id>/helpful", methods=["POST"])
def mark_helpful(post_id):
    data = request.json or {}
    helpful = data.get("helpful")
    user_token = (data.get("user_token") or "").strip()

    if helpful is None or not user_token:
        return jsonify({"ok": False, "message": "helpful (bool) and user_token required"}), 400

    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO helpful_feedback (post_id, user_token, helpful, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(post_id, user_token) DO UPDATE SET
                helpful = excluded.helpful,
                created_at = excluded.created_at
        """, (post_id, user_token, 1 if helpful else 0, datetime.utcnow().isoformat()))
        conn.commit()

    return jsonify({"ok": True})

@app.route("/feedback", methods=["POST"])
def submit_app_feedback():
    data = request.json or {}
    rating = data.get("rating")
    comment = (data.get("comment") or "").strip()
    user_token = (data.get("user_token") or "").strip()

    if not isinstance(rating, int) or rating < 1 or rating > 5 or not user_token:
        return jsonify({"ok": False, "message": "rating (1-5) and user_token required"}), 400

    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO app_feedback (user_token, rating, comment, created_at)
            VALUES (?, ?, ?, ?)
        """, (user_token, rating, comment, datetime.utcnow().isoformat()))
        conn.commit()

    return jsonify({"ok": True, "message": "Feedback received. Thank you!"})

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/analyze-scene", methods=["POST"])
def analyze_scene():
    data = request.json or {}
    image_raw = data.get("image", "")
    image_bytes = base64.b64decode(image_raw.split(",")[1])
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    lighting, brightness = detect_lighting(img)
    vehicle_count, vehicles_by_type = detect_vehicles_dnn(img)
    people_count, crowd = detect_crowd_opencv(img)

    camera_traffic = estimate_traffic_from_vehicles(vehicle_count)
    lat = data.get("lat")
    lon = data.get("lon")

    traffic_api = {"available": False, "message": "Location not provided."}
    if lat is not None and lon is not None:
        traffic_api = fetch_external_traffic(float(lat), float(lon))

    final_traffic = traffic_api["traffic"] if traffic_api.get("available") else camera_traffic

    return jsonify({
        "lighting": lighting,
        "brightness": brightness,
        "vehicle_count": vehicle_count,
        "vehicles_by_type": vehicles_by_type,
        "crowd_count": people_count,
        "crowd": crowd,
        "camera_traffic": camera_traffic,
        "traffic": final_traffic,
        "traffic_api": traffic_api
    })

@app.route("/analyze-lighting", methods=["POST"])
def analyze_lighting():
    data = request.json
    image_bytes = base64.b64decode(data["image"].split(',')[1])
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    lighting, brightness = detect_lighting(img)
    return jsonify({"lighting": lighting, "brightness": brightness})

init_storage()

if __name__ == "__main__":
    log_service_status()
    app.run(debug=True)
