import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
pose = mp_pose.Pose(model_complexity=2)
holistic = mp_holistic.Holistic()

# Constants
KNOWN_OBJECT_WIDTH_CM = 21.0  # A4 paper width in cm
FOCAL_LENGTH = 600
DEFAULT_HEIGHT_CM = 152.0

# Load depth estimation model
def load_depth_model():
    """Load MiDaS depth estimation model."""
    try:
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading MiDaS model: {e}")
        return None

depth_model = load_depth_model()


def calibrate_focal_length(image, real_width_cm, detected_width_px):
    """Calibrates focal length dynamically using a known reference object."""
    return (detected_width_px * FOCAL_LENGTH) / real_width_cm if detected_width_px else FOCAL_LENGTH


def detect_reference_object(image):
    """Detects a rectangular reference (like a sheet of paper) to calibrate scaling."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        focal_length = calibrate_focal_length(image, KNOWN_OBJECT_WIDTH_CM, w)
        scale_factor = KNOWN_OBJECT_WIDTH_CM / w
        return scale_factor, focal_length
    return 0.05, FOCAL_LENGTH


def estimate_depth(image):
    """Generate a depth map using MiDaS."""
    if depth_model is None:
        return None

    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    input_tensor = F.interpolate(input_tensor, size=(384, 384), mode="bilinear", align_corners=False)

    with torch.no_grad():
        depth_map = depth_model(input_tensor)

    return depth_map.squeeze().numpy()


def calculate_distance_using_height(landmarks, image_height, user_height_cm):
    """Use user height to estimate distance and scale."""
    top_head = landmarks[mp_pose.PoseLandmark.NOSE.value].y * image_height
    bottom_foot = max(
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    ) * image_height

    person_height_px = abs(bottom_foot - top_head)
    distance = (user_height_cm * FOCAL_LENGTH) / person_height_px
    scale_factor = user_height_cm / person_height_px
    return distance, scale_factor


def get_body_width_at_height(frame, height_px, center_x):
    """Measure body width horizontally at a given height."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    height_px = min(height_px, frame.shape[0] - 1)
    line = thresh[height_px, :]
    center_x = int(center_x * frame.shape[1])

    left_edge, right_edge = center_x, center_x
    for i in range(center_x, 0, -1):
        if line[i] == 0:
            left_edge = i
            break
    for i in range(center_x, len(line)):
        if line[i] == 0:
            right_edge = i
            break

    width_px = right_edge - left_edge
    min_width = 0.1 * frame.shape[1]
    return max(width_px, min_width)


# ... (keep your calculate_measurements and validate_front_image exactly as-is)
# No change needed in the logic — only formatting/structure.

@app.route("/upload_images", methods=["POST"])
def upload_images():
    """Main API endpoint for uploading front and side images."""
    if "front" not in request.files:
        return jsonify({"error": "Missing front image for reference."}), 400

    front_file = request.files["front"]
    front_np = np.frombuffer(front_file.read(), np.uint8)
    front_file.seek(0)

    is_valid, error_msg = validate_front_image(cv2.imdecode(front_np, cv2.IMREAD_COLOR))
    if not is_valid:
        return jsonify({"error": error_msg, "pose": "front", "code": "INVALID_POSE"}), 400

    user_height_cm = request.form.get("height_cm")
    try:
        user_height_cm = float(user_height_cm) if user_height_cm else DEFAULT_HEIGHT_CM
    except ValueError:
        user_height_cm = DEFAULT_HEIGHT_CM

    received = {k: request.files[k] for k in ["front", "left_side"] if k in request.files}
    measurements, scale_factor, focal_length = {}, None, FOCAL_LENGTH
    frames, results = {}, {}

    for pose_name, file in received.items():
        image_np = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        frames[pose_name] = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results[pose_name] = holistic.process(rgb)
        h, w, _ = frame.shape

        if pose_name == "front" and results[pose_name].pose_landmarks:
            _, scale_factor = calculate_distance_using_height(
                results[pose_name].pose_landmarks.landmark, h, user_height_cm
            )
        else:
            scale_factor, focal_length = detect_reference_object(frame)

        depth_map = estimate_depth(frame)
        if pose_name == "front" and results[pose_name].pose_landmarks:
            measurements.update(
                calculate_measurements(
                    results[pose_name],
                    scale_factor,
                    w,
                    h,
                    depth_map,
                    frames[pose_name],
                    user_height_cm,
                )
            )

    debug_info = {
        "scale_factor": float(scale_factor) if scale_factor else None,
        "focal_length": float(focal_length),
        "user_height_cm": float(user_height_cm),
    }

    return jsonify({"measurements": measurements, "debug_info": debug_info})


# --- Main entrypoint ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
