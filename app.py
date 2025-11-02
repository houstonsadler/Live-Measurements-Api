import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify

# Local depth model loader from your bundled code
from depth.loader import load_depth_model

# -------------------------------------------------------------------
# Flask app init
# -------------------------------------------------------------------

app = Flask(__name__)

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
pose = mp_pose.Pose(model_complexity=2)
holistic = mp_holistic.Holistic()

# Constants
KNOWN_OBJECT_WIDTH_CM = 21.0  # ~A4 paper width
FOCAL_LENGTH = 600            # default focal length guess
DEFAULT_HEIGHT_CM = 152.0     # fallback assumed height if user doesn't pass one

# Load depth model once, from local files
depth_model = load_depth_model()

# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------

def cm_to_in(x):
    """Convert centimeters to inches, rounded to 2 decimals."""
    return round(x / 2.54, 2)

def clamp_circumference(width_px, raw_circumference_cm, scale_factor, max_multiplier=2.5):
    """
    Safety cap: stop insane circumferences caused by bad silhouette or depth.
    max_multiplier means circumference can't exceed (width_cm * max_multiplier).
    """
    width_cm = width_px * scale_factor
    max_reasonable = width_cm * max_multiplier
    return min(raw_circumference_cm, max_reasonable)

def quality_note(measurements_cm):
    """
    Generate warnings if something looks off.
    We'll flag if hip is >2x chest, or waist >1.5x chest.
    """
    notes = []

    chest_circ = measurements_cm.get("chest_circumference_cm")
    hip_circ   = measurements_cm.get("hip_circumference_cm")
    waist_circ = measurements_cm.get("waist_circumference_cm")

    if chest_circ and hip_circ:
        # If hip is more than 2.0x chest, probably bad segmentation or pose
        if hip_circ > (2.0 * chest_circ):
            notes.append(
                "Hip reading may be distorted by clothing, pose, or camera angle. "
                "Stand straight, feet under hips, arms slightly out, and avoid baggy clothing."
            )

    if chest_circ and waist_circ:
        # If waist >> chest in odd way
        if waist_circ > (1.5 * chest_circ):
            notes.append(
                "Waist reading looks unusually large vs chest. "
                "Make sure torso is fully visible and camera is about mid-body height."
            )

    return notes

# -------------------------------------------------------------------
# Depth handling
# -------------------------------------------------------------------

def estimate_depth(image):
    """
    Run the bundled/local depth model to get a coarse depth map.
    Return: depth_map as a 2D numpy array (H,W) or None if not available.
    Never raises (we catch and return None).
    """
    if depth_model is None:
        return None

    try:
        # BGR -> RGB, normalize [0,1]
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        # (H, W, 3) -> (1, 3, H, W)
        input_tensor = torch.tensor(
            input_image,
            dtype=torch.float32
        ).permute(2, 0, 1).unsqueeze(0)

        # resize to model expected input shape
        input_tensor = F.interpolate(
            input_tensor,
            size=(384, 384),
            mode="bilinear",
            align_corners=False,
        )

        with torch.no_grad():
            depth_map = depth_model(input_tensor)

        # depth_map should be (1,1,H,W); squeeze to (H,W)
        depth_np = depth_map.squeeze().cpu().numpy()
        return depth_np

    except Exception as e:
        print(f"[Depth disabled at runtime] {e}")
        return None

# -------------------------------------------------------------------
# Geometry / scaling helpers
# -------------------------------------------------------------------

def calibrate_focal_length(image, real_width_cm, detected_width_px):
    """
    If we detect a known-width object (e.g. sheet of paper),
    update focal length guess. Otherwise use default.
    """
    if detected_width_px:
        return (detected_width_px * FOCAL_LENGTH) / real_width_cm
    return FOCAL_LENGTH

def detect_reference_object(image):
    """
    Try to detect a large rectangular object (like paper) to estimate scaling.
    Returns:
        scale_factor (cm per pixel),
        focal_length (pixels).
    Falls back to a guess if no obvious contour is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        focal_length = calibrate_focal_length(
            image,
            KNOWN_OBJECT_WIDTH_CM,
            w
        )
        scale_factor = KNOWN_OBJECT_WIDTH_CM / w  # cm per pixel
        return scale_factor, focal_length

    # fallback guess if nothing detected
    return 0.05, FOCAL_LENGTH

def calculate_distance_using_height(landmarks, image_height, user_height_cm):
    """
    If we know the user's real height in cm, use that to infer scale factor.
    We measure pixel distance from nose to ankles.
    Returns:
        est_distance,
        scale_factor (cm per pixel)
    """
    top_head = landmarks[mp_pose.PoseLandmark.NOSE.value].y * image_height
    bottom_foot = max(
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    ) * image_height

    pixel_height = abs(bottom_foot - top_head)

    est_distance = (user_height_cm * FOCAL_LENGTH) / pixel_height
    scale_factor = user_height_cm / pixel_height  # cm per pixel

    return est_distance, scale_factor

def get_body_width_at_height(frame, height_px, center_x_ratio):
    """
    At a given row (height_px in pixels), estimate body width by thresholding
    and scanning left/right from a reference center.
    We enforce a floor width (10% of frame width) to avoid zeros.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    # clamp row to frame bounds
    if height_px >= frame.shape[0]:
        height_px = frame.shape[0] - 1

    line = thresh[height_px, :]
    center_x = int(center_x_ratio * frame.shape[1])

    left_edge = center_x
    right_edge = center_x

    # scan left
    for i in range(center_x, 0, -1):
        if line[i] == 0:  # black pixel => body
            left_edge = i
            break

    # scan right
    for i in range(center_x, len(line)):
        if line[i] == 0:
            right_edge = i
            break

    width_px = right_edge - left_edge

    # enforce a minimum width ~10% of frame width
    min_width = 0.1 * frame.shape[1]
    if width_px < min_width:
        width_px = min_width

    return width_px

# -------------------------------------------------------------------
# Measurement extraction
# -------------------------------------------------------------------

def calculate_measurements(
    results,
    scale_factor,
    image_width,
    image_height,
    depth_map,
    frame=None,
    user_height_cm=None
):
    """
    Convert pose landmarks + (optional) depth map + scale
    into human-friendly body measurements in cm/in.
    """

    landmarks = results.pose_landmarks.landmark

    # refine scale_factor using known user height if provided
    if user_height_cm:
        _, scale_factor = calculate_distance_using_height(
            landmarks,
            image_height,
            user_height_cm
        )

    def pixel_to_cm(px_val):
        return round(px_val * scale_factor, 2)

    def estimate_circumference(width_px, depth_ratio=1.0):
        """
        Approximate ellipse circumference around torso/waist/hip:
        C ≈ 2π * sqrt((a² + b²)/2)
        a = width/2, b = depth/2.
        We approximate depth ≈ 0.7 * width * depth_ratio.
        """
        width_cm = width_px * scale_factor
        est_depth_cm = width_cm * depth_ratio * 0.7
        a = width_cm / 2.0
        b = est_depth_cm / 2.0
        c = 2.0 * np.pi * np.sqrt((a**2 + b**2) / 2.0)
        return round(c, 2)

    def sample_depth_at(depth_map_local, px_x, px_y, img_w, img_h):
        """
        Safely read a depth value from depth_map using original image coords.
        depth_map may be any (D_H, D_W); we scale and clamp.
        Returns float depth value or None.
        """
        if depth_map_local is None:
            return None

        D_H, D_W = depth_map_local.shape[:2]

        x_scaled = int(px_x * (D_W / img_w))
        y_scaled = int(px_y * (D_H / img_h))

        # clamp to valid bounds
        x_scaled = max(0, min(D_W - 1, x_scaled))
        y_scaled = max(0, min(D_H - 1, y_scaled))

        return float(depth_map_local[y_scaled, x_scaled])

    measurements = {}

    # --------------------
    # Shoulders
    # --------------------
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    shoulder_width_px = abs(
        left_shoulder.x * image_width
        - right_shoulder.x * image_width
    )

    # widen slightly
    shoulder_width_px *= 1.1
    measurements["shoulder_width_cm"] = pixel_to_cm(shoulder_width_px)

    # --------------------
    # Chest
    # --------------------
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # chest y ~ 15% down from shoulder->hip
    chest_y_ratio = 0.15
    chest_y = left_shoulder.y + (left_hip.y - left_shoulder.y) * chest_y_ratio

    # start chest width ~ shoulder span * 1.15
    chest_width_px = abs(
        (right_shoulder.x - left_shoulder.x) * image_width
    ) * 1.15

    # refine with contour at chest height
    if frame is not None:
        chest_y_px = int(chest_y * image_height)
        center_x_ratio = (left_shoulder.x + right_shoulder.x) / 2.0
        detected_width = get_body_width_at_height(
            frame,
            chest_y_px,
            center_x_ratio
        )
        if detected_width > 0:
            chest_width_px = max(chest_width_px, detected_width)

    # depth ratio for chest
    chest_depth_ratio = 1.0
    chest_x_px = int(((left_shoulder.x + right_shoulder.x) / 2.0) * image_width)
    chest_y_px = int(chest_y * image_height)
    chest_depth_val = sample_depth_at(
        depth_map,
        chest_x_px,
        chest_y_px,
        image_width,
        image_height
    )
    if chest_depth_val is not None:
        max_depth = float(np.max(depth_map))
        chest_depth_ratio = 1.0 + 0.5 * (1.0 - chest_depth_val / max_depth)

    # chest circumference
    chest_circ_raw_cm = estimate_circumference(chest_width_px, chest_depth_ratio)
    chest_circ_cm = clamp_circumference(
        chest_width_px,
        chest_circ_raw_cm,
        scale_factor,
        max_multiplier=2.5
    )

    measurements["chest_width_cm"] = pixel_to_cm(chest_width_px)
    measurements["chest_circumference_cm"] = chest_circ_cm
    measurements["chest_circumference_in"] = cm_to_in(chest_circ_cm)

    # --------------------
    # Waist
    # --------------------
    # waist y ~ 35% down shoulder->hip (higher than hips)
    waist_y_ratio = 0.35
    waist_y = left_shoulder.y + (left_hip.y - left_shoulder.y) * waist_y_ratio

    # contour-based waist width or fallback to ~hip*0.9
    if frame is not None:
        waist_y_px = int(waist_y * image_height)
        center_x_ratio = (left_hip.x + right_hip.x) / 2.0
        waist_width_px = get_body_width_at_height(
            frame,
            waist_y_px,
            center_x_ratio
        )
    else:
        waist_width_px = abs(right_hip.x - left_hip.x) * image_width * 0.9

    # bump waist outward ~16%
    waist_width_px *= 1.16

    # depth ratio waist
    waist_depth_ratio = 1.0
    waist_x_px = int(((left_hip.x + right_hip.x) / 2.0) * image_width)
    waist_y_px = int(waist_y * image_height)
    waist_depth_val = sample_depth_at(
        depth_map,
        waist_x_px,
        waist_y_px,
        image_width,
        image_height
    )
    if waist_depth_val is not None:
        max_depth = float(np.max(depth_map))
        waist_depth_ratio = 1.0 + 0.5 * (1.0 - waist_depth_val / max_depth)

    waist_circ_raw_cm = estimate_circumference(waist_width_px, waist_depth_ratio)
    waist_circ_cm = clamp_circumference(
        waist_width_px,
        waist_circ_raw_cm,
        scale_factor,
        max_multiplier=2.5
    )

    measurements["waist_width_cm"] = pixel_to_cm(waist_width_px)
    measurements["waist_circumference_cm"] = waist_circ_cm
    measurements["waist_circumference_in"] = cm_to_in(waist_circ_cm)

    # --------------------
    # Hips
    # --------------------
    # hip width ~35% bump over detected hip landmark span
    hip_width_px = abs(
        left_hip.x * image_width
        - right_hip.x * image_width
    ) * 1.35

    # refine hips using contour slightly below hip line
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    if frame is not None:
        hip_y_offset_ratio = 0.10  # 10% toward thigh
        hip_y = left_hip.y + (left_knee.y - left_hip.y) * hip_y_offset_ratio

        hip_y_px = int(hip_y * image_height)
        center_x_ratio = (left_hip.x + right_hip.x) / 2.0
        detected_width = get_body_width_at_height(
            frame,
            hip_y_px,
            center_x_ratio
        )
        if detected_width > 0:
            hip_width_px = max(hip_width_px, detected_width)

    hip_depth_ratio = 1.0
    hip_x_px = int(((left_hip.x + right_hip.x) / 2.0) * image_width)
    hip_y_px = int(left_hip.y * image_height)
    hip_depth_val = sample_depth_at(
        depth_map,
        hip_x_px,
        hip_y_px,
        image_width,
        image_height
    )
    if hip_depth_val is not None:
        max_depth = float(np.max(depth_map))
        hip_depth_ratio = 1.0 + 0.5 * (1.0 - hip_depth_val / max_depth)

    hip_circ_raw_cm = estimate_circumference(hip_width_px, hip_depth_ratio)
    hip_circ_cm = clamp_circumference(
        hip_width_px,
        hip_circ_raw_cm,
        scale_factor,
        max_multiplier=2.5
    )

    measurements["hip_width_cm"] = pixel_to_cm(hip_width_px)
    measurements["hip_circumference_cm"] = hip_circ_cm
    measurements["hip_circumference_in"] = cm_to_in(hip_circ_cm)

    # --------------------
    # Neck
    # --------------------
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]

    neck_width_px = abs(
        nose.x * image_width
        - left_ear.x * image_width
    ) * 2.0

    neck_circ_raw_cm = estimate_circumference(neck_width_px, 1.0)
    neck_circ_cm = clamp_circumference(
        neck_width_px,
        neck_circ_raw_cm,
        scale_factor,
        max_multiplier=2.5
    )

    measurements["neck_width_cm"] = pixel_to_cm(neck_width_px)
    measurements["neck_circumference_cm"] = neck_circ_cm
    measurements["neck_circumference_in"] = cm_to_in(neck_circ_cm)

    # --------------------
    # Arm length (shoulder -> wrist)
    # --------------------
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    sleeve_length_px = abs(
        left_shoulder.y * image_height
        - left_wrist.y * image_height
    )
    measurements["arm_length_cm"] = pixel_to_cm(sleeve_length_px)

    # --------------------
    # Shirt length (shoulder -> hip, padded 1.2x)
    # --------------------
    shirt_length_px = abs(
        left_shoulder.y * image_height
        - left_hip.y * image_height
    ) * 1.2
    measurements["shirt_length_cm"] = pixel_to_cm(shirt_length_px)

    # --------------------
    # Thigh
    # --------------------
    thigh_y_ratio = 0.20  # ~20% down hip->knee
    thigh_y = left_hip.y + (left_knee.y - left_hip.y) * thigh_y_ratio

    thigh_width_px = hip_width_px * 0.5 * 1.2  # base guess from hips

    if frame is not None:
        thigh_y_px = int(thigh_y * image_height)
        thigh_center_x_ratio = left_hip.x * 0.9  # a bit inward
        detected_width = get_body_width_at_height(
            frame,
            thigh_y_px,
            thigh_center_x_ratio
        )
        # sanity: don't let thigh appear wider than hips
        if 0 < detected_width < hip_width_px:
            thigh_width_px = detected_width

    thigh_depth_ratio = 1.0
    thigh_x_px = int(left_hip.x * image_width)
    thigh_y_px = int(thigh_y * image_height)
    thigh_depth_val = sample_depth_at(
        depth_map,
        thigh_x_px,
        thigh_y_px,
        image_width,
        image_height
    )
    if thigh_depth_val is not None:
        max_depth = float(np.max(depth_map))
        thigh_depth_ratio = 1.0 + 0.5 * (1.0 - thigh_depth_val / max_depth)

    thigh_circ_raw_cm = estimate_circumference(thigh_width_px, thigh_depth_ratio)
    thigh_circ_cm = clamp_circumference(
        thigh_width_px,
        thigh_circ_raw_cm,
        scale_factor,
        max_multiplier=2.5
    )

    measurements["thigh_width_cm"] = pixel_to_cm(thigh_width_px)
    measurements["thigh_circumference_cm"] = thigh_circ_cm
    measurements["thigh_circumference_in"] = cm_to_in(thigh_circ_cm)

    # --------------------
    # Trouser length (hip -> ankle)
    # --------------------
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    trouser_length_px = abs(
        left_hip.y * image_height
        - left_ankle.y * image_height
    )
    measurements["trouser_length_cm"] = pixel_to_cm(trouser_length_px)

    # --------------------
    # Add notes if anything looks off
    # --------------------
    circ_summary_for_qc = {
        "chest_circumference_cm": measurements.get("chest_circumference_cm"),
        "waist_circumference_cm": measurements.get("waist_circumference_cm"),
        "hip_circumference_cm": measurements.get("hip_circumference_cm"),
    }

    notes_list = quality_note(circ_summary_for_qc)
    if notes_list:
        measurements["notes"] = notes_list

    return measurements

# -------------------------------------------------------------------
# Pose validation
# -------------------------------------------------------------------

def validate_front_image(image_np):
    """
    Basic validation to help ensure:
    - There's a full-ish body in frame
    - We have key joints
    - It's not just a selfie close-up
    Returns (True, msg) if OK, (False, msg) if not.
    """
    try:
        rgb_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_height, image_width = image_np.shape[:2]

        # Lightweight holistic model in static mode to validate
        with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False
        ) as validator_model:

            results = validator_model.process(rgb_frame)

        # Did we detect any pose at all?
        if not hasattr(results, "pose_landmarks") or not results.pose_landmarks:
            return False, (
                "No person detected. Please make sure your full body is visible in the frame."
            )

        REQUIRED = [
            mp_holistic.PoseLandmark.NOSE,
            mp_holistic.PoseLandmark.LEFT_SHOULDER,
            mp_holistic.PoseLandmark.RIGHT_SHOULDER,
            mp_holistic.PoseLandmark.LEFT_ELBOW,
            mp_holistic.PoseLandmark.RIGHT_ELBOW,
            mp_holistic.PoseLandmark.LEFT_KNEE,
            mp_holistic.PoseLandmark.RIGHT_KNEE,
        ]

        # Ensure these joints are visible and in-frame
        missing = []
        for lm in REQUIRED:
            lm_data = results.pose_landmarks.landmark[lm]
            if (
                lm_data.visibility < 0.5 or
                lm_data.x < 0 or lm_data.x > 1 or
                lm_data.y < 0 or lm_data.y > 1
            ):
                missing.append(lm.name.replace("_", " "))

        if missing:
            return False, (
                "Couldn't detect full body. Please stand fully in frame head-to-toe."
            )

        # Selfie-too-close check:
        nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
        l_sh = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        r_sh = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

        shoulder_width_px = abs(l_sh.x - r_sh.x) * image_width
        head_to_shoulder_px = abs(l_sh.y - nose.y) * image_height

        # If shoulders are extremely narrow compared to head distance at this crop,
        # user is probably too close
        if shoulder_width_px < head_to_shoulder_px * 1.2:
            return False, (
                "Please step back so we can see more of your upper body, not just your face."
            )

        return True, "Validation passed - proceeding with measurements"

    except Exception as e:
        print(f"Error validating body image: {e}")
        return False, "Image could not be read. Please retake and try again."

# -------------------------------------------------------------------
# Flask routes
# -------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "service": "live-measurements-api",
        "message": "POST /upload_images with 'front', 'left_side', and 'height_cm' to get body measurements."
    }), 200

@app.route("/upload_images", methods=["POST"])
def upload_images():
    """
    Main API endpoint.
    Expects multipart/form-data containing:
        front      (REQUIRED)  -> full-body front image
        left_side  (OPTIONAL)  -> side view image
        height_cm  (OPTIONAL)  -> numeric height in cm to improve scaling
    Returns measurements (cm/in) + debug info.
    """
    # Must have a 'front' image
    if "front" not in request.files:
        return jsonify({"error": "Missing front image for reference."}), 400

    # Read the 'front' image now
    front_file = request.files["front"]
    front_np = np.frombuffer(front_file.read(), np.uint8)
    front_file.seek(0)  # reset in case we need to reread
    front_frame = cv2.imdecode(front_np, cv2.IMREAD_COLOR)

    # Validate pose / framing
    is_valid, error_msg = validate_front_image(front_frame)
    if not is_valid:
        return jsonify({
            "error": error_msg,
            "pose": "front",
            "code": "INVALID_POSE"
        }), 400

    # Pull user's reported height if provided (use default otherwise)
    user_height_cm = request.form.get("height_cm")
    try:
        user_height_cm = float(user_height_cm) if user_height_cm else DEFAULT_HEIGHT_CM
    except ValueError:
        user_height_cm = DEFAULT_HEIGHT_CM

    # Collect provided images
    received_images = {
        key: request.files[key]
        for key in ["front", "left_side"]
        if key in request.files
    }

    measurements = {}
    scale_factor = None
    focal_length = FOCAL_LENGTH

    for pose_name, file in received_images.items():
        img_np = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # Run holistic pose on this frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = holistic.process(rgb)

        h, w, _ = frame.shape

        # If this is the front view and we got pose landmarks, use the user's height
        if pose_name == "front" and pose_results.pose_landmarks:
            _, scale_factor = calculate_distance_using_height(
                pose_results.pose_landmarks.landmark,
                h,
                user_height_cm
            )
        else:
            # fallback / backup: try to infer scale using reference object
            scale_factor, focal_length = detect_reference_object(frame)

        # Generate depth map for this frame
        depth_map = estimate_depth(frame)

        # For now, only generate measurements from the front image
        if pose_name == "front" and pose_results.pose_landmarks:
            measurements.update(
                calculate_measurements(
                    pose_results,
                    scale_factor,
                    w,
                    h,
                    depth_map,
                    frame,
                    user_height_cm
                )
            )

    debug_info = {
        "scale_factor": float(scale_factor) if scale_factor else None,
        "focal_length": float(focal_length),
        "user_height_cm": float(user_height_cm),
    }

    return jsonify({
        "measurements": measurements,
        "debug_info": debug_info
    })

# -------------------------------------------------------------------
# Entrypoint (local dev only; Render uses gunicorn CMD in Dockerfile)
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Locally and in Docker run mode (not gunicorn), default to 8000
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
