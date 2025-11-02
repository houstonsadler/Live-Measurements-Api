import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify

# Import your local model loader
from depth.loader import load_depth_model

# -------------------------------------------------------------------
# Flask init
# -------------------------------------------------------------------

app = Flask(__name__)

# Initialize MediaPipe once (not per request)
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
pose = mp_pose.Pose(model_complexity=2)
holistic = mp_holistic.Holistic()

# Constants / tunables
KNOWN_OBJECT_WIDTH_CM = 21.0   # A4 paper width reference
FOCAL_LENGTH = 600             # fallback focal length constant
DEFAULT_HEIGHT_CM = 152.0      # default height if none provided
MAX_DIM = 1280                 # max dimension (px) after downscale to save memory

# Load local depth model once (used only as fallback if no side image)
depth_model = load_depth_model()

# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------

def cm_to_in(x_cm: float) -> float:
    return round(x_cm / 2.54, 2)

def clamp_circumference(width_px, raw_circumference_cm, scale_factor, max_multiplier=2.5):
    """
    Prevents insane blow-ups (like 300cm hips) from noisy contours.
    We cap circumference at ~max_multiplier * width_cm.
    """
    width_cm = width_px * scale_factor
    max_reasonable = width_cm * max_multiplier
    return min(raw_circumference_cm, max_reasonable)

def quality_note(measurements_cm):
    """
    Create human-readable notes if proportions look suspicious,
    so the UI can guide the user to retake.
    """
    notes = []

    chest_circ = measurements_cm.get("chest_circumference_cm")
    waist_circ = measurements_cm.get("waist_circumference_cm")
    hip_circ   = measurements_cm.get("hip_circumference_cm")

    # hips way bigger than chest => likely stance/arm/background issue
    if chest_circ and hip_circ:
        if hip_circ > (1.4 * chest_circ):
            notes.append(
                "Hip reading may be distorted by stance, arm position, or camera angle. "
                "Stand straight, arms slightly out, feet under hips, and avoid loose fabric."
            )

    # waist way too tiny relative to chest => usually shadow/arm blocking torso
    if chest_circ and waist_circ:
        if waist_circ < (0.6 * chest_circ):
            notes.append(
                "Waist reading may be too small. Make sure arms are not blocking your midsection "
                "and that your torso is clearly visible."
            )

    return notes

def downscale_frame_if_needed(frame):
    """
    Shrink very large phone images so they don't blow RAM.
    Caps longest side to MAX_DIM pixels.
    """
    if frame is None:
        return frame

    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= MAX_DIM:
        return frame  # no resize needed

    scale = MAX_DIM / longest
    new_w = int(w * scale)
    new_h = int(h * scale)

    print(f"[resize] original={w}x{h} -> resized={new_w}x{new_h}")

    frame = cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )
    return frame

# -------------------------------------------------------------------
# Depth helpers (fallback mode if no side frame)
# -------------------------------------------------------------------

def estimate_depth(image):
    """
    Generate a depth map using the local MiDaS-like model.
    Returns None if model is unavailable or fails.
    """
    if depth_model is None:
        return None

    try:
        rgb_norm = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        # (H,W,3) -> (1,3,H,W)
        input_tensor = torch.tensor(
            rgb_norm,
            dtype=torch.float32
        ).permute(2, 0, 1).unsqueeze(0)

        # MiDaS_small expects ~384x384
        input_tensor = F.interpolate(
            input_tensor,
            size=(384, 384),
            mode="bilinear",
            align_corners=False,
        )

        with torch.no_grad():
            depth_map = depth_model(input_tensor)

        depth_np = depth_map.squeeze().cpu().numpy()
        return depth_np

    except Exception as e:
        print(f"[Depth fallback disabled] {e}")
        return None

def sample_depth_at(depth_map_local, px_x, px_y, img_w, img_h):
    """
    Safely read a single depth value at (px_x, px_y) in image space by
    mapping coordinates into the depth_map space.
    """
    if depth_map_local is None:
        return None

    D_H, D_W = depth_map_local.shape[:2]

    x_scaled = int(px_x * (D_W / img_w))
    y_scaled = int(px_y * (D_H / img_h))

    x_scaled = max(0, min(D_W - 1, x_scaled))
    y_scaled = max(0, min(D_H - 1, y_scaled))

    return float(depth_map_local[y_scaled, x_scaled])

# -------------------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------------------

def calibrate_focal_length(image, real_width_cm, detected_width_px):
    """
    Estimate focal length based on a known-size rectangular object (like paper).
    """
    if detected_width_px:
        return (detected_width_px * FOCAL_LENGTH) / real_width_cm
    return FOCAL_LENGTH

def detect_reference_object(image):
    """
    Try to detect a large rectangular contour (like an A4 sheet held in frame).
    Returns:
        scale_factor (cm/px),
        focal_length (for debug).
    Falls back to generic guesses.
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
        scale_factor = KNOWN_OBJECT_WIDTH_CM / w  # cm/px
        return scale_factor, focal_length

    return 0.05, FOCAL_LENGTH

def calculate_distance_using_height(landmarks, image_height, user_height_cm):
    """
    Use a user's stated real height (cm) to infer scale (cm per pixel).
    We measure pixel distance from nose to ankle.
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

def get_body_width_at_height_front(frame, height_px, center_x_ratio):
    """
    FRONT VIEW:
    Scan horizontally at a given vertical row (height_px),
    starting from center_x_ratio and walking left/right to find body edge.
    Returns "left-right width in pixels".
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    if height_px >= frame.shape[0]:
        height_px = frame.shape[0] - 1

    row = thresh[height_px, :]
    center_x = int(center_x_ratio * frame.shape[1])

    left_edge = center_x
    right_edge = center_x

    # walk left
    for i in range(center_x, 0, -1):
        if row[i] == 0:  # 0 => foreground/body
            left_edge = i
            break

    # walk right
    for j in range(center_x, len(row)):
        if row[j] == 0:
            right_edge = j
            break

    width_px = right_edge - left_edge

    # enforce minimum width ~10% of frame width
    min_width = 0.1 * frame.shape[1]
    if width_px < min_width:
        width_px = min_width

    return width_px

def get_body_thickness_side(frame, height_px):
    """
    SIDE VIEW:
    At a given vertical row, scan from entire left edge to right edge to find
    the first and last body pixels. That approximates front-to-back thickness.
    We do NOT assume symmetry around a 'center' because from the side
    the torso is basically one blob.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    if height_px >= frame.shape[0]:
        height_px = frame.shape[0] - 1

    row = thresh[height_px, :]

    # first "body" pixel from left
    left_edge = None
    for i in range(len(row)):
        if row[i] == 0:
            left_edge = i
            break

    # first "body" pixel from right
    right_edge = None
    for j in range(len(row) - 1, -1, -1):
        if row[j] == 0:
            right_edge = j
            break

    if left_edge is None or right_edge is None:
        # fallback to tiny thickness
        thickness_px = 0.05 * frame.shape[1]
    else:
        thickness_px = right_edge - left_edge

    # enforce a floor ~5% of frame width
    min_width = 0.05 * frame.shape[1]
    if thickness_px < min_width:
        thickness_px = min_width

    return thickness_px

# -------------------------------------------------------------------
# Measurement computation
# -------------------------------------------------------------------

def ellipse_circumference_from_axes(width_cm, thickness_cm):
    """
    Estimate body circumference assuming an ellipse cross-section.
    width_cm   = left-right width
    thickness_cm = front-back thickness
    """
    a = width_cm / 2.0
    b = thickness_cm / 2.0
    return 2.0 * np.pi * np.sqrt((a**2 + b**2) / 2.0)

def calculate_measurements(
    results_front,
    scale_factor,
    front_w,
    front_h,
    front_frame,
    user_height_cm,
    side_frame=None,
    depth_map_front=None
):
    """
    Build all measurements:
    - Use front_frame for width at chest/waist/hips
    - Use side_frame for thickness at those same vertical slices if available
    - Fall back to depth_map_front if side_frame is missing
    - Output cm + inches + notes
    """

    lm = results_front.pose_landmarks.landmark

    # Optionally refine the scale factor using declared height again (precision bump)
    if user_height_cm:
        _, scale_factor = calculate_distance_using_height(lm, front_h, user_height_cm)

    def px_to_cm(px):
        return round(px * scale_factor, 2)

    def safe_thickness_cm_from_side(y_ratio):
        """
        Measure thickness from the side image at a given normalized vertical location.
        Returns None if we can't.
        """
        if side_frame is None:
            return None
        side_h, side_w = side_frame.shape[:2]
        y_px_side = int(y_ratio * side_h)
        thick_px = get_body_thickness_side(side_frame, y_px_side)
        return thick_px * scale_factor  # cm

    def fallback_thickness_cm_from_depth(px_x, px_y):
        """
        Use depth map to approximate body thickness if no side frame.
        We treat relative depth variation as thickness proxy.
        """
        if depth_map_front is None:
            return None

        # sample depth at the "center" pixel
        d_center = sample_depth_at(depth_map_front, px_x, px_y, front_w, front_h)
        if d_center is None:
            return None

        # derive a fake thickness based on relative difference.
        # This is heuristic / legacy behavior.
        max_depth = float(np.max(depth_map_front))
        depth_ratio = 1.0 + 0.5 * (1.0 - d_center / max_depth)
        # assume thickness is ~0.7 * width * depth_ratio (old heuristic)
        return ("use_width_scale", depth_ratio)

    # =========== SHOULDER WIDTH ===========
    left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    shoulder_width_px = abs(
        left_shoulder.x * front_w - right_shoulder.x * front_w
    )
    shoulder_width_px *= 1.1  # small outward pad
    shoulder_width_cm = px_to_cm(shoulder_width_px)

    # =========== CHEST ===========
    left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # chest vertical slice ratio between shoulders and hips
    chest_y_ratio = left_shoulder.y + (left_hip.y - left_shoulder.y) * 0.15
    chest_y_px_front = int(chest_y_ratio * front_h)

    # front width at chest
    center_x_ratio_chest = (left_shoulder.x + right_shoulder.x) / 2.0
    chest_width_px = get_body_width_at_height_front(
        front_frame,
        chest_y_px_front,
        center_x_ratio_chest
    )
    # small pad
    chest_width_px *= 1.15

    # chest thickness from side view
    chest_thickness_cm = safe_thickness_cm_from_side(chest_y_ratio)

    if chest_thickness_cm is None:
        # fallback: approximate from depth
        chest_x_px_front = int(center_x_ratio_chest * front_w)
        depth_fallback = fallback_thickness_cm_from_depth(
            chest_x_px_front,
            chest_y_px_front
        )
        if isinstance(depth_fallback, tuple) and depth_fallback[0] == "use_width_scale":
            depth_ratio = depth_fallback[1]
            chest_width_cm = chest_width_px * scale_factor
            est_thick_cm = chest_width_cm * 0.7 * depth_ratio
            chest_thickness_cm = est_thick_cm
        else:
            # worst case guess
            chest_width_cm = chest_width_px * scale_factor
            chest_thickness_cm = chest_width_cm * 0.7

    # compute chest circumference using ellipse
    chest_width_cm = chest_width_px * scale_factor
    chest_circ_raw_cm = ellipse_circumference_from_axes(
        chest_width_cm,
        chest_thickness_cm
    )
    chest_circ_cm = clamp_circumference(
        chest_width_px,
        chest_circ_raw_cm,
        scale_factor,
        max_multiplier=2.5
    )

    # =========== WAIST ===========
    # waist ~35% down from shoulder->hip
    waist_y_ratio = left_shoulder.y + (left_hip.y - left_shoulder.y) * 0.35
    waist_y_px_front = int(waist_y_ratio * front_h)

    center_x_ratio_waist = (left_hip.x + right_hip.x) / 2.0
    waist_width_px = get_body_width_at_height_front(
        front_frame,
        waist_y_px_front,
        center_x_ratio_waist
    )
    waist_width_px *= 1.16  # pad

    waist_thickness_cm = safe_thickness_cm_from_side(waist_y_ratio)

    if waist_thickness_cm is None:
        waist_x_px_front = int(center_x_ratio_waist * front_w)
        depth_fallback = fallback_thickness_cm_from_depth(
            waist_x_px_front,
            waist_y_px_front
        )
        if isinstance(depth_fallback, tuple) and depth_fallback[0] == "use_width_scale":
            depth_ratio = depth_fallback[1]
            waist_width_cm = waist_width_px * scale_factor
            est_thick_cm = waist_width_cm * 0.7 * depth_ratio
            waist_thickness_cm = est_thick_cm
        else:
            waist_width_cm = waist_width_px * scale_factor
            waist_thickness_cm = waist_width_cm * 0.7

    waist_width_cm = waist_width_px * scale_factor
    waist_circ_raw_cm = ellipse_circumference_from_axes(
        waist_width_cm,
        waist_thickness_cm
    )
    waist_circ_cm = clamp_circumference(
        waist_width_px,
        waist_circ_raw_cm,
        scale_factor,
        max_multiplier=2.5
    )

    # =========== HIPS ===========
    # hips: start with distance between LHIP & RHIP
    hip_width_px = abs(
        left_hip.x * front_w - right_hip.x * front_w
    )
    # Inflate to approximate the real outer curve of the pelvis/seat.
    hip_width_px *= 1.35

    # refine hips using a slightly-lower slice toward thigh
    left_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
    hip_y_ratio = left_hip.y + (left_knee.y - left_hip.y) * 0.10
    hip_y_px_front = int(hip_y_ratio * front_h)

    center_x_ratio_hip = (left_hip.x + right_hip.x) / 2.0
    detected_width_px_hip = get_body_width_at_height_front(
        front_frame,
        hip_y_px_front,
        center_x_ratio_hip
    )
    if detected_width_px_hip > 0:
        hip_width_px = max(hip_width_px, detected_width_px_hip)

    hip_thickness_cm = safe_thickness_cm_from_side(hip_y_ratio)
    if hip_thickness_cm is None:
        hip_x_px_front = int(center_x_ratio_hip * front_w)
        depth_fallback = fallback_thickness_cm_from_depth(
            hip_x_px_front,
            hip_y_px_front
        )
        if isinstance(depth_fallback, tuple) and depth_fallback[0] == "use_width_scale":
            depth_ratio = depth_fallback[1]
            hip_width_cm_tmp = hip_width_px * scale_factor
            est_thick_cm = hip_width_cm_tmp * 0.7 * depth_ratio
            hip_thickness_cm = est_thick_cm
        else:
            hip_width_cm_tmp = hip_width_px * scale_factor
            hip_thickness_cm = hip_width_cm_tmp * 0.7

    hip_width_cm = hip_width_px * scale_factor
    hip_circ_raw_cm = ellipse_circumference_from_axes(
        hip_width_cm,
        hip_thickness_cm
    )
    hip_circ_cm = clamp_circumference(
        hip_width_px,
        hip_circ_raw_cm,
        scale_factor,
        max_multiplier=2.5
    )

    # =========== NECK ===========
    nose = lm[mp_pose.PoseLandmark.NOSE.value]
    left_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]

    neck_width_px = abs(
        nose.x * front_w - left_ear.x * front_w
    ) * 2.0  # approx full neck width
    neck_width_cm = px_to_cm(neck_width_px)

    # Neck thickness approximation: assume ~0.7 * width.
    neck_thickness_cm = neck_width_cm * 0.7
    neck_circ_raw_cm = ellipse_circumference_from_axes(
        neck_width_cm,
        neck_thickness_cm
    )
    neck_circ_cm = clamp_circumference(
        neck_width_px,
        neck_circ_raw_cm,
        scale_factor,
        max_multiplier=2.5
    )

    # =========== ARM LENGTH ===========
    left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
    sleeve_length_px = abs(
        left_shoulder.y * front_h - left_wrist.y * front_h
    )
    arm_length_cm = px_to_cm(sleeve_length_px)

    # =========== SHIRT LENGTH ===========
    shirt_length_px = abs(
        left_shoulder.y * front_h - left_hip.y * front_h
    ) * 1.2
    shirt_length_cm = px_to_cm(shirt_length_px)

    # =========== THIGH CIRCUMFERENCE ===========
    thigh_y_ratio = left_hip.y + (left_knee.y - left_hip.y) * 0.20
    thigh_y_px_front = int(thigh_y_ratio * front_h)

    # start from hip width scaling
    thigh_width_px = hip_width_px * 0.5 * 1.2

    # refine from contour at thigh slice
    thigh_center_ratio = left_hip.x * 0.9  # slight inward offset
    detected_thigh_px = get_body_width_at_height_front(
        front_frame,
        thigh_y_px_front,
        thigh_center_ratio
    )
    # sanity check: thigh shouldn't exceed hip width
    if 0 < detected_thigh_px < hip_width_px:
        thigh_width_px = detected_thigh_px

    thigh_width_cm = thigh_width_px * scale_factor

    # thickness from side at thigh level
    thigh_thickness_cm = safe_thickness_cm_from_side(thigh_y_ratio)
    if thigh_thickness_cm is None:
        thigh_thickness_cm = thigh_width_cm * 0.7  # last-resort fallback

    thigh_circ_raw_cm = ellipse_circumference_from_axes(
        thigh_width_cm,
        thigh_thickness_cm
    )
    thigh_circ_cm = clamp_circumference(
        thigh_width_px,
        thigh_circ_raw_cm,
        scale_factor,
        max_multiplier=2.5
    )

    # =========== TROUSER LENGTH ===========
    left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    trouser_length_px = abs(
        left_hip.y * front_h - left_ankle.y * front_h
    )
    trouser_length_cm = px_to_cm(trouser_length_px)

    # -------------------------------------------------------------------
    # Build output dict
    # -------------------------------------------------------------------

    measurements = {
        "shoulder_width_cm": round(shoulder_width_cm, 2),

        "chest_circumference_cm": round(chest_circ_cm, 2),
        "chest_circumference_in": cm_to_in(chest_circ_cm),
        "chest_width_cm": round(chest_width_cm, 2),

        "waist_circumference_cm": round(waist_circ_cm, 2),
        "waist_circumference_in": cm_to_in(waist_circ_cm),
        "waist_width_cm": round(waist_width_cm, 2),

        "hip_circumference_cm": round(hip_circ_cm, 2),
        "hip_circumference_in": cm_to_in(hip_circ_cm),
        "hip_width_cm": round(hip_width_cm, 2),

        "neck_circumference_cm": round(neck_circ_cm, 2),
        "neck_circumference_in": cm_to_in(neck_circ_cm),
        "neck_width_cm": round(neck_width_cm, 2),

        "arm_length_cm": round(arm_length_cm, 2),
        "shirt_length_cm": round(shirt_length_cm, 2),

        "thigh_circumference_cm": round(thigh_circ_cm, 2),
        "thigh_circumference_in": cm_to_in(thigh_circ_cm),
        "thigh_width_cm": round(thigh_width_cm, 2),

        "trouser_length_cm": round(trouser_length_cm, 2),
    }

    # Add quality guidance notes if something looks off
    circ_summary_for_qc = {
        "chest_circumference_cm": measurements.get("chest_circumference_cm"),
        "waist_circumference_cm": measurements.get("waist_circumference_cm"),
        "hip_circumference_cm": measurements.get("hip_circumference_cm"),
    }
    notes_list = quality_note(circ_summary_for_qc)
    if notes_list:
        measurements["notes"] = notes_list

    # Optional: proportion sanity clamp, e.g. cap hips to <=1.4x chest
    if (
        "hip_circumference_cm" in measurements and
        "chest_circumference_cm" in measurements
    ):
        chest_val = measurements["chest_circumference_cm"]
        hip_val = measurements["hip_circumference_cm"]
        if chest_val > 0 and hip_val > chest_val * 1.4:
            measurements["hip_circumference_cm"] = round(chest_val * 1.4, 2)
            measurements["hip_circumference_in"] = cm_to_in(measurements["hip_circumference_cm"])
            measurements.setdefault("notes", [])
            measurements["notes"].append(
                "Hip estimate looked unusually high compared to chest and was adjusted. "
                "Please retake standing sideways and front at the same camera distance."
            )

    return measurements

# -------------------------------------------------------------------
# Pose validation
# -------------------------------------------------------------------

def validate_front_image(image_np):
    """
    Basic pose validation for the front image:
    - Must see full body (or close)
    - Reject selfie-close crops
    - Check key joints exist
    """
    try:
        rgb_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_np.shape[:2]

        # Use a lightweight Holistic just for validation
        with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False
        ) as validator_model:
            results = validator_model.process(rgb_frame)

        if not hasattr(results, "pose_landmarks") or not results.pose_landmarks:
            return False, "No person detected. Make sure your full body is visible."

        REQUIRED = [
            mp_holistic.PoseLandmark.NOSE,
            mp_holistic.PoseLandmark.LEFT_SHOULDER,
            mp_holistic.PoseLandmark.RIGHT_SHOULDER,
            mp_holistic.PoseLandmark.LEFT_ELBOW,
            mp_holistic.PoseLandmark.RIGHT_ELBOW,
            mp_holistic.PoseLandmark.LEFT_KNEE,
            mp_holistic.PoseLandmark.RIGHT_KNEE,
        ]

        lm = results.pose_landmarks.landmark
        missing = []
        for lm_id in REQUIRED:
            pt = lm[lm_id]
            if (
                pt.visibility < 0.5 or
                pt.x < 0 or pt.x > 1 or
                pt.y < 0 or pt.y > 1
            ):
                missing.append(lm_id.name.replace("_", " "))

        if missing:
            return False, "Couldn't detect full body. Please stand fully in frame head-to-toe."

        # selfie check: if shoulders look narrower than head depth => too close
        nose_pt = lm[mp_holistic.PoseLandmark.NOSE]
        l_sh = lm[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        r_sh = lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

        shoulder_width_px = abs(l_sh.x - r_sh.x) * img_w
        head_to_shoulder_px = abs(l_sh.y - nose_pt.y) * img_h

        if shoulder_width_px < head_to_shoulder_px * 1.2:
            return False, "Please step back so we can see your upper body and hips, not just your face."

        return True, "OK"

    except Exception as e:
        print(f"Error validating image: {e}")
        return False, "Image could not be read. Please try again with full body visible."

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "service": "live-measurements-api",
        "message": "POST /upload_images with 'front', optional 'left_side', and 'height_cm' (in cm)."
    }), 200

@app.route("/upload_images", methods=["POST"])
def upload_images():
    """
    multipart/form-data:
      - front (REQUIRED): full-body front photo
      - left_side (OPTIONAL): full-body side photo (same distance, same framing)
      - height_cm (OPTIONAL): numeric height in cm (improves scaling)
    """

    if "front" not in request.files:
        return jsonify({"error": "Missing 'front' image."}), 400

    # Read front image once for validation
    front_file = request.files["front"]
    front_np = np.frombuffer(front_file.read(), np.uint8)
    front_file.seek(0)
    front_frame_for_validation = cv2.imdecode(front_np, cv2.IMREAD_COLOR)
    front_frame_for_validation = downscale_frame_if_needed(front_frame_for_validation)

    # Validate pose / framing
    is_valid, msg = validate_front_image(front_frame_for_validation)
    if not is_valid:
        return jsonify({
            "error": msg,
            "pose": "front",
            "code": "INVALID_POSE"
        }), 400

    # Parse user height (cm)
    user_height_cm = request.form.get("height_cm")
    try:
        user_height_cm = float(user_height_cm) if user_height_cm else DEFAULT_HEIGHT_CM
    except ValueError:
        user_height_cm = DEFAULT_HEIGHT_CM

    # Now process all provided images ("front" and maybe "left_side")
    frames = {}
    pose_results_map = {}

    # We'll also keep track of which one is 'front' and which is 'left_side'
    for pose_name in ["front", "left_side"]:
        if pose_name not in request.files:
            continue

        f = request.files[pose_name]
        arr = np.frombuffer(f.read(), np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        frame = downscale_frame_if_needed(frame)

        frames[pose_name] = frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results_map[pose_name] = holistic.process(rgb)

    # Derive scale factor using the FRONT pose landmarks, if available.
    scale_factor = None
    focal_length = FOCAL_LENGTH

    if "front" in pose_results_map and pose_results_map["front"].pose_landmarks:
        front_lm = pose_results_map["front"].pose_landmarks.landmark

        # scale from height first (best)
        _, scale_factor = calculate_distance_using_height(
            front_lm,
            frames["front"].shape[0],
            user_height_cm
        )
    else:
        # fallback to reference object if no pose landmarks for some reason
        scale_factor, focal_length = detect_reference_object(frames["front"])

    # Prepare optional depth map for legacy fallback if no side frame
    depth_map_front = None
    if "front" in frames:
        depth_map_front = estimate_depth(frames["front"])

    # Calculate final measurements from FRONT frame plus SIDE frame (if given)
    if "front" in pose_results_map and pose_results_map["front"].pose_landmarks:
        measurements = calculate_measurements(
            results_front=pose_results_map["front"],
            scale_factor=scale_factor,
            front_w=frames["front"].shape[1],
            front_h=frames["front"].shape[0],
            front_frame=frames["front"],
            user_height_cm=user_height_cm,
            side_frame=frames.get("left_side"),
            depth_map_front=depth_map_front,
        )
    else:
        # We can't produce measurements without a valid front pose.
        return jsonify({
            "error": "Unable to detect body landmarks in front image. Please retake.",
            "code": "NO_LANDMARKS"
        }), 400

    debug_info = {
        "scale_factor": float(scale_factor) if scale_factor else None,
        "focal_length": float(focal_length),
        "user_height_cm": float(user_height_cm)
    }

    return jsonify({
        "measurements": measurements,
        "debug_info": debug_info
    })

# -------------------------------------------------------------------
# Local dev entrypoint
# (Render will run gunicorn via Dockerfile CMD, not this block)
# -------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
