import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify

# If you have local depth model loader (fallback for when side photo missing)
from depth.loader import load_depth_model

# -------------------------------------------------------------------
# Flask + model init
# -------------------------------------------------------------------

app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# Keep these model objects global so we don't re-load per request
pose = mp_pose.Pose(model_complexity=2)
holistic = mp_holistic.Holistic()

# Constants / tuning knobs
KNOWN_OBJECT_WIDTH_CM = 21.0   # A4 paper width (reference object)
FOCAL_LENGTH = 600             # base focal length constant
DEFAULT_HEIGHT_CM = 152.0      # fallback if user doesn't provide height
MAX_DIM = 1280                 # max image side after downscale to save memory
BAND_HALF = 5                  # +/- rows sampled around target height for stability

# Load local depth model for fallback thickness if no side image
depth_model = load_depth_model()

# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------

def cm_to_in(x_cm: float) -> float:
    return round(x_cm / 2.54, 2)

def clamp_circumference(width_px, raw_circumference_cm, scale_factor, max_multiplier=2.5):
    """
    Prevent insane outliers. We cap circumference at ~max_multiplier * width_cm.
    """
    width_cm = width_px * scale_factor
    max_reasonable = width_cm * max_multiplier
    return min(raw_circumference_cm, max_reasonable)

def quality_note(meas):
    """
    Attach warnings if proportions look suspicious.
    meas is dict with *_circumference_cm entries.
    """
    notes = []

    chest_circ = meas.get("chest_circumference_cm")
    waist_circ = meas.get("waist_circumference_cm")
    hip_circ   = meas.get("hip_circumference_cm")

    # Hip huge vs chest: usually arm/stance/background contamination
    if chest_circ and hip_circ:
        if hip_circ > (1.4 * chest_circ):
            notes.append(
                "Hip reading may be distorted by stance, arm position, or camera angle. "
                "Stand straight, arms slightly out, feet under hips, and avoid loose fabric."
            )

    # Waist tiny vs chest: usually arm blocking torso or shadow gap
    if chest_circ and waist_circ:
        if waist_circ < (0.6 * chest_circ):
            notes.append(
                "Waist reading may be too small. Make sure arms are not blocking your midsection "
                "and that your torso is clearly visible."
            )

    return notes

def downscale_frame_if_needed(frame):
    """
    Shrink large phone images so they don't blow memory.
    Longest side capped at MAX_DIM.
    """
    if frame is None:
        return frame

    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= MAX_DIM:
        return frame

    scale = MAX_DIM / longest
    new_w = int(w * scale)
    new_h = int(h * scale)

    print(f"[resize] {w}x{h} -> {new_w}x{new_h}")
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame

# -------------------------------------------------------------------
# Depth fallback (used if no side image)
# -------------------------------------------------------------------

def estimate_depth(image):
    """
    Produce a depth map using the local/lightweight model.
    Returns None if model unavailable.
    """
    if depth_model is None:
        return None

    try:
        rgb_norm = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        input_tensor = torch.tensor(rgb_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        # Resize to something like 384x384 for MiDaS_small
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
    Map pixel coordinates (px_x, px_y) in the original image
    into the downscaled depth map and return a depth value.
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
# Geometry / calibration helpers
# -------------------------------------------------------------------

def calibrate_focal_length(image, real_width_cm, detected_width_px):
    """
    Estimate focal length if we detect a known object width.
    """
    if detected_width_px:
        return (detected_width_px * FOCAL_LENGTH) / real_width_cm
    return FOCAL_LENGTH

def detect_reference_object(image):
    """
    Try to detect a 'reference rectangle' (like A4 paper).
    Returns (scale_factor_cm_per_px, focal_length).
    Falls back to heuristics if not found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        focal_length = calibrate_focal_length(image, KNOWN_OBJECT_WIDTH_CM, w)
        scale_factor = KNOWN_OBJECT_WIDTH_CM / w  # cm/px
        return scale_factor, focal_length

    return 0.05, FOCAL_LENGTH  # worst-case guess

def calculate_distance_using_height(landmarks, image_height, user_height_cm):
    """
    Infer real-world scale from user height.
    We measure nose->ankle in pixels.
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

# -------------------------------------------------------------------
# STABLE BAND SCANNERS
# -------------------------------------------------------------------

def get_body_width_band_front(frame, height_px, center_x_ratio, band_half=BAND_HALF):
    """
    FRONT VIEW width with stability.
    - We scan multiple rows around height_px instead of just one row.
    - For each row we look left/right from center_x_ratio until we hit body.
    - We take the median width across the band.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    h, w = frame.shape[:2]
    center_x = int(center_x_ratio * w)

    widths = []

    y_start = max(0, height_px - band_half)
    y_end   = min(h - 1, height_px + band_half)

    for y in range(y_start, y_end + 1):
        row = thresh[y, :]

        # walk left from center
        left_edge = center_x
        for i in range(center_x, 0, -1):
            if row[i] == 0:  # black pixel = foreground/body
                left_edge = i
                break

        # walk right from center
        right_edge = center_x
        for j in range(center_x, w):
            if row[j] == 0:
                right_edge = j
                break

        width_px = right_edge - left_edge

        # enforce sane minimum: at least 10% of frame width
        min_width = 0.1 * w
        if width_px < min_width:
            width_px = min_width

        widths.append(width_px)

    if not widths:
        return 0

    widths.sort()
    median_width = widths[len(widths)//2]
    return median_width

def get_body_thickness_band_side(frame, height_px, band_half=BAND_HALF):
    """
    SIDE VIEW thickness with stability.
    - We scan multiple rows around height_px.
    - For each row, find first and last body pixel from left/right edges (front/back).
    - Take median thickness in pixels.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    h, w = frame.shape[:2]
    thicknesses = []

    y_start = max(0, height_px - band_half)
    y_end   = min(h - 1, height_px + band_half)

    for y in range(y_start, y_end + 1):
        row = thresh[y, :]

        # first body pixel from the left
        left_edge = None
        for i in range(w):
            if row[i] == 0:
                left_edge = i
                break

        # first body pixel from the right
        right_edge = None
        for j in range(w - 1, -1, -1):
            if row[j] == 0:
                right_edge = j
                break

        if left_edge is None or right_edge is None:
            thick_px = 0.05 * w  # fallback
        else:
            thick_px = right_edge - left_edge

        # min sanity: at least 5% of frame width
        min_thick = 0.05 * w
        if thick_px < min_thick:
            thick_px = min_thick

        thicknesses.append(thick_px)

    if not thicknesses:
        return 0

    thicknesses.sort()
    median_thickness = thicknesses[len(thicknesses)//2]
    return median_thickness

# -------------------------------------------------------------------
# Circumference math
# -------------------------------------------------------------------

def ellipse_circumference_from_axes(width_cm, thickness_cm):
    """
    Approximate body circumference treating cross-section as an ellipse.
    """
    a = width_cm / 2.0
    b = thickness_cm / 2.0
    return 2.0 * np.pi * np.sqrt((a**2 + b**2) / 2.0)

# -------------------------------------------------------------------
# Measurement builder
# -------------------------------------------------------------------

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
    Core measurement logic:
    - width from front (band median)
    - thickness from side (band median), else fallback depth/heuristic
    - assemble final measurements with sanity clamps and notes
    """

    lm = results_front.pose_landmarks.landmark

    # Optionally refine scale factor using height again (helps stability)
    if user_height_cm:
        _, scale_factor = calculate_distance_using_height(lm, front_h, user_height_cm)

    def px_to_cm(px):
        return round(px * scale_factor, 2)

    def safe_thickness_cm_from_side(y_ratio):
        """
        If we have a side image, measure "front↔back" thickness using a band.
        Otherwise None.
        """
        if side_frame is None:
            return None
        side_h, side_w = side_frame.shape[:2]
        y_px_side = int(y_ratio * side_h)

        thick_px = get_body_thickness_band_side(
            side_frame,
            y_px_side,
            band_half=BAND_HALF
        )
        return thick_px * scale_factor  # cm

    def fallback_thickness_cm(y_ratio, center_x_ratio):
        """
        If no side frame, try depth map. Otherwise heuristic ~0.7 * width.
        We'll sample depth at (center_x_ratio, y_ratio).
        """
        y_px_front = int(y_ratio * front_h)
        x_px_front = int(center_x_ratio * front_w)

        if depth_map_front is not None:
            d_center = sample_depth_at(depth_map_front, x_px_front, y_px_front, front_w, front_h)
            if d_center is not None:
                max_depth = float(np.max(depth_map_front))
                depth_ratio = 1.0 + 0.5 * (1.0 - d_center / max_depth)
                # We'll return a function of width later (because we don't know width yet here)
                return ("needs_width_scale", depth_ratio)

        # ultimate fallback: 0.7 * width once we know width
        return ("needs_width_scale", 1.0)

    # Helper to build circumference for each band slice (chest/waist/hip/etc.)
    def region_circumference(y_ratio, center_x_ratio, width_pad_factor, clamp_mult=2.5):
        """
        1. Get width in FRONT at that y band using median-of-band.
        2. Apply a small pad to compensate under-detection.
        3. Convert to cm.
        4. Get thickness either from SIDE frame (preferred)
           or from depth fallback (which may depend on width).
        5. Build ellipse circumference and clamp.
        """
        y_px_front = int(y_ratio * front_h)

        # --- width from front (median band) ---
        width_px_front = get_body_width_band_front(
            front_frame,
            y_px_front,
            center_x_ratio,
            band_half=BAND_HALF
        )

        # apply pad factor
        width_px_front *= width_pad_factor

        width_cm = width_px_front * scale_factor

        # --- thickness from side if available ---
        thickness_cm = safe_thickness_cm_from_side(y_ratio)

        # if no side, fallback to depth/heuristic
        if thickness_cm is None:
            fb = fallback_thickness_cm(y_ratio, center_x_ratio)
            if isinstance(fb, tuple) and fb[0] == "needs_width_scale":
                depth_ratio = fb[1]
                thickness_cm = width_cm * 0.7 * depth_ratio
            else:
                # shouldn't really happen, but keep a guard
                thickness_cm = width_cm * 0.7

        # ellipse around torso
        raw_circ_cm = ellipse_circumference_from_axes(width_cm, thickness_cm)

        # clamp
        circ_cm = clamp_circumference(
            width_px_front,
            raw_circ_cm,
            scale_factor,
            max_multiplier=clamp_mult
        )

        return {
            "width_px": width_px_front,
            "width_cm": width_cm,
            "circumference_cm": circ_cm
        }

    # landmarks we'll reuse
    left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    nose = lm[mp_pose.PoseLandmark.NOSE.value]
    left_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
    left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]

    # SHOULDER WIDTH (straight landmark distance with small pad)
    shoulder_width_px = abs(
        left_shoulder.x * front_w - right_shoulder.x * front_w
    )
    shoulder_width_px *= 1.1  # slight outward pad
    shoulder_width_cm = px_to_cm(shoulder_width_px)

    # CHEST:
    # slice ~15% down from shoulder->hip
    chest_y_ratio = left_shoulder.y + (left_hip.y - left_shoulder.y) * 0.15
    center_x_ratio_chest = (left_shoulder.x + right_shoulder.x) / 2.0
    chest_data = region_circumference(
        y_ratio=chest_y_ratio,
        center_x_ratio=center_x_ratio_chest,
        width_pad_factor=1.15,     # same pad logic as before
        clamp_mult=2.5
    )
    chest_width_cm = round(chest_data["width_cm"], 2)
    chest_circ_cm = round(chest_data["circumference_cm"], 2)

    # WAIST:
    # slice ~35% down from shoulder->hip
    waist_y_ratio = left_shoulder.y + (left_hip.y - left_shoulder.y) * 0.35
    center_x_ratio_waist = (left_hip.x + right_hip.x) / 2.0
    waist_data = region_circumference(
        y_ratio=waist_y_ratio,
        center_x_ratio=center_x_ratio_waist,
        width_pad_factor=1.16,     # waist pad
        clamp_mult=2.5
    )
    waist_width_cm = round(waist_data["width_cm"], 2)
    waist_circ_cm = round(waist_data["circumference_cm"], 2)

    # HIP:
    # first estimate hip from hip landmarks (with 1.35x bump),
    # then confirm with a band slice ~10% down from hip->knee.
    hip_base_width_px = abs(
        left_hip.x * front_w - right_hip.x * front_w
    ) * 1.35

    hip_y_ratio = left_hip.y + (left_knee.y - left_hip.y) * 0.10
    center_x_ratio_hip = (left_hip.x + right_hip.x) / 2.0
    hip_data = region_circumference(
        y_ratio=hip_y_ratio,
        center_x_ratio=center_x_ratio_hip,
        width_pad_factor=1.0,      # we'll merge with hip_base below
        clamp_mult=2.5
    )

    # merge hip landmark-based width bump with band-detected width
    merged_hip_width_px = max(hip_base_width_px, hip_data["width_px"])
    hip_width_cm = merged_hip_width_px * scale_factor

    # recalc hip circumference using merged width and (side/depth) thickness logic:
    # we need to rebuild thickness for hip_y_ratio using merged width
    # replicate fallback thickness logic:
    if side_frame is not None:
        # use side thickness band
        side_h, side_w = side_frame.shape[:2]
        hip_y_px_side = int(hip_y_ratio * side_h)
        hip_thick_px = get_body_thickness_band_side(
            side_frame,
            hip_y_px_side,
            band_half=BAND_HALF
        )
        hip_thickness_cm = hip_thick_px * scale_factor
    else:
        # fallback to depth ratio
        fb = None
        if depth_map_front is not None:
            hip_y_px_front = int(hip_y_ratio * front_h)
            hip_x_px_front = int(center_x_ratio_hip * front_w)
            d_center = sample_depth_at(depth_map_front, hip_x_px_front, hip_y_px_front, front_w, front_h)
            if d_center is not None:
                max_depth = float(np.max(depth_map_front))
                depth_ratio = 1.0 + 0.5 * (1.0 - d_center / max_depth)
                fb = depth_ratio
        if fb is None:
            fb = 1.0
        hip_thickness_cm = hip_width_cm * 0.7 * fb

    hip_raw_circ_cm = ellipse_circumference_from_axes(
        hip_width_cm,
        hip_thickness_cm
    )
    hip_circ_cm = clamp_circumference(
        merged_hip_width_px,
        hip_raw_circ_cm,
        scale_factor,
        max_multiplier=2.5
    )
    hip_circ_cm = round(hip_circ_cm, 2)

    # NECK:
    neck_width_px = abs(
        nose.x * front_w - left_ear.x * front_w
    ) * 2.0
    neck_width_cm = px_to_cm(neck_width_px)
    neck_thickness_cm = neck_width_cm * 0.7  # heuristic
    neck_raw_circ_cm = ellipse_circumference_from_axes(
        neck_width_cm,
        neck_thickness_cm
    )
    neck_circ_cm = clamp_circumference(
        neck_width_px,
        neck_raw_circ_cm,
        scale_factor,
        max_multiplier=2.5
    )
    neck_circ_cm = round(neck_circ_cm, 2)

    # ARM LENGTH (shoulder -> wrist)
    sleeve_length_px = abs(
        left_shoulder.y * front_h - left_wrist.y * front_h
    )
    arm_length_cm = px_to_cm(sleeve_length_px)

    # SHIRT LENGTH (shoulder -> hip, padded)
    shirt_length_px = abs(
        left_shoulder.y * front_h - left_hip.y * front_h
    ) * 1.2
    shirt_length_cm = px_to_cm(shirt_length_px)

    # THIGH:
    thigh_y_ratio = left_hip.y + (left_knee.y - left_hip.y) * 0.20
    thigh_y_px_front = int(thigh_y_ratio * front_h)

    # baseline thigh width: ~0.5 of hip, padded 1.2
    thigh_width_px_guess = (merged_hip_width_px * 0.5 * 1.2)

    # refine using band scan around thigh line in the FRONT
    # note: center_x for thigh = a little inward from left hip to avoid outer crease
    thigh_center_ratio = left_hip.x * 0.9
    thigh_width_band_px = get_body_width_band_front(
        front_frame,
        thigh_y_px_front,
        thigh_center_ratio,
        band_half=BAND_HALF
    )
    if 0 < thigh_width_band_px < merged_hip_width_px:
        thigh_width_px = thigh_width_band_px
    else:
        thigh_width_px = thigh_width_px_guess

    thigh_width_cm = thigh_width_px * scale_factor

    # thickness for thigh
    if side_frame is not None:
        side_h, side_w = side_frame.shape[:2]
        thigh_y_px_side = int(thigh_y_ratio * side_h)
        thigh_thick_px = get_body_thickness_band_side(
            side_frame,
            thigh_y_px_side,
            band_half=BAND_HALF
        )
        thigh_thickness_cm = thigh_thick_px * scale_factor
    else:
        thigh_thickness_cm = thigh_width_cm * 0.7

    thigh_raw_circ_cm = ellipse_circumference_from_axes(
        thigh_width_cm,
        thigh_thickness_cm
    )
    thigh_circ_cm = clamp_circumference(
        thigh_width_px,
        thigh_raw_circ_cm,
        scale_factor,
        max_multiplier=2.5
    )
    thigh_circ_cm = round(thigh_circ_cm, 2)

    # TROUSER LENGTH (hip -> ankle)
    trouser_length_px = abs(
        left_hip.y * front_h - left_ankle.y * front_h
    )
    trouser_length_cm = px_to_cm(trouser_length_px)

    # -------------------------------------------------------------------
    # Package measurements
    # -------------------------------------------------------------------

    measurements = {
        "shoulder_width_cm": round(shoulder_width_cm, 2),

        "chest_circumference_cm": chest_circ_cm,
        "chest_circumference_in": cm_to_in(chest_circ_cm),
        "chest_width_cm": chest_width_cm,

        "waist_circumference_cm": waist_circ_cm,
        "waist_circumference_in": cm_to_in(waist_circ_cm),
        "waist_width_cm": waist_width_cm,

        "hip_circumference_cm": hip_circ_cm,
        "hip_circumference_in": cm_to_in(hip_circ_cm),
        "hip_width_cm": round(hip_width_cm, 2),

        "neck_circumference_cm": neck_circ_cm,
        "neck_circumference_in": cm_to_in(neck_circ_cm),
        "neck_width_cm": round(neck_width_cm, 2),

        "arm_length_cm": round(arm_length_cm, 2),
        "shirt_length_cm": round(shirt_length_cm, 2),

        "thigh_circumference_cm": thigh_circ_cm,
        "thigh_circumference_in": cm_to_in(thigh_circ_cm),
        "thigh_width_cm": round(thigh_width_cm, 2),

        "trouser_length_cm": round(trouser_length_cm, 2),
    }

    # quality guidance notes
    circ_for_qc = {
        "chest_circumference_cm": measurements["chest_circumference_cm"],
        "waist_circumference_cm": measurements["waist_circumference_cm"],
        "hip_circumference_cm": measurements["hip_circumference_cm"],
    }
    notes_list = quality_note(circ_for_qc)
    if notes_list:
        measurements["notes"] = notes_list

    # final sanity clamp: hips shouldn't be crazy vs chest
    if (
        "hip_circumference_cm" in measurements and
        "chest_circumference_cm" in measurements
    ):
        chest_val = measurements["chest_circumference_cm"]
        hip_val   = measurements["hip_circumference_cm"]
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
# Pose validation for front image
# -------------------------------------------------------------------

def validate_front_image(image_np):
    """
    Basic validation for front image:
    - Full body visible-ish
    - Not just a selfie crop
    - Key joints are in-frame
    """
    try:
        rgb_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_np.shape[:2]

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

        # reject selfie-close shots
        nose_pt = lm[mp_holistic.PoseLandmark.NOSE]
        l_sh = lm[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        r_sh = lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

        shoulder_width_px = abs(l_sh.x - r_sh.x) * img_w
        head_to_shoulder_px = abs(l_sh.y - nose_pt.y) * img_h
        # if shoulders look narrower than head height, you're too close / cropped
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
    POST multipart/form-data:
      - front (REQUIRED): full-body front photo
      - left_side (OPTIONAL): full-body side photo, same camera distance
      - height_cm (OPTIONAL but strongly recommended): user's height in cm
    """

    if "front" not in request.files:
        return jsonify({"error": "Missing 'front' image."}), 400

    # read front for validation
    front_file = request.files["front"]
    front_np_raw = np.frombuffer(front_file.read(), np.uint8)
    front_file.seek(0)

    front_frame_for_validation = cv2.imdecode(front_np_raw, cv2.IMREAD_COLOR)
    front_frame_for_validation = downscale_frame_if_needed(front_frame_for_validation)

    is_valid, msg = validate_front_image(front_frame_for_validation)
    if not is_valid:
        return jsonify({
            "error": msg,
            "pose": "front",
            "code": "INVALID_POSE"
        }), 400

    # parse height
    user_height_cm = request.form.get("height_cm")
    try:
        user_height_cm = float(user_height_cm) if user_height_cm else DEFAULT_HEIGHT_CM
    except ValueError:
        user_height_cm = DEFAULT_HEIGHT_CM

    # decode both front and left_side fully (downscaled)
    frames = {}
    pose_results_map = {}

    for pose_name in ["front", "left_side"]:
        if pose_name not in request.files:
            continue

        img_file = request.files[pose_name]
        data = np.frombuffer(img_file.read(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        frame = downscale_frame_if_needed(frame)

        frames[pose_name] = frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results_map[pose_name] = holistic.process(rgb)

    # derive scale factor from front pose (best case),
    # otherwise from reference object fallback
    scale_factor = None
    focal_length = FOCAL_LENGTH

    if "front" in pose_results_map and pose_results_map["front"].pose_landmarks:
        lm_front = pose_results_map["front"].pose_landmarks.landmark
        _, scale_factor = calculate_distance_using_height(
            lm_front,
            frames["front"].shape[0],
            user_height_cm
        )
    else:
        scale_factor, focal_length = detect_reference_object(frames["front"])

    # prepare depth map fallback if no side frame
    depth_map_front = None
    if "front" in frames:
        depth_map_front = estimate_depth(frames["front"])

    # compute final measurements
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
        return jsonify({
            "error": "Unable to detect body landmarks in front image. Please retake.",
            "code": "NO_LANDMARKS"
        }), 400

    debug_info = {
        "scale_factor": float(scale_factor) if scale_factor else None,
        "focal_length": float(focal_length),
        "user_height_cm": float(user_height_cm),
        "front_shape": frames["front"].shape if "front" in frames else None,
        "side_shape": frames["left_side"].shape if "left_side" in frames else None
    }

    return jsonify({
        "measurements": measurements,
        "debug_info": debug_info
    })

# -------------------------------------------------------------------
# Local dev (Render will use gunicorn instead)
# -------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # For local debugging only. In production you're using gunicorn in Docker.
    app.run(host="0.0.0.0", port=port)
