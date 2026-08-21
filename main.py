import cv2
import mediapipe as mp
import math
import os
import json
import importlib.util
import collections

# ==========================================
# 1. DYNAMIC COMPONENT & DETECTOR LOADER
# ==========================================
# Load Priority Index
try:
    with open("rules.json", "r") as f:
        MEME_PRIORITY = json.load(f)
    print("Successfully loaded rules.json (Priority Index)")
except Exception as e:
    print(f"Error loading rules.json: {e}")
    MEME_PRIORITY = []

MEME_RULES = {}
memes = {}

# A. Load Visual Components (Memes & Rules)
for meme in MEME_PRIORITY:
    meme_dir = os.path.join("modules", meme)
    if not os.path.isdir(meme_dir):
        continue

    rule_path = os.path.join(meme_dir, "rule.json")
    if os.path.exists(rule_path):
        try:
            with open(rule_path, "r") as f:
                content = f.read().strip()
                if content:
                    MEME_RULES[meme] = json.loads(content)
                else:
                    MEME_RULES[meme] = {}
        except Exception as e:
            print(f"Warning: Could not parse rule.json for {meme}: {e}")
            MEME_RULES[meme] = {}

    image_files = [
        f for f in os.listdir(meme_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if image_files:
        memes[meme] = cv2.imread(os.path.join(meme_dir, image_files[0]))
        print(f"Loaded component: {meme}")

# Load Default Image
default_dir = os.path.join("modules", "default")
if os.path.isdir(default_dir):
    default_images = [
        f
        for f in os.listdir(default_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if default_images:
        memes["default"] = cv2.imread(os.path.join(default_dir, default_images[0]))

# B. Load Reusable Detectors
DETECTORS = []
detector_dir = "detectors"
if not os.path.exists(detector_dir):
    os.makedirs(detector_dir)

for filename in os.listdir(detector_dir):
    if filename.endswith(".py"):
        mod_name = filename[:-3]
        filepath = os.path.join(detector_dir, filename)
        spec = importlib.util.spec_from_file_location(mod_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        DETECTORS.append(module)
        print(f"Loaded detector: {mod_name}")

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.4, min_tracking_confidence=0.4
)
mp_draw = mp.solutions.drawing_utils
green_spec = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)


# ==========================================
# 2. BASE FEATURE EXTRACTION
# ==========================================
def get_dist(p1, p2):
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2 + (p2.z - p1.z) ** 2)


def extract_base_state(results, brow_low_thresh, brow_high_thresh, eye_closed_thresh):
    state = {}
    face_width = 1.0

    if results.face_landmarks:
        face = results.face_landmarks.landmark
        face_width = get_dist(face[234], face[454])

        yaw_ratio = (face[234].z - face[454].z) / face_width
        state["RAW_YAW"] = round(yaw_ratio, 3)
        state["is_looking_left"] = yaw_ratio < -0.4
        state["is_looking_right"] = yaw_ratio > 0.4

        state["is_mouth_open"] = (get_dist(face[13], face[14]) / face_width) > 0.15
        smile_ratio = get_dist(face[78], face[308]) / face_width
        state["RAW_SMILE"] = round(smile_ratio, 3)
        state["is_smiling"] = smile_ratio > 0.35

        left_brow_dist = get_dist(face[159], face[52]) / face_width
        right_brow_dist = get_dist(face[386], face[282]) / face_width
        state["RAW_LEFT_BROW"] = round(left_brow_dist, 3)
        state["RAW_RIGHT_BROW"] = round(right_brow_dist, 3)
        state["are_eyebrows_lowered"] = (
            left_brow_dist < brow_low_thresh and right_brow_dist < brow_low_thresh
        )
        state["are_eyebrows_raised"] = (
            left_brow_dist > brow_high_thresh and right_brow_dist > brow_high_thresh
        )

        brow_difference = abs(left_brow_dist - right_brow_dist)
        state["is_one_eyebrow_raised"] = brow_difference > 0.015 and (
            left_brow_dist > brow_high_thresh - 0.005
            or right_brow_dist > brow_high_thresh - 0.005
        )

        avg_eye = (
            get_dist(face[159], face[145]) / face_width
            + get_dist(face[386], face[374]) / face_width
        ) / 2.0
        state["RAW_EYES"] = round(avg_eye, 3)
        state["are_eyes_closed"] = avg_eye < eye_closed_thresh

    return state, face_width


# ==========================================
# 3. MAIN APPLICATION LOOP
# ==========================================
cap = cv2.VideoCapture(0)

is_calibrated = False
calibration_frames_collected = 0
CALIBRATION_TARGET = 30
brow_history, eye_history = [], []

# Temporal Smoothing Variables (10-frame history)
meme_history = collections.deque(maxlen=10)
stable_meme = None

dynamic_brow_low, dynamic_brow_high, dynamic_eye_closed = 0.105, 0.135, 0.082
BROW_RAISE_OFFSET, BROW_LOWER_OFFSET, EYE_CLOSED_OFFSET = 0.015, 0.005, 0.015

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    if results.face_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            green_spec,
            green_spec,
        )
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            green_spec,
            green_spec,
        )
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            green_spec,
            green_spec,
        )
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            green_spec,
            green_spec,
        )

    # --- CALIBRATION PHASE ---
    if not is_calibrated:
        if results.face_landmarks:
            face = results.face_landmarks.landmark
            face_width = get_dist(face[234], face[454])

            brow_history.append(
                (
                    get_dist(face[159], face[52]) / face_width
                    + get_dist(face[386], face[282]) / face_width
                )
                / 2
            )
            eye_history.append(
                (
                    get_dist(face[159], face[145]) / face_width
                    + get_dist(face[386], face[374]) / face_width
                )
                / 2.0
            )

            calibration_frames_collected += 1
            cv2.putText(
                frame,
                f"CALIBRATING: Keep neutral face... {calibration_frames_collected}/{CALIBRATION_TARGET}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )

            if calibration_frames_collected >= CALIBRATION_TARGET:
                avg_resting_brow = sum(brow_history) / len(brow_history)
                dynamic_brow_low, dynamic_brow_high = (
                    avg_resting_brow - BROW_LOWER_OFFSET,
                    avg_resting_brow + BROW_RAISE_OFFSET,
                )
                dynamic_eye_closed = (
                    sum(eye_history) / len(eye_history)
                ) - EYE_CLOSED_OFFSET
                is_calibrated = True
        else:
            cv2.putText(
                frame,
                "CALIBRATING: No face detected!",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Meme Pattern Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # --- MEME DETECTION PHASE ---
    current_meme = None
    status_text = "Tracking..."

    # 1. Base tracking (Face, Eyes, Eyebrows)
    body_state, current_face_width = extract_base_state(
        results, dynamic_brow_low, dynamic_brow_high, dynamic_eye_closed
    )

    # 2. Run all loaded Detectors (e.g., is_flexing)
    for detector in DETECTORS:
        if hasattr(detector, "update_state"):
            body_state = detector.update_state(results, body_state, current_face_width)

    # 3. Check against loaded Rules
    for meme_name in MEME_PRIORITY:
        if meme_name not in memes or meme_name not in MEME_RULES:
            continue

        is_match = True
        for state_key, expected_value in MEME_RULES[meme_name].items():
            # Force strict evaluation: if the state key doesn't exist, it defaults to False
            actual_value = body_state.get(state_key, False)
            if actual_value != expected_value:
                is_match = False
                break

        if is_match:
            current_meme = meme_name
            break  # We found the raw match for this single frame

    # --- TEMPORAL SMOOTHING (DEBOUNCING) ---
    # 1. Add the current frame's raw detection to our 10-frame history
    meme_history.append(current_meme)

    # 2. If the current raw detection has happened at least 7 times in the last 10 frames, lock it in!
    if meme_history.count(current_meme) >= 7:
        stable_meme = current_meme

    # 3. Update the UI text based on the STABLE meme, not the raw one
    if stable_meme:
        status_text = f"{stable_meme.capitalize()} Detected!"
    else:
        status_text = "Tracking..."

    # --- DEBUG UI ---
    cv2.putText(
        frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
    )
    y_bool, y_raw = 80, 80

    for key, value in body_state.items():
        if key.startswith("RAW_"):
            cv2.putText(
                frame,
                f"{key}: {value}",
                (320, y_raw),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )
            y_raw += 25
        else:
            text_color = (0, 255, 0) if value else (0, 0, 255)
            cv2.putText(
                frame,
                f"{key}: {value}",
                (20, y_bool),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                2,
            )
            y_bool += 25

    cv2.imshow("Meme Pattern Detector", frame)

    # Use the debounced stable_meme to trigger the image popup!
    if stable_meme:
        cv2.imshow("Matched Meme", memes[stable_meme])
    elif "default" in memes:
        cv2.imshow("Matched Meme", memes["default"])
    else:
        try:
            cv2.destroyWindow("Matched Meme")
        except:
            pass

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
