import cv2
import mediapipe as mp
import math
import os
import glob
import json

# ==========================================
# 1. SETUP & AUTO-LOAD
# ==========================================
try:
    with open('rules.json', 'r') as f:
        MEME_RULES = json.load(f)
    print("Successfully loaded rules.json")
except Exception as e:
    print(f"Error loading rules.json: {e}")
    MEME_RULES = {}

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4)
mp_draw = mp.solutions.drawing_utils
green_spec = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

memes = {}
for filepath in glob.glob("memes/*.*"):
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    if ext.lower() in ['.jpg', '.jpeg', '.png']:
        memes[name] = cv2.imread(filepath)
        print(f"Loaded meme image: {name}")

# ==========================================
# 2. FEATURE EXTRACTION HELPERS
# ==========================================
def get_dist(p1, p2):
    """Calculates the true 3D distance to ignore head rotation."""
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

def is_fist(hand_landmarks):
    if not hand_landmarks: return False
    wrist = hand_landmarks.landmark[0]
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17] # The base knuckles
    
    # A finger is curled into a fist if its tip is physically closer to the wrist than its base knuckle is
    return all(get_dist(hand_landmarks.landmark[t], wrist) < get_dist(hand_landmarks.landmark[m], wrist) for t, m in zip(tips, mcps))

def is_open(hand_landmarks):
    if not hand_landmarks: return False
    tips, knuckles = [8, 12, 16, 20], [6, 10, 14, 18]
    return all(hand_landmarks.landmark[t].y < hand_landmarks.landmark[k].y for t, k in zip(tips, knuckles))

def is_ok_sign(hand_landmarks):
    if not hand_landmarks: return False
    is_pinched = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[8]) < 0.05
    others_open = all(hand_landmarks.landmark[t].y < hand_landmarks.landmark[k].y for t, k in zip([12, 16, 20], [10, 14, 18]))
    return is_pinched and others_open

def is_pointing_up(hand_landmarks):
    if not hand_landmarks: return False
    index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    others_down = all(hand_landmarks.landmark[t].y > hand_landmarks.landmark[k].y for t, k in zip([12, 16, 20], [10, 14, 18]))
    return index_up and others_down

# Notice we now pass the dynamic thresholds into this function!
def extract_body_state(results, brow_low_thresh, brow_high_thresh):
    state = {
        "is_smiling": False,
        "is_mouth_open": False,
        "left_is_fist": False,
        "right_is_fist": False,
        "right_is_pointing_up": is_pointing_up(results.right_hand_landmarks),
        "right_ok_sign": is_ok_sign(results.right_hand_landmarks),
        "index_fingers_touching": False,
        "are_eyebrows_lowered": False,
        "are_eyebrows_raised": False,
        "is_one_eyebrow_raised": False,
        "RAW_LEFT_BROW": 0.0,
        "RAW_RIGHT_BROW": 0.0,
        "RAW_SMILE": 0.0,
        "RAW_L_FIST": "Not Visible",
        "RAW_R_FIST": "Not Visible"
    }

    face_width = 1.0 
    if results.face_landmarks:
        face = results.face_landmarks.landmark
        face_width = get_dist(face[234], face[454])
        
        state["is_mouth_open"] = (get_dist(face[13], face[14]) / face_width) > 0.15
        
        smile_ratio = get_dist(face[78], face[308]) / face_width
        state["RAW_SMILE"] = round(smile_ratio, 3)
        state["is_smiling"] = smile_ratio > 0.33

        left_brow_dist = get_dist(face[159], face[52]) / face_width
        right_brow_dist = get_dist(face[386], face[282]) / face_width

        state["RAW_LEFT_BROW"] = round(left_brow_dist, 3)
        state["RAW_RIGHT_BROW"] = round(right_brow_dist, 3)

        state["are_eyebrows_lowered"] = left_brow_dist < brow_low_thresh and right_brow_dist < brow_low_thresh
        state["are_eyebrows_raised"] = left_brow_dist > brow_high_thresh and right_brow_dist > brow_high_thresh
        
        brow_difference = abs(left_brow_dist - right_brow_dist)
        state["is_one_eyebrow_raised"] = brow_difference > 0.015 and (left_brow_dist > brow_high_thresh - 0.005 or right_brow_dist > brow_high_thresh - 0.005)

    # LEFT FIST LOGIC
    if results.left_hand_landmarks:
        state["left_is_fist"] = is_fist(results.left_hand_landmarks)
        state["RAW_L_FIST"] = "Mesh OK"
    elif results.pose_landmarks:
        pose = results.pose_landmarks.landmark
        if pose[15].visibility > 0.5 and pose[19].visibility > 0.5:
            fist_ratio = get_dist(pose[15], pose[19]) / face_width
            state["RAW_L_FIST"] = round(fist_ratio, 3)
            state["left_is_fist"] = fist_ratio < 1.2

    # RIGHT FIST LOGIC
    if results.right_hand_landmarks:
        state["right_is_fist"] = is_fist(results.right_hand_landmarks)
        state["RAW_R_FIST"] = "Mesh OK"
    elif results.pose_landmarks:
        pose = results.pose_landmarks.landmark
        if pose[16].visibility > 0.5 and pose[20].visibility > 0.5:
            fist_ratio = get_dist(pose[16], pose[20]) / face_width
            state["RAW_R_FIST"] = round(fist_ratio, 3)
            state["right_is_fist"] = fist_ratio < 1.2

    lh = results.left_hand_landmarks.landmark if results.left_hand_landmarks else None
    rh = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None
    if lh and rh:
        state["index_fingers_touching"] = get_dist(lh[8], rh[8]) < 0.05

    return state

# ==========================================
# 3. MAIN APPLICATION LOOP
# ==========================================
cap = cv2.VideoCapture(0)

# Variables to handle Auto-Calibration
is_calibrated = False
calibration_frames_collected = 0
CALIBRATION_TARGET = 30
brow_history = []

# Default values (these will be overwritten by calibration)
dynamic_brow_low = 0.105
dynamic_brow_high = 0.135

BROW_RAISE_OFFSET = 0.015  # Keep this as is, since raised works perfectly!
BROW_LOWER_OFFSET = 0.005  # Adjust this one to tune your frown sensitivity

while True:
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    # Always draw the green mesh first so you can see tracking immediately
    if results.face_landmarks:
        mp_draw.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, green_spec, green_spec)
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, green_spec, green_spec)
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, green_spec, green_spec)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, green_spec, green_spec)

    # ==========================================
    # AUTO-CALIBRATION PHASE
    # ==========================================
    if not is_calibrated:
        if results.face_landmarks:
            face = results.face_landmarks.landmark
            face_width = get_dist(face[234], face[454])
            left_brow = get_dist(face[159], face[52]) / face_width
            right_brow = get_dist(face[386], face[282]) / face_width
            
            # Record the average of both eyebrows
            brow_history.append((left_brow + right_brow) / 2)
            calibration_frames_collected += 1
            
            # Display calibration warning overlay
            overlay_text = f"CALIBRATING: Keep neutral face... {calibration_frames_collected}/{CALIBRATION_TARGET}"
            cv2.putText(frame, overlay_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Once we have enough frames, set the thresholds
            if calibration_frames_collected >= CALIBRATION_TARGET:
                avg_resting_brow = sum(brow_history) / len(brow_history)
                dynamic_brow_low = avg_resting_brow - BROW_LOWER_OFFSET
                dynamic_brow_high = avg_resting_brow + BROW_RAISE_OFFSET
                is_calibrated = True
                print(f"Calibration Complete! Base: {avg_resting_brow:.3f} | Low: {dynamic_brow_low:.3f} | High: {dynamic_brow_high:.3f}")
        else:
            cv2.putText(frame, "CALIBRATING: No face detected!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Meme Pattern Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue  # Skip meme detection until calibration finishes!

    # ==========================================
    # MEME DETECTION PHASE (Post-Calibration)
    # ==========================================
    current_meme = None
    status_text = "Tracking..."

    # Extract state using your new dynamically calibrated variables
    body_state = extract_body_state(results, dynamic_brow_low, dynamic_brow_high)

    for meme_name, conditions in MEME_RULES.items():
        if meme_name not in memes:
            continue
            
        is_match = True
        for state_key, expected_value in conditions.items():
            if body_state.get(state_key) != expected_value:
                is_match = False
                break 
                
        if is_match:
            current_meme = meme_name
            status_text = f"{current_meme.capitalize()} Detected!"
            break 

    # Draw the main status at the top
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Draw the Debug Overlay
    y_offset = 80
    for key, value in body_state.items():
        text_color = (0, 255, 0) if value else (0, 0, 255)
        debug_text = f"{key}: {value}"
        cv2.putText(frame, debug_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        y_offset += 25 

    cv2.imshow('Meme Pattern Detector', frame)

    if current_meme:
        cv2.imshow('Matched Meme', memes[current_meme])
    elif "default" in memes:
        cv2.imshow('Matched Meme', memes["default"])
    else:
        try:
            cv2.destroyWindow('Matched Meme')
        except:
            pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()