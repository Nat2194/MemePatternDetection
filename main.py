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

def get_angle(p1, p2, p3):
    """Calculates the 2D interior angle (in degrees) at vertex p2."""
    # Use atan2 to get the absolute angle of the two connecting lines
    angle = math.degrees(math.atan2(p3.y - p2.y, p3.x - p2.x) - 
                         math.atan2(p1.y - p2.y, p1.x - p2.x))
    # Normalize to an internal joint angle (0 to 180)
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle

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

def extract_body_state(results, brow_low_thresh, brow_high_thresh, eye_closed_thresh):
    state = {
        "is_smiling": False,
        "is_mouth_open": False,
        "left_is_fist": False,
        "right_is_fist": False,
        "is_flexing": False,
        "right_is_pointing_up": is_pointing_up(results.right_hand_landmarks),
        "right_ok_sign": is_ok_sign(results.right_hand_landmarks),
        "index_fingers_touching": False,
        "are_eyebrows_lowered": False,
        "are_eyebrows_raised": False,
        "is_one_eyebrow_raised": False,
        "is_looking_left": False,
        "is_looking_right": False,
        "is_shushing": False,
        "are_hands_clasped": False,
        "are_eyes_closed": False,
        "RAW_LEFT_BROW": 0.0,
        "RAW_RIGHT_BROW": 0.0,
        "RAW_SMILE": 0.0,
        "RAW_L_FIST": "Not Visible",
        "RAW_R_FIST": "Not Visible",
        "RAW_FLEX": "Not Visible",
        "RAW_YAW": 0.0,
        "RAW_EYES": 0.0,
        "RAW_SHUSH": "No Hand",
        "RAW_CLASP": "No Hands"
    }

    face_width = 1.0 
    if results.face_landmarks:
        face = results.face_landmarks.landmark
        face_width = get_dist(face[234], face[454])
        
        # --- HEAD ROTATION (YAW) ---
        yaw_ratio = (face[234].z - face[454].z) / face_width
        state["RAW_YAW"] = round(yaw_ratio, 3)
        state["is_looking_left"] = yaw_ratio < -0.4
        state["is_looking_right"] = yaw_ratio > 0.4
        
        # --- MOUTH & SMILE ---
        state["is_mouth_open"] = (get_dist(face[13], face[14]) / face_width) > 0.15
        smile_ratio = get_dist(face[78], face[308]) / face_width
        state["RAW_SMILE"] = round(smile_ratio, 3)
        state["is_smiling"] = smile_ratio > 0.38

        # --- EYEBROWS ---
        left_brow_dist = get_dist(face[159], face[52]) / face_width
        right_brow_dist = get_dist(face[386], face[282]) / face_width
        state["RAW_LEFT_BROW"] = round(left_brow_dist, 3)
        state["RAW_RIGHT_BROW"] = round(right_brow_dist, 3)
        state["are_eyebrows_lowered"] = left_brow_dist < brow_low_thresh and right_brow_dist < brow_low_thresh
        state["are_eyebrows_raised"] = left_brow_dist > brow_high_thresh and right_brow_dist > brow_high_thresh
        brow_difference = abs(left_brow_dist - right_brow_dist)
        state["is_one_eyebrow_raised"] = brow_difference > 0.015 and (left_brow_dist > brow_high_thresh - 0.005 or right_brow_dist > brow_high_thresh - 0.005)

        # --- EYES CLOSED ---
        left_eye_dist = get_dist(face[159], face[145]) / face_width
        right_eye_dist = get_dist(face[386], face[374]) / face_width
        avg_eye = (left_eye_dist + right_eye_dist) / 2.0
        state["RAW_EYES"] = round(avg_eye, 3)
        state["are_eyes_closed"] = avg_eye < eye_closed_thresh

        # --- SHUSHING ---
        mouth_top = face[13]
        if results.left_hand_landmarks:
            lh_shush = get_dist(results.left_hand_landmarks.landmark[8], mouth_top) / face_width
            state["RAW_SHUSH"] = round(lh_shush, 3)
            if lh_shush < 0.3: state["is_shushing"] = True
        if results.right_hand_landmarks:
            rh_shush = get_dist(results.right_hand_landmarks.landmark[8], mouth_top) / face_width
            state["RAW_SHUSH"] = round(rh_shush, 3)
            if rh_shush < 0.3: state["is_shushing"] = True

    # --- LEFT FIST LOGIC ---
    if results.left_hand_landmarks:
        state["left_is_fist"] = is_fist(results.left_hand_landmarks)
        state["RAW_L_FIST"] = "Mesh OK"
    elif results.pose_landmarks:
        pose = results.pose_landmarks.landmark
        if pose[15].visibility > 0.2 and pose[19].visibility > 0.2:
            fist_ratio = get_dist(pose[15], pose[19]) / face_width
            state["RAW_L_FIST"] = round(fist_ratio, 3)
            state["left_is_fist"] = fist_ratio < 1.2 

    # --- RIGHT FIST LOGIC ---
    if results.right_hand_landmarks:
        state["right_is_fist"] = is_fist(results.right_hand_landmarks)
        state["RAW_R_FIST"] = "Mesh OK"
    elif results.pose_landmarks:
        pose = results.pose_landmarks.landmark
        if pose[16].visibility > 0.2 and pose[20].visibility > 0.2:
            fist_ratio = get_dist(pose[16], pose[20]) / face_width
            state["RAW_R_FIST"] = round(fist_ratio, 3)
            state["right_is_fist"] = fist_ratio < 1.2

    # --- ARM FLEXING (MUSCLE) ---
    if results.pose_landmarks:
        pose = results.pose_landmarks.landmark
        
        # Right Arm (12=Shoulder, 14=Elbow, 16=Wrist)
        r_shoulder, r_elbow, r_wrist = pose[12], pose[14], pose[16]
        # Left Arm (11=Shoulder, 13=Elbow, 15=Wrist)
        l_shoulder, l_elbow, l_wrist = pose[11], pose[13], pose[15]
        
        # 1. Check Right Arm
        if r_shoulder.visibility > 0.2 and r_elbow.visibility > 0.2 and r_wrist.visibility > 0.2:
            angle_r = get_angle(r_shoulder, r_elbow, r_wrist)
            # MediaPipe Y is inverted (0 is at the top of the screen), so < means "higher than"
            is_high_r = r_wrist.y < r_shoulder.y
            
            state["RAW_FLEX"] = f"R: {int(angle_r)}deg | High: {is_high_r}"
            
            if angle_r < 90 and is_high_r:
                state["is_flexing"] = True
                state["RAW_FLEX"] = f"FLEX RIGHT! ({int(angle_r)}deg)"

        # 2. Check Left Arm
        if l_shoulder.visibility > 0.2 and l_elbow.visibility > 0.2 and l_wrist.visibility > 0.2:
            angle_l = get_angle(l_shoulder, l_elbow, l_wrist)
            is_high_l = l_wrist.y < l_shoulder.y
            
            # Update debug text if left arm is being moved (angle < 150)
            if angle_l < 150: 
                state["RAW_FLEX"] = f"L: {int(angle_l)}deg | High: {is_high_l}"
                
            if angle_l < 90 and is_high_l:
                state["is_flexing"] = True
                state["RAW_FLEX"] = f"FLEX LEFT! ({int(angle_l)}deg)"

    # --- HANDS CLASPED ---
    if results.left_hand_landmarks and results.right_hand_landmarks:
        lw = results.left_hand_landmarks.landmark[0]
        rw = results.right_hand_landmarks.landmark[0]
        clasp_dist = get_dist(lw, rw) / face_width
        state["RAW_CLASP"] = round(clasp_dist, 3)
        if clasp_dist < 0.5: state["are_hands_clasped"] = True

    # --- INDEX FINGERS TOUCHING ---
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
eye_history = []

# Default values (these will be overwritten by calibration)
dynamic_brow_low = 0.105
dynamic_brow_high = 0.135
dynamic_eye_closed = 0.082 # NEW: Default eye threshold

BROW_RAISE_OFFSET = 0.015  
BROW_LOWER_OFFSET = 0.005  
EYE_CLOSED_OFFSET = 0.015  # NEW: The gap between open and closed eyes

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
            
            # Record Eyebrows
            left_brow = get_dist(face[159], face[52]) / face_width
            right_brow = get_dist(face[386], face[282]) / face_width
            brow_history.append((left_brow + right_brow) / 2)
            
            # Record Eyes
            left_eye = get_dist(face[159], face[145]) / face_width
            right_eye = get_dist(face[386], face[374]) / face_width
            eye_history.append((left_eye + right_eye) / 2.0)
            
            calibration_frames_collected += 1
            
            # Display calibration warning overlay
            overlay_text = f"CALIBRATING: Keep neutral face... {calibration_frames_collected}/{CALIBRATION_TARGET}"
            cv2.putText(frame, overlay_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Once we have enough frames, set the thresholds
            if calibration_frames_collected >= CALIBRATION_TARGET:
                # Calculate Brow Thresholds
                avg_resting_brow = sum(brow_history) / len(brow_history)
                dynamic_brow_low = avg_resting_brow - BROW_LOWER_OFFSET
                dynamic_brow_high = avg_resting_brow + BROW_RAISE_OFFSET
                
                # Calculate Eye Thresholds
                avg_resting_eye = sum(eye_history) / len(eye_history)
                dynamic_eye_closed = avg_resting_eye - EYE_CLOSED_OFFSET
                
                is_calibrated = True
                print(f"Calibration Complete!")
                print(f"Brow Base: {avg_resting_brow:.3f} | Low: {dynamic_brow_low:.3f} | High: {dynamic_brow_high:.3f}")
                print(f"Eye Base: {avg_resting_eye:.3f} | Closed Thresh: {dynamic_eye_closed:.3f}")
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

    # Extract state using the dynamically calibrated variables
    body_state = extract_body_state(results, dynamic_brow_low, dynamic_brow_high, dynamic_eye_closed)

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
    
    # Draw the Debug Overlay (Split into Two Columns)
    y_bool = 80
    y_raw = 80
    
    for key, value in body_state.items():
        # Put RAW variables in a second column on the right (in yellow)
        if key.startswith("RAW_"):
            cv2.putText(frame, f"{key}: {value}", (320, y_raw), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            y_raw += 25
        # Put Boolean True/False variables on the left (Green/Red)
        else:
            text_color = (0, 255, 0) if value else (0, 0, 255)
            cv2.putText(frame, f"{key}: {value}", (20, y_bool), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            y_bool += 25

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