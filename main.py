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
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

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
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def is_fist(hand_landmarks):
    if not hand_landmarks: return False
    tips, knuckles = [8, 12, 16, 20], [6, 10, 14, 18]
    return all(hand_landmarks.landmark[t].y > hand_landmarks.landmark[k].y for t, k in zip(tips, knuckles))

def is_open(hand_landmarks):
    if not hand_landmarks: return False
    tips, knuckles = [8, 12, 16, 20], [6, 10, 14, 18]
    return all(hand_landmarks.landmark[t].y < hand_landmarks.landmark[k].y for t, k in zip(tips, knuckles))

def is_ok_sign(hand_landmarks):
    """Checks if thumb and index are pinched while other fingers are open."""
    if not hand_landmarks: return False
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    
    # 1. Are thumb and index pinched?
    is_pinched = get_dist(thumb, index) < 0.05
    
    # 2. Are the middle, ring, and pinky fingers raised?
    tips, knuckles = [12, 16, 20], [10, 14, 18]
    others_open = all(hand_landmarks.landmark[t].y < hand_landmarks.landmark[k].y for t, k in zip(tips, knuckles))
    
    return is_pinched and others_open

def extract_body_state(results):
    state = {
        "is_smiling": False,
        "is_mouth_open": False,
        "left_is_fist": is_fist(results.left_hand_landmarks),
        "right_is_fist": is_fist(results.right_hand_landmarks),
        "left_is_open": is_open(results.left_hand_landmarks),
        "right_is_open": is_open(results.right_hand_landmarks),
        "right_ok_sign": is_ok_sign(results.right_hand_landmarks),
        "left_ok_sign": is_ok_sign(results.left_hand_landmarks),
        "index_fingers_touching": False,
        "right_index_on_nose": False,
        "right_pinching_eye": False,
        "right_fist_near_mouth": False,
        "hands_rubbing": False
    }

    # Extract Face Features
    if results.face_landmarks:
        face = results.face_landmarks.landmark
        face_width = get_dist(face[234], face[454])
        state["is_mouth_open"] = (get_dist(face[13], face[14]) / face_width) > 0.15
        state["is_smiling"] = (get_dist(face[78], face[308]) / face_width) > 0.45

        # Check Hand-to-Face interactions
        rh = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None
        if rh:
            state["right_index_on_nose"] = get_dist(rh[8], face[1]) < 0.1
            state["right_pinching_eye"] = get_dist(rh[4], rh[8]) < 0.03 and get_dist(rh[8], face[159]) < 0.15
            # Lollipop check: Right hand is a fist and the wrist/palm area (0) is close to the mouth (13)
            state["right_fist_near_mouth"] = state["right_is_fist"] and get_dist(rh[0], face[13]) < 0.25

    # Check Hand-to-Hand interactions
    lh = results.left_hand_landmarks.landmark if results.left_hand_landmarks else None
    rh = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None
    if lh and rh:
        state["index_fingers_touching"] = get_dist(lh[8], rh[8]) < 0.05
        state["hands_rubbing"] = get_dist(lh[9], rh[9]) < 0.15

    return state

# ==========================================
# 3. MAIN APPLICATION LOOP
# ==========================================
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    current_meme = None
    status_text = "Tracking..."

    body_state = extract_body_state(results)

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

    cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Meme Pattern Detector', frame)

    if current_meme:
        cv2.imshow('Matched Meme', memes[current_meme])
    else:
        try:
            cv2.destroyWindow('Matched Meme')
        except:
            pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()