import math

def get_dist(p1, p2):
    """Calculates the true 3D distance between two points."""
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

def update_state(results, state, face_width):
    """Detects if a hand index finger is placed near the mouth for shushing."""
    state["is_shushing"] = False
    state["RAW_SHUSH"] = "No Hand"

    # Face landmark 13 is the center top lip
    if results.face_landmarks:
        face = results.face_landmarks.landmark
        mouth_top = face[13]
        
        # Check Left Hand
        if results.left_hand_landmarks:
            lh_shush = get_dist(results.left_hand_landmarks.landmark[8], mouth_top) / face_width
            state["RAW_SHUSH"] = round(lh_shush, 3)
            if lh_shush < 0.3:
                state["is_shushing"] = True
                
        # Check Right Hand (overrides debug if active)
        if results.right_hand_landmarks:
            rh_shush = get_dist(results.right_hand_landmarks.landmark[8], mouth_top) / face_width
            state["RAW_SHUSH"] = round(rh_shush, 3)
            if rh_shush < 0.3:
                state["is_shushing"] = True

    return state