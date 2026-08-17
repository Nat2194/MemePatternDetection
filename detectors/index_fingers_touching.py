import math

def get_dist(p1, p2):
    """Calculates the true 3D distance between two points."""
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

def update_state(results, state, face_width):
    """Detects if the left and right index fingertips are touching together."""
    state["index_fingers_touching"] = False
    state["RAW_TOUCH"] = "Missing Hands"

    lh = results.left_hand_landmarks.landmark if results.left_hand_landmarks else None
    rh = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None
    
    if lh and rh:
        # Landmark 8 is the index finger tip
        touch_dist = get_dist(lh[8], rh[8])
        state["RAW_TOUCH"] = round(touch_dist, 3)
        
        if touch_dist < 0.05:
            state["index_fingers_touching"] = True

    return state