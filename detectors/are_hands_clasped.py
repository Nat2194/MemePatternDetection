import math

def get_dist(p1, p2):
    """Calculates the true 3D distance between two points."""
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

def update_state(results, state, face_width):
    """Detects if both hands are clasped together near the chest/face."""
    state["are_hands_clasped"] = False
    state["RAW_CLASP"] = "No Hands"

    if results.left_hand_landmarks and results.right_hand_landmarks:
        # Landmark 0 is the wrist for both hands
        lw = results.left_hand_landmarks.landmark[0]
        rw = results.right_hand_landmarks.landmark[0]
        
        clasp_dist = get_dist(lw, rw) / face_width
        state["RAW_CLASP"] = round(clasp_dist, 3)
        
        # If wrists are close together, hands are clasped
        if clasp_dist < 0.5:
            state["are_hands_clasped"] = True

    return state