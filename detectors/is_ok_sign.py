import math

def get_dist(p1, p2):
    """Calculates the true 3D distance between two points."""
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

def update_state(results, state, face_width):
    """Detects if the right hand is making an OK sign."""
    state["right_ok_sign"] = False
    state["RAW_OK"] = "No Hand"

    if results.right_hand_landmarks:
        hand = results.right_hand_landmarks.landmark
        
        # 4 = Thumb tip, 8 = Index tip
        thumb_index_dist = get_dist(hand[4], hand[8])
        state["RAW_OK"] = round(thumb_index_dist, 3)
        
        # Thumb and index must be pinched together
        is_pinched = thumb_index_dist < 0.05
        
        # Middle (12), Ring (16), and Pinky (20) tips must be higher than their PIP knuckles (10, 14, 18)
        others_open = all(
            hand[t].y < hand[k].y 
            for t, k in zip([12, 16, 20], [10, 14, 18])
        )
        
        if is_pinched and others_open:
            state["right_ok_sign"] = True

    return state