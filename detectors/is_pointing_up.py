def is_pointing_up_hand(hand_landmarks):
    """Helper to check if a specific hand is pointing up with a strict margin."""
    if not hand_landmarks: 
        return False
    # Index finger tip (8) must be distinctly higher than its base joint/knuckle (6)
    index_up = hand_landmarks.landmark[8].y < (hand_landmarks.landmark[6].y - 0.02)
    
    # Middle (12, 10), Ring (16, 14), and Pinky (20, 18) tips must be lower than their knuckles
    others_down = all(
        hand_landmarks.landmark[t].y > hand_landmarks.landmark[k].y 
        for t, k in zip([12, 16, 20], [10, 14, 18])
    )
    return index_up and others_down

def update_state(results, state, face_width):
    """Detects if the right hand is pointing straight up with strict fallback."""
    state["right_is_pointing_up"] = False
    state["RAW_POINT"] = "No Hand"
    
    if results.right_hand_landmarks:
        is_pointing = is_pointing_up_hand(results.right_hand_landmarks)
        state["right_is_pointing_up"] = is_pointing
        state["RAW_POINT"] = "Pointing" if is_pointing else "Not Pointing"
        
    return state