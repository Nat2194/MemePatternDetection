import math

def get_angle(p1, p2, p3):
    """Calculates the 2D interior angle (in degrees)."""
    angle = math.degrees(math.atan2(p3.y - p2.y, p3.x - p2.x) - 
                         math.atan2(p1.y - p2.y, p1.x - p2.x))
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle

def update_state(results, state, face_width):
    """Detects if either arm is flexing."""
    state["is_flexing"] = False
    state["RAW_R_FLEX"] = "Not Visible"
    state["RAW_L_FLEX"] = "Not Visible"

    if not results.pose_landmarks:
        return state
        
    pose = results.pose_landmarks.landmark
    
    # Right Arm
    r_shoulder, r_elbow, r_wrist = pose[12], pose[14], pose[16]
    if r_shoulder.visibility > 0.2 and r_elbow.visibility > 0.2 and r_wrist.visibility > 0.2:
        angle_r = get_angle(r_shoulder, r_elbow, r_wrist)
        is_high_r = r_wrist.y < r_shoulder.y
        state["RAW_R_FLEX"] = f"{int(angle_r)}deg | High:{is_high_r}"
        if angle_r < 90 and is_high_r:
            state["is_flexing"] = True

    # Left Arm
    l_shoulder, l_elbow, l_wrist = pose[11], pose[13], pose[15]
    if l_shoulder.visibility > 0.2 and l_elbow.visibility > 0.2 and l_wrist.visibility > 0.2:
        angle_l = get_angle(l_shoulder, l_elbow, l_wrist)
        is_high_l = l_wrist.y < l_shoulder.y
        state["RAW_L_FLEX"] = f"{int(angle_l)}deg | High:{is_high_l}"
        if angle_l < 90 and is_high_l:
            state["is_flexing"] = True
            
    return state