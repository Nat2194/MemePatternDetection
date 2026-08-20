import math


def get_dist(p1, p2):
    """Calculates the true 3D distance between two points."""
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2 + (p2.z - p1.z) ** 2)


def check_hand_fist(hand_landmarks):
    """Strict mesh-based exact fist check."""
    if not hand_landmarks:
        return False
    wrist = hand_landmarks.landmark[0]
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17]  # Base knuckles

    # Require tip to be significantly closer to the wrist than the knuckle (with a strict margin)
    return all(
        get_dist(hand_landmarks.landmark[t], wrist)
        < (get_dist(hand_landmarks.landmark[m], wrist) * 0.85)
        for t, m in zip(tips, mcps)
    )


def update_state(results, state, face_width):
    """Detects if either hand is making a fist with strict thresholds to prevent false positives."""
    state["left_is_fist"] = False
    state["right_is_fist"] = False
    state["RAW_L_FIST"] = "Not Visible"
    state["RAW_R_FIST"] = "Not Visible"

    # --- LEFT FIST ---
    if results.left_hand_landmarks:
        state["left_is_fist"] = check_hand_fist(results.left_hand_landmarks)
        state["RAW_L_FIST"] = "Mesh OK"
    elif results.pose_landmarks:
        pose = results.pose_landmarks.landmark
        if (
            pose[15].visibility > 0.5 and pose[19].visibility > 0.5
        ):  # Increased visibility threshold
            fist_ratio = get_dist(pose[15], pose[19]) / face_width
            state["RAW_L_FIST"] = round(fist_ratio, 3)
            state["left_is_fist"] = fist_ratio < 0.95  # Tightened from 1.2 to 0.95

    # --- RIGHT FIST ---
    if results.right_hand_landmarks:
        state["right_is_fist"] = check_hand_fist(results.right_hand_landmarks)
        state["RAW_R_FIST"] = "Mesh OK"
    elif results.pose_landmarks:
        pose = results.pose_landmarks.landmark
        if (
            pose[16].visibility > 0.5 and pose[20].visibility > 0.5
        ):  # Increased visibility threshold
            fist_ratio = get_dist(pose[16], pose[20]) / face_width
            state["RAW_R_FIST"] = round(fist_ratio, 3)
            state["right_is_fist"] = fist_ratio < 0.95  # Tightened from 1.2 to 0.95

    return state
