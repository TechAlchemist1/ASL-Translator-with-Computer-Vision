import cv2
import mediapipe as mp
import time
from pynput.keyboard import Controller, Key, KeyCode

# Setup webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize MediaPipe Hands
hands = mp.solutions.hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
draw = mp.solutions.drawing_utils

keyboard = Controller()

prev_action = None
prev_right_action = None
right_hold_action = None

def is_fist(landmarks):
    return all(landmarks[tip].y > landmarks[pip].y for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]))

def count_raised_fingers(landmarks):
    tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    pips = [3, 6, 10, 14, 18]
    count = 0
    for tip, pip in zip(tips, pips):
        if tip == 4:
            if abs(landmarks[tip].x - landmarks[pip].x) > 0.04:
                count += 1
        else:
            if landmarks[tip].y < landmarks[pip].y:
                count += 1
    return count

# Mapping actions to keys
key_map = {
    "Jump": Key.space,
    "Forward": KeyCode.from_char('w'),
    "Backward": KeyCode.from_char('s'),
    "Right": KeyCode.from_char('d'),
    "Left": KeyCode.from_char('a'),
    "Crouch": Key.alt_l,
    "Shoot": KeyCode.from_char('y'),
    "Zoom": KeyCode.from_char('u')
}

action_key_state = {k: False for k in key_map}

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[idx].classification[0].label  # "Left" or "Right"
            draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # LEFT HAND - Movement
            if hand_label == "Left":
                action = None
                if is_fist(landmarks):
                    action = "Jump"
                else:
                    finger_count = count_raised_fingers(landmarks)
                    if finger_count == 1:
                        action = "Forward"
                    elif finger_count == 2:
                        action = "Backward"
                    elif finger_count == 3:
                        action = "Right"
                    elif finger_count == 4:
                        action = "Left"
                    elif finger_count == 5:
                        action = "Crouch"

                for act in action_key_state:
                    if action == act:
                        if not action_key_state[act] and act != "Jump":
                            keyboard.press(key_map[act])
                            action_key_state[act] = True
                    else:
                        if action_key_state[act] and act != "Jump" and act not in ["Shoot", "Zoom"]:
                            keyboard.release(key_map[act])
                            action_key_state[act] = False

                if action == "Jump":
                    keyboard.press(key_map[action])
                    keyboard.release(key_map[action])

                prev_action = action

                cv2.putText(frame, f"Left: {action}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # RIGHT HAND - Shoot, Zoom, Zoom+Shoot
            elif hand_label == "Right":
                finger_count = count_raised_fingers(landmarks)
                right_action = None
                if finger_count == 1:
                    right_action = "Shoot"
                elif finger_count == 2:
                    right_action = "Zoom"
                elif finger_count == 3:
                    right_action = "Zoom+Shoot"

                if right_action != right_hold_action:
                    # Release previously held keys
                    if right_hold_action == "Zoom":
                        keyboard.release(key_map["Zoom"])
                        action_key_state["Zoom"] = False
                    if right_hold_action == "Zoom+Shoot":
                        keyboard.release(key_map["Zoom"])
                        keyboard.release(key_map["Shoot"])
                        action_key_state["Zoom"] = False
                        action_key_state["Shoot"] = False

                    # Perform new action
                    if right_action == "Shoot":
                        keyboard.press(key_map["Shoot"])
                        keyboard.release(key_map["Shoot"])
                    elif right_action == "Zoom":
                        keyboard.press(key_map["Zoom"])
                        action_key_state["Zoom"] = True
                    elif right_action == "Zoom+Shoot":
                        keyboard.press(key_map["Zoom"])
                        keyboard.press(key_map["Shoot"])
                        action_key_state["Zoom"] = True
                        action_key_state["Shoot"] = True

                    right_hold_action = right_action

                prev_right_action = right_action

                cv2.putText(frame, f"Right: {right_action}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Two-Hand Gesture Controller", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
