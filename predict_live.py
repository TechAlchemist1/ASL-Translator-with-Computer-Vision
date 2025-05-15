import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque
import time

# Load trained KNN model
model = joblib.load('asl_random_forest_model.pkl')


# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera was not found")
    exit()

# State and buffers
current_sentence = ""
current_word = ""
recent_predictions = deque(maxlen=5)
last_confirmed_letter = None
live_detected_letter = ""

# Letter hold delay
letter_hold_time = 4.0  # seconds
last_letter_time = time.time()

# Sentence output file path
file_path = r'C:\Users\qalid\OneDrive\Desktop\gesture_controller\asl_sentences.txt'
show_saved_msg = False
saved_msg_time = 0

# Cooldown helpers
hand_present = False
hand_was_present = False

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    predicted_letter = None
    hand_present = bool(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            prediction = model.predict(np.array(data).reshape(1, -1))[0]
            recent_predictions.append(prediction)

            if len(recent_predictions) == recent_predictions.maxlen:
                if all(p == prediction for p in recent_predictions):
                    predicted_letter = prediction
                    live_detected_letter = predicted_letter
                    print(f"[] Detecting: '{predicted_letter}'")

    # Append letter if cooldown is satisfied
    if predicted_letter and predicted_letter != last_confirmed_letter:
        now = time.time()
        
        if now - last_letter_time > letter_hold_time:
            current_word += ' ' if predicted_letter == 'space' else predicted_letter
            last_confirmed_letter = predicted_letter
            last_letter_time = now

    # Keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Space = finish word
        current_sentence += current_word + " "
        current_word = ""
    elif key == 8:  # Backspace
        current_word = current_word[:-1]
    elif key == 13:  # Enter = finalize sentence
        full_sentence = (current_sentence + current_word).strip()
        print(" Final Sentence:", full_sentence)
        with open(file_path, 'a') as f:
            f.write(full_sentence + "\n")
        current_sentence = ""
        current_word = ""
        show_saved_msg = True
        saved_msg_time = time.time()
    elif key == ord('q'):
        break

    # Reset letter tracker once hand disappears
    if not hand_present and hand_was_present:
        last_confirmed_letter = None

    hand_was_present = hand_present

    # UI Display
    if live_detected_letter:
        cv2.putText(frame, f" Detecting: {live_detected_letter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 255, 200), 2)

    cv2.putText(frame, f"Word: {current_word}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
    #cv2.putText(frame, f"Sentence: {current_sentence}", (10, 110),
               # cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"Full: {current_sentence + current_word}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    if show_saved_msg and (time.time() - saved_msg_time < 2):
        cv2.putText(frame, " Saved", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    elif time.time() - saved_msg_time >= 2:
        show_saved_msg = False

    cv2.imshow("ASL Translator", frame)

# Cleanup
cap.release()
cv2.destroyAllWindows()
