#We want a csv file that contains the data of the hand landmarks and the labels of the signs
import csv #This imports the csv library which is used to write the data to a csv file
import os #This imports the os library which is used to create the directory for the csv file
#this imports the library needed for the camera
import cv2
#import midead pipe for hands detection
import mediapipe as mp
#Gets the defult camera which is cam zero which is the dfeult camera index
cap = cv2.VideoCapture(0)

#this tests of  camera is not working and gives us an error 

if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

# Set the desired directory path
folder_path = r'C:\Users\qalid\OneDrive\Desktop\gesture_controller'
filename = os.path.join(folder_path, 'asl_data.csv')

# Always overwrite the file with fresh headers
if not os.path.exists(filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{axis}{i}" for i in range(21) for axis in ['x','y','z']] + ['label']
        writer.writerow(header)


mp_hands = mp.solutions.hands  # Initialize MediaPipe Hands
    # this sets the hands detector and initializes the hands detection
hands = mp_hands.Hands(    
    max_num_hands=2,  # Maximum number of hands to detect
    min_detection_confidence=0.7,  # Minimum confidence for detection
    min_tracking_confidence=0.7  # Minimum confidence for tracking
)
mp_draw = mp.solutions.drawing_utils  # Drawing utilities for drawing landmarks on the image




#Get the frames in a loop fasion since the camera is just a bunch of pictires in a loop
while True:
    #Ret is a boolean that checks if the frame is captured and frame is the image captured from the camera
    #If it is then its saved
    

# Ret returns 2 values , the first value which is a boolean that checks if the frame is captured and the second value is the image captured from the camera
    result = cap.read() #Read the image from the camera
    ret = result[0] #Get the first value which is a boolean that checks if the frame is captured
    if not ret:
        print("Print Camera arropr") #Display the image captured from the camera in a window called "Camera"
        break #If the image is not captured then break the loop and close the camera


    frame = result[1] #Get the second value which is the image captured from the camera
    #Flip the image horizontally to get a mirror image of the camera
    frame = cv2.flip(frame, 1) 

    #This converts the image from BGR to RGB since the hands detection works with RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

    #This gets the hands from the image captured from the camera and processes it to get the landmarks of the hands
    #This is done by using the hands detection from mediapipe which is a library that uses machine learning to detect hands in images
    results = hands.process(rgb_frame)
    #This checks if the hands are detected in the image captured from the camera and if they are then it draws the landmarks of the hands on the image captured from the camera
    #This is done by using the drawing utilities from mediapipe which is a library that uses machine learning to detect hands in images
    if results.multi_hand_landmarks:
        # Loop through each hand detected
        for hand_landmarks in results.multi_hand_landmarks:
            #Draws the land marks of the hands on the image captured from the camera
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            key = cv2.waitKey(1) & 0xFF
            if (ord('a') <= key <= ord('z')) or key == 32:  # 32 = spacebar
                label = 'space' if key == 32 else chr(key)
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])
                with open(filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data + [label])
                    print(f"[✓] Saved sample for '{label}'")



    
    cv2.imshow("Qalids ASL Translator", frame) #Display the image captured from the camera in a window called "Camera" 
    
    #This checks if the key 'q' is pressed and if it is then it breaks the loop and closes the camera
    if cv2.waitKey(1) & 0xFF == ord('q'): #If the key 'q' is pressed, break the loop
        break

#Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()