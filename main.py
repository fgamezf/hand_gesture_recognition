# -*- coding: utf-8 -*-
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
import os
import numpy as np

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

OUTPUT_DIR = './output'
OUTPUT_FILE = '/hand_gesture_data'
OUTPUT_EXT = '.csv'


def drawLandmarks(image, results):
    # 1. Draw face landmarks
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
        
    # 2. Right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )

    # 3. Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
       mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
       mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
       )

    # 4. Pose Detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    return image


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

# For each person, enter one numeric face id
class_name= input('\n Introduce nombre gesto <class> ==>  ')

print("\n [INFO] Inicializamos camara ...")

data_left = []
data_right = []
left_hand = None
right_hand = None
cap = cv2.VideoCapture(0)
# Initiate holistic model
#with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

if holistic is not None:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        # Make Detections
        results = holistic.process(image)
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        #image = cv2.flip(image,1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
       
        # Export coordinates
        try:
            # Extract Left Hand landmarks
            if results.left_hand_landmarks.landmark  is not None:
                left_hand = results.left_hand_landmarks.landmark
                left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())
                left_hand_row.insert(0, class_name + "_left")

                data_left.append(left_hand_row)
            
            if results.right_hand_landmarks.landmark  is not None:
                # Extract Right Hand landmarks
                right_hand = results.right_hand_landmarks.landmark
                right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())
                right_hand_row.insert(0, class_name + "_right")
                
                data_right.append(right_hand_row)
 
        except:
            pass

         # Draw landmarks
        image = drawLandmarks(image, results)

        
        key = cv2.waitKey(1)
        if key == 27:
            break
    
        cv2.imshow('Raw Webcam Feed', image)

cap.release()
cv2.destroyAllWindows()

print ("Generando CSV")

num_coords = 0

if left_hand is not None:
    num_coords = len(left_hand)

if right_hand is not None:
    num_coords = len(right_hand)

if num_coords > 0:
    landmarks = ['class']
    for val in range(1, num_coords+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
                
    with open(OUTPUT_DIR + OUTPUT_FILE + '_' + class_name + OUTPUT_EXT, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)

    # Export to CSV
    with open(OUTPUT_DIR + OUTPUT_FILE + '_' + class_name + OUTPUT_EXT, mode='a', newline='') as f:
        for row in data_left:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(row) 
        for row in data_right:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(row) 