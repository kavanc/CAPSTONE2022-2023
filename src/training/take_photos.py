'''
A simple interface for taking photos of a pose.
'''

import cv2
import mediapipe as mp
import time
import os

if __name__ == '__main__':
    title = "Fighting"
    
    # media pipe setup
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # initialize some stuff
    cap = cv2.VideoCapture(0)
    results = None
    st = time.time()

    # enable mediapipe
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        # determine where to start adding suffixes to the file name
        # counter = len(os.listdir(f"./evaluation_data/{title}"))
        counter = len(os.listdir(title))

        # photo taking loop
        while cap.isOpened():
            # read image and add pose points
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
                            
            cv2.imshow(title, image)

            # slight delay from the start of photo taking
            if time.time() - st > 3:
                counter += 1
                try:
                    os.mkdir(title)
                except:
                    pass

                # write images to directories
                # path = f"./evaluation_data/{title}/{title}_{counter}.jpg"
                path = f"./{title}/{title}_{counter}.jpg"
                cv2.imwrite(path, image)

            # terminate by pressing q
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()