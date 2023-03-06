import cv2
from ultralytics import YOLO
from model_thread import MThread
from framerate import CountsPerSec
from ultralytics.yolo.utils.plotting import Annotator
import mediapipe as mp 
import pickle 
import pandas as pd
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

def get_frame(f_num, path):
    # vid_path = "../resources/multi_knife.mp4"
    cap = cv2.VideoCapture(path)
    frame = None

    for i in range(0, 500):
        _, frame = cap.read()
        if i < f_num:
            continue
        cap.release()
        return frame

def draw_header(img, name, x1, y1, x2):
    y = y1 - 30
    # cv2.rectangle(img, (x1, y), (x2, y1), (0, 0, 255), 2)
    cv2.putText(img, name, (x1, y1 - 10), FONT, 0.9, (0, 0, 255), 2)

def draw_framerate(img, fps):
    cv2.putText(img, str(int(fps)), (10, 25), FONT, 0.8, (0, 255, 0), 2)

'''

frame 51
'''

def test():
    knife_model = "models/knifeDetector.pt"
    gun_model = "models/KavanGunbest.pt"
    vid_path = "../resources/multi_knife.mp4"
    # vid_path = "../resources/capstone01.mp4"

    model = YOLO(knife_model)
    # model = YOLO(gun_model)

    # frame = get_frame(167, vid_path)
    frame = get_frame(51, vid_path)

    res = model.predict(source=frame, show=False, conf=0.1)

    for r in res:
        annotator = Annotator(frame)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls

            annotator.box_label(b, f"{model.names[int(c)]} {box.conf[0]:.2f}", (0, 0, 255))

    img = annotator.result()
    cv2.imshow('RES', img)
    cv2.waitKey()

def thread_test():
    knife_model_path = "models/knifeDetector.pt"
    gun_model_path = "models/KavanGunbest.pt"
    
    knife_model = YOLO(knife_model_path)
    gun_model = YOLO(gun_model_path)

    cap = cv2.VideoCapture(0)

    cps = CountsPerSec().start()

    while True:
        ret, img = cap.read()

        knife_thread = MThread(img, knife_model, 0.7).start()
        gun_thread = MThread(img, gun_model, 0.5).start()

        knife_thread.join()
        gun_thread.join()

        knife_res = knife_thread.get_res()
        gun_res = gun_thread.get_res()

        # res = model.predict(source=img, show=False, conf=0.7)

        try:
            knife_boxes = []
            gun_boxes = []

            knife_bb = []
            gun_bb = []
            for r in knife_res:
                knife_boxes = r.boxes.xyxy

            for r in gun_res:
                gun_boxes = r.boxes.xyxy

            if len(knife_boxes):
                for i in range(len(knife_boxes)):
                    knife_bb.append([int(x) for x in knife_boxes[i]])

                for b in knife_bb:
                    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    draw_header(img, "Knife", x1, y1, x2)

            if len(gun_boxes):
                for i in range(len(gun_boxes)):
                    gun_bb.append([int(x) for x in gun_boxes[i]])

                for b in gun_bb:
                    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    draw_header(img, "Gun", x1, y1, x2)

        except:
            pass

        draw_framerate(img, cps.get_framerate())
        cv2.imshow("Result", img)
        cps.increment()

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break


# COLE, THIS RUNS ON WEBCAM
def main_failed():
    knife_model = "models/knifeDetector.pt"
    gun_model = "models/KavanGunbest.pt"
    model2 = YOLO(knife_model)
    model = YOLO(gun_model)

    cap = cv2.VideoCapture(0)

    cps = CountsPerSec().start()

    while True:
        ret, img = cap.read()

        res = model.predict(source=img, show=False, conf=0.7)
        model2.predict(source=img, show=False, conf=0.7)

        try:
            boxes = []
            bb_list = []
            for r in res:
                boxes = r.boxes.xyxy

            if len(boxes):
                for i in range(len(boxes)):
                    bb_list.append([int(x) for x in boxes[i]])

                for b in bb_list:
                    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    draw_header(img, "Knife", x1, y1, x2)

        except:
            pass

        draw_framerate(img, cps.get_framerate())
        cv2.imshow("Result", img)
        cps.increment()

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

def main():
    knife_model = "models/WeaponsBest.pt"
    model = YOLO(knife_model)
    cap = cv2.VideoCapture(0)
    cps = CountsPerSec().start()
    
    # MediaPipe Pose
    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions

    with open('models/body_language.pkl', 'rb') as f:
        poseModel = pickle.load(f)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
        results = None
        while True:
            ret, img = cap.read()

            res = model.predict(source=img, show=False, conf=0.5)

            for r in res:
                annotator = Annotator(img)

                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]
                    c = box.cls

                    annotator.box_label(b, f"{model.names[int(c)]} {box.conf[0]:.2f}", (0, 0, 255))

            img = annotator.result()
            draw_framerate(img, cps.get_framerate())

            
            # Recolor Feed
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False        
            
            # Make Detections
            results = holistic.process(img)
            
            # Recolor image back to BGR for rendering
            img.flags.writeable = True   
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 4. Pose Detections
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
            

            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Concate rows
                row = pose_row
                
                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = poseModel.predict(X)[0]
                body_language_prob = poseModel.predict_proba(X)[0]
                print(body_language_class, body_language_prob)
                
                # Grab ear coords
                coords = tuple(np.multiply(
                                np.array(
                                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                            , [640,480]).astype(int))
                
                cv2.rectangle(img, 
                            (coords[0], coords[1]+5), 
                            (coords[0]+len(body_language_class)*20, coords[1]-30), 
                            (245, 117, 16), -1)
                cv2.putText(img, body_language_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Get status box
                cv2.rectangle(img, (0,0), (250, 60), (245, 117, 16), -1)
                
                # Display Class
                cv2.putText(img, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, body_language_class.split(' ')[0]
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(img, 'PROB'
                            , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            except:
                pass


            cv2.imshow("Result", img)
            cps.increment()

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break


if __name__ == '__main__':
    main()