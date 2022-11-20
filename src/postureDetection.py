import cv2
import mediapipe as mp
import time
from knife_classifier import detect_knife

def main():
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(10, 150) # adjust brightness

    pTime = 0
    counter = 0
    personDetected = False

    while True:
        _, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        print(results.pose_landmarks)
        detect_knife(img)
        if results.pose_landmarks:
            personDetected = True
            counter += 1
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h,w,c = img.shape
                print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if personDetected == True and counter > 40:
            cv2.putText(img, str("Person Detected"), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
        
        if personDetected == True and counter > 160:
            counter = 0


        cv2.putText(img, str(int(fps)), (550,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)


        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()