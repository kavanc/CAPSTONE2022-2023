# libs
import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from datetime import datetime
import pandas as pd

# ultralytics
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

# media pipe
import mediapipe as mp
import pickle

# custom classes
from framerate import CountsPerSec
from image_utils import draw_framerate, save_image


'''
    TODO: 
    - add posture recognition
    - add toggles for posture and weapon detection
    - add settings menu instead of checkboxes
'''

class App:
    def __init__(self, window):
        # some setup stuff
        self.window = window
        self.window.title("Suspicious Activity Detection")
        self.vid_stopped = True

        self.header_font = "helvetica 12 bold"

        # layout
        # video frame
        self.vid_frame = Frame(window)
        self.vid_frame.grid(row=0, column=0)
        # video dispay
        self.canvas = Canvas(self.vid_frame, width=960, height=720)
        self.canvas.grid(row=0, column=0)
        # start stop button
        self.start_btn = Button(self.vid_frame, text='Start', height=2, width=10, command=self.handle_start_stop)
        self.start_btn.grid(row=1, column=0)

        # logging frame
        self.log_frame = Frame(window)
        self.log_frame.grid(row=0, column=1)
        self.log_label = Label(self.log_frame, text="Logs", font=self.header_font)
        self.log_label.grid(row=0, column=0)
        self.log_box = Text(self.log_frame, height=45, width=30, cursor=None)
        self.log_box.configure(state='disabled')
        self.log_box.grid(row=1, column=0)

        # options frame
        self.options_frame = Frame(window)
        self.options_frame.grid(row=0, column=2)
        # options header
        self.options_label = Label(self.options_frame, text="Options", font=self.header_font)
        self.options_label.grid(row=0, column=0)
        # input checkbox
        self.webcam_cb_state = False
        self.webcam_cb = Checkbutton(self.options_frame, text="Webcam", variable=self.webcam_cb_state, onvalue=True, offvalue=False, command=self.toggle_webcam_cb)
        self.webcam_cb.grid(row=1, column=0)
        # screenshot checkbox
        self.sh_cb_state = True
        self.sh_cb = Checkbutton(self.options_frame, text="Screenshot", variable=self.sh_cb_state, onvalue=True, offvalue=False, command=self.toggle_sh_cb)
        self.sh_cb.select()
        self.sh_cb.grid(row=2, column=0)


        # media pipe setup
        self.mp_drawing = mp.solutions.drawing_utils # drawing helpers
        self.mp_holistic = mp.solutions.holistic # Mediapipe Solutions

        self.blf = open("models/body_language.pkl", "rb")
        self.pose_model = None
        with open("models/body_language.pkl", "rb") as f:
            self.pose_model = pickle.load(f)
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # models setup
        self.w_model = YOLO("models/knifeDetector.pt")

        self.window.mainloop()

    def toggle_webcam_cb(self):
        if self.vid_stopped:
            self.webcam_cb_state = not self.webcam_cb_state

    def toggle_sh_cb(self):
        self.sh_cb_state = not self.sh_cb_state


    # handles start, stop and restart of video
    def handle_start_stop(self):
        self.last_sh = 0 # screenshot frame counter

        if self.vid_stopped:
            self.vid_stopped = False
            self.start_btn.config(text="Stop")
            self.on_start()
            self.video_loop()
        else:
            self.start_btn.config(text="Start")
            self.vid_stopped = True
            self.on_stop()

    # this behaves as the logic outside the while loop
    def on_start(self):
        # this should be changed to accept webcam instead
        if self.webcam_cb_state:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture("../resources/capstone01.mp4")

        # framerate
        self.cps = CountsPerSec().start()

    # video object cleanup
    def on_stop(self):
        if self.cap.isOpened():
            self.cap.release()

    # takes screenshots
    def take_screenshot(self, box, img):
        current_frame = self.cps.get_occurrence()
        if (current_frame - self.last_sh) > 20 and float(box.conf[0]) > 0.85:
            self.last_sh = current_frame
            save_image(img)

    # updates log box to display results
    def update_log_box(self, text):
        self.log_box.configure(state='normal')
        self.log_box.insert(END, f"{str(datetime.now())}\n{text}")
        self.log_box.configure(state='disabled')

    def video_loop(self):
        ret, img = self.cap.read()

        # if video is being played
        if ret:
            # weapon prediction
            k_res = self.w_model.predict(source=img, conf=0.5)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False

            results = self.holistic.process(img)

            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 4. Pose Detections
            self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                        self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                        self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                                        )

            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark

                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                row = pose_row

                # make detections
                X = pd.DataFrame([row])

                body_language_class = self.pose_model.predict(X)[0]
                body_language_prob = self.pose_model.predict_proba(X)[0]

                # Get status box
                cv2.rectangle(img, (0,0), (250, 60), (245, 117, 16), -1)
                
                # Display Class
                pose_class = body_language_class.split(' ')[0]
                cv2.putText(img, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, pose_class
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                if pose_class == 'Fighting':
                    self.update_log_box("Aggressive behaviour detected!")
                
                # Display Probability
                cv2.putText(img, 'PROB'
                            , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except:
                pass

            # results display logic
            for r in k_res:
                annotator = Annotator(img)

                boxes = r.boxes

                for box in boxes:
                    b = box.xyxy[0]
                    c = box.cls

                    confidence = box.conf[0]
                    w_type = self.w_model.names[int(c)]
                    header = f"{w_type} {confidence:.2f}"
                    annotator.box_label(b, header, (0, 0, 255))

                    if confidence > 0.5 and confidence < 0.8:
                        self.update_log_box(f"Potential {w_type} found.\n\n")

                    if confidence > 0.8:
                        self.update_log_box(f"Positive {w_type} found.\n\n")

                    # takes a screenshot with minimum 20 frame (~1s) separation
                    if self.sh_cb_state:
                        self.take_screenshot(box, img)

            # rendering logic
            draw_framerate(img, self.cps.get_framerate())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.frame = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.canvas.create_image(0, 0, image=self.frame, anchor=NW)
            self.window.after(1, self.video_loop)

            # fps incrementer
            self.cps.increment()

        # if video has ended or been stopped
        else:
            black_img = np.zeros((720, 960, 3), dtype=np.uint8)
            self.frame = ImageTk.PhotoImage(image=Image.fromarray(black_img))
            self.canvas.create_image(0, 0, image=self.frame, anchor=NW)

if __name__ == '__main__':
    App(Tk())