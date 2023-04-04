# libs
import cv2
import numpy as np
from tkinter import *
from tkinter import scrolledtext
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
import customtkinter

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

'''
    TODO: 
    - add posture recognition
    - add toggles for posture and weapon detection
    - add settings menu instead of checkboxes
'''

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()


    # configure window
        self.title("Suspicious Activity Detection")
        self.geometry(f"{1500}x{725}")
        self.vid_stopped = True

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets  OPTIONS COL
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Options", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))


        # START BUTTON
        self.startbtn = customtkinter.CTkButton(self.sidebar_frame, text="Start Monitor", command=self.handle_start_stop)
        self.startbtn.grid(row=1, column=0, padx=20, pady=10)

        
        # input checkbox
        self.webcam_cb_state = IntVar(value=0)
        self.webcam_cb = customtkinter.CTkCheckBox(self.sidebar_frame, text="Webcam", variable=self.webcam_cb_state, onvalue=1, command=self.toggle_webcam_cb)
        self.webcam_cb.grid(row=2, column=0, pady=(20, 0), padx=20, sticky="n")
        self.webcam_cb.select()


        # screenshot checkbox
        self.sh_cb_state = IntVar(value=1)
        self.sh_cb = customtkinter.CTkCheckBox(self.sidebar_frame, text="Screenshot", variable=self.sh_cb_state, onvalue=1, offvalue=0, command=self.toggle_sh_cb)
        self.sh_cb.grid(row=3, column=0, pady=(20, 0), padx=20, sticky="n")
        self.sh_cb.select()


        # weapon detection checkbox
        self.show_weapon = IntVar(value=1)
        self.weapon_cb = customtkinter.CTkCheckBox(self.sidebar_frame, text="Weapon Detection", variable=self.show_weapon, onvalue=1, offvalue=0, command=self.toggle_weapon)
        self.weapon_cb.grid(row=4, column=0, pady=(20, 0), padx=20, sticky="n")
        self.weapon_cb.select()


        # pose_detection checkbox
        self.show_pose = IntVar(value=1)
        self.pose_cb = customtkinter.CTkCheckBox(self.sidebar_frame, text="Pose Detection", variable=self.show_pose, onvalue=1, command=self.toggle_pose)
        self.pose_cb.grid(row=5, column=0, pady=(20, 0), padx=20, sticky="n")
        self.pose_cb.select()
        
       
        # LIGHT AND DARK MODE ETC
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=6, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))


    # Widgets

        # OUTPUT VIDEO
        self.vid_frame = customtkinter.CTkFrame(self, corner_radius=10, width=800)
        self.vid_frame.grid(row=0, column=1, padx=(20, 20), pady=(20, 20))

        self.vid_label = customtkinter.CTkLabel(self.vid_frame, text="Livefeed", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.vid_label.grid(row=0, column=1, columnspan=1, padx=10, pady=10)

        self.cam = customtkinter.CTkLabel(self.vid_frame, corner_radius=10, width=800, height=608, text="")
        self.cam.grid(row=1, column=1, sticky="nsew")

        # Output log
        self.log_frame = customtkinter.CTkFrame(self, corner_radius=10)
        self.log_frame.grid(row=0, column=2, padx=(20, 20), pady=(20, 20))

        self.log_label = customtkinter.CTkLabel(self.log_frame, text="Output Log", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.log_label.grid(row=0, column=1, columnspan=1, padx=10, pady=10, sticky="")
       
        self.log_box = customtkinter.CTkTextbox(self.log_frame, width=400, height=400, cursor=None, yscrollcommand=True, corner_radius=10)
        self.log_box.grid(row=2, column=1, padx=(10, 10), pady=(10, 10), sticky="nsew")

        # set default values
        self.appearance_mode_optionemenu.set("Dark")
        

        # media pipe setup
        self.mp_drawing = mp.solutions.drawing_utils # drawing helpers
        self.mp_holistic = mp.solutions.holistic # Mediapipe Solutions

        self.blf = open("models/body_language.pkl", "rb")
        self.pose_model = None
        with open("models/body_language.pkl", "rb") as f:
            self.pose_model = pickle.load(f)
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # models setup
        self.w_model = YOLO("models/WeaponsBest.pt")

        # self.window.mainloop()

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def toggle_webcam_cb(self):
        if self.vid_stopped:
            self.webcam_cb_state = not self.webcam_cb_state

    def toggle_sh_cb(self):
        self.sh_cb_state = not self.sh_cb_state

    def toggle_weapon(self):
        self.show_weapon = not self.show_weapon

    def toggle_pose(self):
        self.show_pose = not self.show_pose

    # handles start, stop and restart of video
    def handle_start_stop(self):
        self.last_sh = 0 # screenshot frame counter

        if self.vid_stopped:
            self.vid_stopped = False
            self.startbtn.configure(text="Stop")
            self.on_start()
            self.video_loop()
        else:
            self.startbtn.configure(text="Start")
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
            return True

    # updates log box to display results
    def update_log_box(self, text):
        self.log_box.configure(state='normal')
        self.log_box.insert(END, f"{str(datetime.now())}\n{text}")
        self.log_box.configure(state='disabled')
        self.log_box.see(END)

    def display_aggressive(slef, img, pose_class):
        cv2.putText(img, pose_class
                    , (90,48), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def video_loop(self):
        ret, img = self.cap.read()

        # if video is being played
        if ret:
            # weapon prediction
            w_res = None
            if self.show_weapon:
                w_res = self.w_model.predict(source=img, conf=0.5)
                print(w_res[0].names)

            pose_class = None

            results = None
            if self.show_pose:
                results = self.holistic.process(img)

                # 4. Pose Detections
                self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                                            )

                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark

                    row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())


                    # make detections
                    X = pd.DataFrame([row])

                    body_language_class = self.pose_model.predict(X)[0]
                    body_language_prob = self.pose_model.predict_proba(X)[0]

                    # Get status box
                    cv2.rectangle(img, (0,0), (250, 60), (245, 117, 16), -1)
                    
                    # Display Class
                    pose_class = body_language_class.split(' ')[0]
                    cv2.putText(img, 'CLASS'
                                , (95,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    
                    if not self.show_weapon:
                        cv2.putText(img, pose_class
                                    , (90,48), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    if pose_class == 'Fighting':
                        self.update_log_box("Aggressive behaviour detected.\n\n")
                    
                    # Display Probability
                    cv2.putText(img, 'PROB'
                                , (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(img, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (10,48), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                except:
                    pass

            if self.show_weapon:
                # results display logic
                for r in w_res:
                    annotator = Annotator(img)

                    boxes = r.boxes

                    if len(boxes) and self.show_pose:
                        self.display_aggressive(img, "Aggressive")
                    elif not len(boxes) and self.show_pose:
                        self.display_aggressive(img, pose_class)
                    elif self.show_pose:
                        self.display_aggressive(img, pose_class)

                    for box in boxes:
                        b = box.xyxy[0]
                        c = box.cls

                        confidence = box.conf[0]
                        w_type = self.w_model.names[int(c)]
                        
                        # filters model for unwanted classes
                        if w_type in ['Gun', 'Knife', 'Pistol', 'handgun', 'rifle']:
                            # convert all gun types to be Gun class
                            if w_type != 'Knife':
                                w_type = 'Gun'
                            
                            header = f"{w_type} {confidence:.2f}"

                            annotator.box_label(b, header, (0, 0, 255))

                            if confidence > 0.5 and confidence < 0.8:
                                self.update_log_box(f"Potential {w_type} found.\n\n")

                            if confidence >= 0.8:
                                self.update_log_box(f"Positive {w_type} found.\n\n")

                            # takes a screenshot with minimum 20 frame (~1s) separation
                            if self.sh_cb_state:
                                is_saved = self.take_screenshot(box, img)
                                if is_saved:
                                    self.update_log_box(f"Image Saved.\n\n")

            # rendering logic
            draw_framerate(img, self.cps.get_framerate())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.frame = ImageTk.PhotoImage(image=Image.fromarray(img))

            # Format into tkinter image and add to GUI
            self.cam.imgtk = self.frame
            self.cam.configure(image=self.cam.imgtk)
            self.after(1, self.video_loop)

            # fps incrementer
            self.cps.increment()

        # if video has ended or been stopped
        else:
            black_img = np.zeros((608, 800, 3), dtype=np.uint8)
            self.frame = ImageTk.PhotoImage(image=Image.fromarray(black_img))
            self.cam.imgtk = self.frame
            self.cam.configure(image=self.cam.imgtk)


if __name__ == '__main__':
    app = App()
    app.mainloop()