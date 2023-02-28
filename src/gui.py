import cv2
from tkinter import *
import PIL.Image, PIL.ImageTk
from main import calc_max_coords
from image_utils import draw_rectangle
from framerate import CountsPerSec
from video import VideoGet
import mediapipe as mp

class Gui:
    def __init__(self, window):
        self.window = window
        self.window.title("Suspicious Activity Detection")
        self.stopped = True

        self.canvas = Canvas(window, width=960, height=720)
        self.canvas.grid(row=0, column=0)
        self.start_btn()
        self.fps = "0"
        self.framerate_label()

        self.cascade = cv2.CascadeClassifier("knife_data/classifier/cascade.xml")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils

        # needs to be at the end
        self.window.mainloop()

    def start_btn(self):
        self.stopped = False
        self.start_btn = Button(self.window, text='Start', height=2, width=10, command=self.start_video).grid(row=1, column=0)

    def stop_btn(self):
        self.stop_btn = Button(self.window, text='Stop', height=2, width=10, command=self.stop_video).grid(row=1, column=1)

    def stop_video(self):
        self.stopped = True
        self.vid.stop()

    def framerate_label(self):
        self.framerate = Label(self.window, text=f"Processing Speed:\n{self.fps} FPS").grid(row=0, column=1)

    def update_framerate(self):
        self.fps = round(self.cps.get_framerate())
        self.framerate_label()


    def start_video(self):
        if not self.stopped:
            self.stop_btn()
            self.cps = CountsPerSec().start()
            self.vid = VideoGet(src="../resources/capstone01.mp4").start()
            self.update()


    # need to make this gracefully exit
    # this is my new main function
    def update(self):
        # try:
            img = cv2.cvtColor(self.vid.frame, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pose_result = self.pose.process(img_rgb)

            dr, _, lw = self.cascade.detectMultiScale3(img_gray, scaleFactor=1.01, minNeighbors=11, minSize=[90,90], maxSize=[180, 180], outputRejectLevels=1)
            coords, coords_index = calc_max_coords(dr)

            if coords_index > -1:
                confidence = lw[coords_index]
                draw_rectangle(coords, img, confidence)

            if pose_result.pose_landmarks:
                self.mp_draw.draw_landmarks(img, pose_result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                for lm in pose_result.pose_landmarks.landmark:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w) , int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

            # print(self.cps.get_framerate())
            self.update_framerate()

            # renders photo to screen
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
            self.window.after(1, self.update)

            self.cps.increment()
        # except:
        #     pass

Gui(Tk())