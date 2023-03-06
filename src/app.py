# libs
import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk

# ultralytics
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

# custom classes
from framerate import CountsPerSec
from image_utils import draw_framerate, save_image


'''
    TODO: 
    - clean up layout
'''

class App:
    def __init__(self, window):
        # some setup stuff
        self.window = window
        self.window.title("Suspicious Activity Detection")
        self.vid_stopped = True


        # for now this is the display size
        self.canvas = Canvas(window, width=960, height=720)
        self.canvas.pack()


        # models setup
        self.w_model = YOLO("models/knifeDetector.pt")
        self.model_cb_state = True

        # buttons
        self.ss_button()

        # checkboxes
        # webcam checkbox
        self.webcam_cb_state = False
        self.webcam_checkbox()

        # screenshot checkbox
        self.sh_cb_state = True
        self.sh_checkbox()

        self.window.mainloop()

    # buttons
    # start/stop button
    def ss_button(self):
        self.start_btn = Button(self.window, text='Start', height=2, width=10, command=self.handle_start_stop)
        self.start_btn.pack()

    # checkboxes
    # toggles webcam or video
    def webcam_checkbox(self):
        self.webcam_cb = Checkbutton(self.window, text="Webcam", variable=self.webcam_cb_state, onvalue=True, offvalue=False, command=self.toggle_webcam_cb)
        self.webcam_cb.pack()

    def toggle_webcam_cb(self):
        if self.vid_stopped:
            self.webcam_cb_state = not self.webcam_cb_state

    def sh_checkbox(self):
        self.sh_cb = Checkbutton(self.window, text="Screenshot", variable=self.sh_cb_state, onvalue=True, offvalue=False, command=self.toggle_sh_cb)
        self.sh_cb.select()
        self.sh_cb.pack()

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

    def take_screenshot(self, box, img):
        current_frame = self.cps.get_occurrence()
        if (current_frame - self.last_sh) > 20 and float(box.conf[0]) > 0.85:
            self.last_sh = current_frame
            save_image(img)


    def video_loop(self):
        ret, img = self.cap.read()

        # if video is being played
        if ret:
            # weapon prediction
            k_res = self.w_model.predict(source=img, conf=0.5)
            draw_framerate(img, self.cps.get_framerate())

            # results display logic
            for r in k_res:
                annotator = Annotator(img)

                boxes = r.boxes

                for box in boxes:
                    b = box.xyxy[0]
                    c = box.cls

                    header = f"{self.w_model.names[int(c)]} {box.conf[0]:.2f}"
                    annotator.box_label(b, header, (0, 0, 255))

                    # takes a screenshot with minimum 20 frame (~1s) separation
                    if self.sh_cb_state:
                        self.take_screenshot(box, img)

            # rendering logic
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.frame = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
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