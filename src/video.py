from threading import Thread
import cv2

class VideoGet:
    def __init__(self, src=0, delay=33):
        self.stream = cv2.VideoCapture(src)
        # self.stream.set(10, 100)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.delay = delay # 30 fps
        self.width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            cv2.waitKey(self.delay)
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        if self.stream.isOpened():
            self.stream.release()
            self.stopped = True