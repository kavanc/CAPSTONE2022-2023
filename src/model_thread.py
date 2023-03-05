import cv2
from threading import Thread

class MThread:
    def __init__(self, frame, model, conf, show=False):
        self.frame = frame
        self.model = model
        self.thread = Thread(target=self.predict, args=())
        self.conf = conf

    def start(self):
        self.thread.start()
        return self

    def predict(self):
        self.res = self.model.predict(source=self.frame, show=self.show, conf=self.conf)

    def get_res(self):
        return self.res

    def join(self):
        self.thread.join()