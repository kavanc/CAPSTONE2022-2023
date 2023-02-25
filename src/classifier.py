import cv2
from threading import Thread

# A class for creating weapon classifier threads.
class Classifier:
    def __init__(self, casc_path, frame, num=1, logging=False):
        self.cascade = cv2.CascadeClassifier(casc_path)
        self.frame = frame
        self.num = num
        self.logging = logging
        self.thread = Thread(target=self.classify, args=())
        self.dr = None
        self.lw = None

    # starts the thread
    def start(self):
        if self.logging:
            print(f"Starting: {self.num}")
        self.thread.start()
        return self

    # thread function
    # returns coordinates of weapon and confidence value
    def classify(self):
        self.dr, _, self.lw = self.cascade.detectMultiScale3(self.frame, scaleFactor=1.01, minNeighbors=11, minSize=[90,90], maxSize=[180, 180], outputRejectLevels=1)

    def get_classify_results(self):
        return self.dr, self.lw
        
    # join the thread
    def join(self):
        if self.logging:
            print(f"Joining: {self.num}")
        self.thread.join()
