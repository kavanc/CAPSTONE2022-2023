import cv2
import os
from datetime import datetime
from constants import HIGH_CER, MED_CER, LOW_CER

FONT = cv2.FONT_HERSHEY_SIMPLEX

# draws rectangle according to accuracy
def draw_rectangle(coords, img, accuracy):
    if accuracy >= HIGH_CER:
        color = (0, 255, 0) # green

    if accuracy > MED_CER and accuracy < HIGH_CER:
        color = (255, 0, 0) # blue

    if accuracy <= MED_CER:
        color = (0, 0, 255) # red

    x, y, w, h = coords[0], coords[1], coords[2], coords[3]
    cv2.rectangle(img, (x, y), (x + w, y + h), color,  2)

# saves a frame with a name containing the timestamp
def save_image(frame):
    dt = str(datetime.now()).replace(' ', '_').replace(':', '-')[0:19]
    try:
        os.mkdir("positives")
    except:
        pass
    path = f"./positives/knife_{dt}.jpg"
    cv2.imwrite(path, frame)

def draw_framerate(img, fps):
    cv2.putText(img, str(int(fps)), (10, 25), FONT, 0.8, (0, 255, 0), 2)