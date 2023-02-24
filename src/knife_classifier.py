# system libraries
import os
from datetime import datetime

# installed libraries
import cv2

# custom classes
from video import VideoGet
from framerate import CountsPerSec

HIGH_CER = 3.8
MED_CER = 2.5
LOW_CER = 1

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

# saves frames that are considered very accurate
def save_image(frame):
    dt = str(datetime.now()).replace(' ', '_').replace(':', '-')[0:19]
    try:
        os.mkdir("positives")
    except:
        pass
    path = f"./positives/knife_{dt}.jpg"
    cv2.imwrite(path, frame)

def main():
    # thread and cascade setup
    casc_path = "knife_data/classifier/cascade.xml"
    vid_path = "../resources/capstone01.mp4"
    knife_cascade = cv2.CascadeClassifier(casc_path)

    video_getter = VideoGet(vid_path).start()
    cps = CountsPerSec().start()

    prev_alert = 0
    prev_img = 0

    while True:
        # read in frame and compute weapon location
        img = video_getter.frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        detection_result, rejectLevels, levelWeights = knife_cascade.detectMultiScale3(img_gray, scaleFactor=1.01, minNeighbors=11, minSize=[90,90], maxSize=[180, 180], outputRejectLevels=1)

        # determine the most likely knife location based on size
        max_coords = [0, 0, 0, 0]
        current_index = -1
        for i, (x, y, w, h) in enumerate(detection_result):
            if w > max_coords[2]:
                max_coords[0] = x
                max_coords[1] = y
                max_coords[2] = w
                max_coords[3] = h
                current_index = i
        
        # handle alerts and drawing if sufficient confidence is determined
        try:
            confidence = levelWeights[current_index]
            if confidence > LOW_CER:
                draw_rectangle(max_coords, img, confidence)
            if confidence >= HIGH_CER:
                occ_frame = cps.get_occurrence()

                # saves an image with a half second separation between saves
                if (occ_frame - prev_img) >= 15:
                    save_image(img)
                    prev_img = occ_frame

                # sounds an alert no more than once per second
                if (occ_frame - prev_alert) >= 30:
                    print("knife found")
                    prev_alert = occ_frame
        except:
            pass

        # display result
        print("Frame rate: ", cps.get_framerate())
        cv2.imshow("Result", img)
        cps.increment()

        if (cv2.waitKey(1) & 0xFF == ord('q')) or video_getter.stopped:
            video_getter.stop()
            break

def detect_knife(img):
    casc_path = "src/new_dataset/classifier/cascade.xml"
    knife_cascade = cv2.CascadeClassifier(casc_path)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    knife = knife_cascade.detectMultiScale(img_gray, 1.01, 11, minSize=[90, 90], maxSize=[180, 180]) # goal size is between 125 and 200

    max_coords = [0, 0, 0, 0]
    for (x, y, w, h) in knife:
        if w < 75:
            continue
        
        if w > max_coords[2]:
            max_coords[0] = x
            max_coords[1] = y
            max_coords[2] = w
            max_coords[3] = h
    
    x, y, w, h = max_coords[0], max_coords[1], max_coords[2], max_coords[3]
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0),  2)

    return img

if __name__ == "__main__":
    main()