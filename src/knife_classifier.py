import cv2
from video import VideoGet
from framerate import CountsPerSec
# import numpy as np

def main():
    casc_path = "new_dataset/classifier/cascade.xml"
    vid_path = "../resources/capstone01.mp4"
    knife_cascade = cv2.CascadeClassifier(casc_path)

    video_getter = VideoGet(vid_path).start()
    cps = CountsPerSec().start()

    while True:
        img = video_getter.frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        knife = knife_cascade.detectMultiScale(img_gray, 1.01, 11, minSize=[90, 90], maxSize=[180, 180]) # goal size is between 125 and 200

        max_coords = [0, 0, 0, 0]
        for (x, y, w, h) in knife:
            
            if w > max_coords[2]:
                max_coords[0] = x
                max_coords[1] = y
                max_coords[2] = w
                max_coords[3] = h
        
        x, y, w, h = max_coords[0], max_coords[1], max_coords[2], max_coords[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0),  2)

        print("Frame rate: ", cps.countsPerSec())
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