import cv2
# import numpy as np

def main():
    casc_path = "src/new_dataset/classifier/cascade.xml"
    # casc_path = "haar/temp/classifier/cascade.xml"

    knife_cascade = cv2.CascadeClassifier(casc_path)

    cap = cv2.VideoCapture(0)
    cap.set(10, 100)

    while True:
        _, img = cap.read()
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
        # cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0),  2)

        cv2.imshow("Result", img)
        # cv2.imshow("Result gray", img_gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
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