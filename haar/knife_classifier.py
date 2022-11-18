import cv2
import numpy as np

def main():
    casc_path = "w_faces/classifier/cascade.xml"

    knife_cascade = cv2.CascadeClassifier(casc_path)

    cap = cv2.VideoCapture(0)
    cap.set(10, 150)

    while True:
        _, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        knife = knife_cascade.detectMultiScale(img_gray, 1.3, 22)

        max_coords = [0, 0, 0, 0]
        for i, (x, y, w, h) in enumerate(knife):
            if w < 75:
                continue
            
            if w > max_coords[2]:
                max_coords[0] = x
                max_coords[1] = y
                max_coords[2] = w
                max_coords[3] = h
        
        x, y, w, h = max_coords[0], max_coords[1], max_coords[2], max_coords[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0),  2)

        cv2.imshow("Result", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()