import cv2
from ultralytics import YOLO

def get_frame(f_num):
    vid_path = "../resources/capstone01.mp4"
    cap = cv2.VideoCapture(vid_path)
    frame = None

    for i in range(0, 500):
        _, frame = cap.read()
        if i < f_num:
            continue
        cap.release()
        return frame

def test():
    '''
    Frame list:
[117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 
154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
    '''
    knife_model = "models/knifeDetector.pt"
    vid_path = "../resources/capstone01.mp4"
    model = YOLO(knife_model)

    frame = get_frame(163)

    res = model.predict(source=frame, show=False, conf=0.5)

    boxes = []
    for r in res:
        boxes = r.boxes.xyxy[0]

    print(boxes)

    boxes = [int(x) for x in boxes]
    print(boxes)

    x, y, x2, y2 = int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3])
    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("Result", frame)
    cv2.waitKey()

# COLE, THIS RUNS ON WEBCAM
def main():
    knife_model = "models/knifeDetector.pt"
    model = YOLO(knife_model)

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()

        res = model.predict(source=img, show=False, conf=0.5)

        try:
            boxes = []
            for r in res:
                boxes = r.boxes.xyxy[0]

            if len(boxes):
                boxes = [int(x) for x in boxes]

                x, y, x2, y2 = boxes[0], boxes[1], boxes[2], boxes[3]
                cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)

                cv2.imshow("Result", img)

        except:
            pass

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break


if __name__ == '__main__':
    # main()
    test()