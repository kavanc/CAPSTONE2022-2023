import cv2
from ultralytics import YOLO

FONT = cv2.FONT_HERSHEY_SIMPLEX

def get_frame(f_num):
    vid_path = "../resources/multi_knife.mp4"
    cap = cv2.VideoCapture(vid_path)
    frame = None

    for i in range(0, 500):
        _, frame = cap.read()
        if i < f_num:
            continue
        cap.release()
        return frame

def draw_header(img, name, x1, y1, x2):
    y = y1 - 30
    # cv2.rectangle(img, (x1, y), (x2, y1), (0, 0, 255), 2)
    cv2.putText(img, name, (x1, y1 - 10), FONT, 0.9, (0, 0, 255), 2)

'''

frame 51
'''

def test():
    knife_model = "models/knifeDetector.pt"
    gun_model = "models/KavanGunbest.pt"
    vid_path = "../resources/multi_knife.mp4"
    model = YOLO(knife_model)
    # model = YOLO(gun_model)

    frame = get_frame(51)

    res = model.predict(source=frame, show=False, conf=0.1)

    boxes = []
    bb_list = []
    for r in res:
        boxes = r.boxes.xyxy

    for i in range(len(boxes)):
        bb_list.append([int(x) for x in boxes[i]])

    for b in bb_list:
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        draw_header(frame, "Knife", x1, y1, x2)

    cv2.imshow("Result", frame)
    cv2.waitKey()

# COLE, THIS RUNS ON WEBCAM
def main():
    # knife_model = "models/knifeDetector.pt"
    gun_model = "models/KavanGunbest.pt"
    # model = YOLO(knife_model)
    model = YOLO(gun_model)

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()

        res = model.predict(source=img, show=False, conf=0.7)

        try:
            boxes = []
            bb_list = []
            for r in res:
                boxes = r.boxes.xyxy

            if len(boxes):
                for i in range(len(boxes)):
                    bb_list.append([int(x) for x in boxes[i]])

                for b in bb_list:
                    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    draw_header(img, "Knife", x1, y1, x2)

        except:
            pass

        cv2.imshow("Result", img)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break


if __name__ == '__main__':
    main()
    # test()