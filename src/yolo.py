import cv2
from ultralytics import YOLO
from model_thread import MThread
from framerate import CountsPerSec
from ultralytics.yolo.utils.plotting import Annotator

FONT = cv2.FONT_HERSHEY_SIMPLEX

def get_frame(f_num, path):
    # vid_path = "../resources/multi_knife.mp4"
    cap = cv2.VideoCapture(path)
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

def draw_framerate(img, fps):
    cv2.putText(img, str(int(fps)), (10, 25), FONT, 0.8, (0, 255, 0), 2)

'''

frame 51
'''

def test():
    knife_model = "models/knifeDetector.pt"
    gun_model = "models/KavanGunbest.pt"
    vid_path = "../resources/multi_knife.mp4"
    # vid_path = "../resources/capstone01.mp4"

    model = YOLO(knife_model)
    # model = YOLO(gun_model)

    # frame = get_frame(167, vid_path)
    frame = get_frame(51, vid_path)

    res = model.predict(source=frame, show=False, conf=0.1)

    for r in res:
        annotator = Annotator(frame)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls

            annotator.box_label(b, f"{model.names[int(c)]} {box.conf[0]:.2f}", (0, 0, 255))

    img = annotator.result()
    cv2.imshow('RES', img)
    cv2.waitKey()

def thread_test():
    knife_model_path = "models/knifeDetector.pt"
    gun_model_path = "models/KavanGunbest.pt"
    
    knife_model = YOLO(knife_model_path)
    gun_model = YOLO(gun_model_path)

    cap = cv2.VideoCapture(0)

    cps = CountsPerSec().start()

    while True:
        ret, img = cap.read()

        knife_thread = MThread(img, knife_model, 0.7).start()
        gun_thread = MThread(img, gun_model, 0.5).start()

        knife_thread.join()
        gun_thread.join()

        knife_res = knife_thread.get_res()
        gun_res = gun_thread.get_res()

        # res = model.predict(source=img, show=False, conf=0.7)

        try:
            knife_boxes = []
            gun_boxes = []

            knife_bb = []
            gun_bb = []
            for r in knife_res:
                knife_boxes = r.boxes.xyxy

            for r in gun_res:
                gun_boxes = r.boxes.xyxy

            if len(knife_boxes):
                for i in range(len(knife_boxes)):
                    knife_bb.append([int(x) for x in knife_boxes[i]])

                for b in knife_bb:
                    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    draw_header(img, "Knife", x1, y1, x2)

            if len(gun_boxes):
                for i in range(len(gun_boxes)):
                    gun_bb.append([int(x) for x in gun_boxes[i]])

                for b in gun_bb:
                    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    draw_header(img, "Gun", x1, y1, x2)

        except:
            pass

        draw_framerate(img, cps.get_framerate())
        cv2.imshow("Result", img)
        cps.increment()

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break


# COLE, THIS RUNS ON WEBCAM
def main_failed():
    knife_model = "models/knifeDetector.pt"
    gun_model = "models/KavanGunbest.pt"
    model2 = YOLO(knife_model)
    model = YOLO(gun_model)

    cap = cv2.VideoCapture(0)

    cps = CountsPerSec().start()

    while True:
        ret, img = cap.read()

        res = model.predict(source=img, show=False, conf=0.7)
        model2.predict(source=img, show=False, conf=0.7)

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

        draw_framerate(img, cps.get_framerate())
        cv2.imshow("Result", img)
        cps.increment()

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

def main():
    knife_model = "models/knifeDetector.pt"
    model = YOLO(knife_model)
    cap = cv2.VideoCapture(0)
    cps = CountsPerSec().start()

    while True:
        ret, img = cap.read()

        res = model.predict(source=img, show=False, conf=0.1)

        for r in res:
            annotator = Annotator(img)

            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls

                annotator.box_label(b, f"{model.names[int(c)]} {box.conf[0]:.2f}", (0, 0, 255))

        img = annotator.result()
        draw_framerate(img, cps.get_framerate())
        cv2.imshow("Result", img)
        cps.increment()

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break


if __name__ == '__main__':
    main()