import cv2
import mediapipe as mp

# classes
from framerate import CountsPerSec
from video import VideoGet
from classifier import Classifier
from alerts import handle_alerts

# determines the max coordinates from the list of positive results
def calc_max_coords(coords):
    max_coords = [0, 0, 0, 0]
    coords_index = -1

    for i, (x, y, w, h) in enumerate(coords):
        if w > max_coords[2]:
            max_coords[0] = x
            max_coords[1] = y
            max_coords[2] = w
            max_coords[3] = h
            coords_index = i

    return max_coords, coords_index

def main():
    # setup
    knife_casc_path = "knife_data/classifier/cascade.xml"
    # gun_casc_path = "gun_data/classifier/cascade.xml"
    vid_path = "../resources/capstone01.mp4"

    knife_casc = cv2.CascadeClassifier(knife_casc_path)

    # framerate counter
    cps = CountsPerSec().start()

    # video thread, src defaults to webcam, delay defaults to 33ms
    video_getter = VideoGet(src=vid_path).start()

    # used to determine if a new image should be saved or alert should be sent
    knife_counters = [0, 0, 0]
    # gun_counters = [0, 0, 0]

    # pose crap
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_draw = mp.solutions.drawing_utils

    while True:
        # read in frame and convert to grayscale
        img = video_getter.frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # pose crap
        # should probably add image preprocessing to video frame class
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(img_rgb)

        # begin classifier threads
        knife_classifier = Classifier(knife_casc, img_gray, num=1).start()
        # gun_classifier = Classifier(gun_casc_path, img_gray, num=2).start()

        # join classifier threads
        knife_classifier.join()
        # gun_classifier.join()

        knife_coords, knife_confidence_list = knife_classifier.get_classify_results()
        # gun_coords, gun_confidence_list = gun_classifier.get_classify_results()

        # determine index and coordinates of largest positive value
        max_knife_coords, knife_coords_index = calc_max_coords(knife_coords)
        # max_gun_coords, gun_coords_index = calc_max_coords(gun_coords)

        # handle knife classification
        if knife_coords_index > -1:
            knife_confidence = knife_confidence_list[knife_coords_index]
            knife_counters[0], knife_counters[1], knife_counters[2] = handle_alerts(knife_confidence, max_knife_coords, img, cps, knife_counters, "Knife")

        # handle gun classification
        # if gun_coords_index > -1:
        #     gun_confidence = gun_confidence_list[gun_coords_index]
        #     gun_counters[0], gun_counters[1], gun_counters[2] = handle_alerts(gun_confidence, max_gun_coords, img, cps, gun_counters, "Gun")

        # pose crap
        if pose_result.pose_landmarks:
            mp_draw.draw_landmarks(img, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for lm in pose_result.pose_landmarks.landmark:
                h, w, c = img.shape
                cx, cy = int(lm.x * w) , int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        # display frame rate
        print("Frame rate: ", cps.get_framerate())

        # print to screen
        cv2.imshow("Result", img)

        # increment framerate counter
        cps.increment()

        # handle terminating video feed
        if (cv2.waitKey(1) & 0xFF == ord('q')) or video_getter.stopped:
            video_getter.stop()
            knife_classifier.join()
            break

if __name__ == '__main__':
    main()