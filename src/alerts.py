from constants import HIGH_CER, MED_CER, LOW_CER
from image_utils import draw_rectangle, save_image

# handles alerts and drawing if sufficient confidence is determined
def handle_alerts(confidence, max_coords, img, cps, counters, type):
    img_frame, alert_frame_pot, alert_frame_pos = counters[0], counters[1], counters[2]
    occ_frame = cps.get_occurrence()

    if confidence > LOW_CER:
        draw_rectangle(max_coords, img, confidence)
    if confidence >= HIGH_CER:

        # saves an image with a half second separation between saves
        if (occ_frame - img_frame) >= 15:
            save_image(img)
            img_frame = occ_frame

        # prints a positive alert no more than once per second
        if (occ_frame - alert_frame_pos) >= 30:
            print(f"{type} found!")
            alert_frame_pos = occ_frame

    # prints a potential alert no more than once per second
    elif confidence > MED_CER and confidence < HIGH_CER:
        if (occ_frame - alert_frame_pot) >= 30:
            print(f"{type} potentially found!")
            alert_frame_pot = occ_frame

    return img_frame, alert_frame_pos, alert_frame_pot
