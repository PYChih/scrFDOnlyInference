"""inference fdlm on given video(or webcam)
"""
import os
import cv2

from scrfd import ScrfdFaceDetector
from blink import blink

#VIDEO_SOURCE = 0  # webcam
#VIDEO_SOURCE = os.path.join("data", "example", "sub_MUTA-FatBoyGang.mp4")
VIDEO_SOURCE = "/home/py/workspace/GazeV3/GazeV3OnlyInference/data/gaze.mp4"


INT = False
DEBUG_INFO=False

SAVE_VIDEO = False
OUTPUT_VIDEO_DIR = os.path.join("data", "result")
if isinstance(VIDEO_SOURCE, int):
    OUTPUT_VIDEO_NAME = os.path.basename(str(VIDEO_SOURCE)).strip(".mp4") + "_result.mp4"
else:
    OUTPUT_VIDEO_NAME = os.path.basename(VIDEO_SOURCE).strip(".mp4") + "_result.mp4"
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
OUTPUT_VIDEO_FILEPATH = os.path.join(OUTPUT_VIDEO_DIR, OUTPUT_VIDEO_NAME)
OUTPUT_IMAGE_BASENAME = 'test'  # press 's' to save

# helper for video writer
def video_writer_initial(cap, output_filepath):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print("---write result video to: " + output_filepath)
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (frame_width, frame_height))
    return out

# CONSTANTS
WAIT_KEY_DELAY = 1
STOP_KEY_DELAY = 0
STOP_KEY_CODE = ord(' ')  # press 'sapce' to stop at the moment
QUIT_KEY_CODE = ord('q')  # press 'q' to quit
SAVE_KEY_CODE = ord('s')  # press 's' to save current image
SAVE_IMAGE_CNT = 0  # count for many times image save

if __name__ == "__main__":
    # model initial
    fd = ScrfdFaceDetector(int=INT, conf_thr=0.5, iou_thr=0.1, debug=DEBUG_INFO)
    bk = blink()
    # capture initial
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    # writer initial
    if SAVE_VIDEO:
        writer = video_writer_initial(cap, OUTPUT_VIDEO_FILEPATH)
    # main-loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        else:
            im2show = frame.copy()
        # fdlm inference
        bboxes, lms = fd.inference(frame, debug=DEBUG_INFO)
#        print("bboxes.shape: {}, lms.shape: {}".format(bboxes.shape, lms.shape))
        if bboxes.shape[0] != 1:
            print("WARNING bboxes has shape: {}, select first box".format(bboxes.shape))
        if lms.shape[0] == 0:
            print("no face det")
            continue
        #########################
        netin_images = bk.outer_preprocess(frame, lms[0])
        netin_lst = bk._preprocess(netin_images)
        netout = bk._invoke(netin_lst)
        eyes_state = bk._postprocess(netout)
        im2show_lst = []
        for idx, (eye_state, image) in enumerate(zip(eyes_state, netin_images[:2])):
            if eye_state == 'open':
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            rs_image = cv2.resize(image, (224, 224))
            cv2.rectangle(rs_image, (0, 0), (224, 224), color , thickness=5)
            if idx == 0:
                lm_idx = 1  # left
            else:
                lm_idx = 0  # right
            pos = lms[0, lm_idx]
            length = 20
            cv2.rectangle(im2show, 
                          (int(pos[0] - length), int(pos[1] - length)), 
                          (int(pos[0] + length), int(pos[1] + length)), 
                          color , 
                          thickness=5)
            im2show_lst.append(rs_image)
        #########################
        fd.plot(im2show, bboxes, lms, num=False)
        # imshow
        cv2.imshow('im2show', im2show)
        cv2.imshow("left", im2show_lst[0])
        cv2.imshow("right", im2show_lst[1])
        key = cv2.waitKey(WAIT_KEY_DELAY)
        if key == QUIT_KEY_CODE:
            break
        elif key == STOP_KEY_CODE:
            cv2.waitKey(STOP_KEY_DELAY)
        elif key == SAVE_KEY_CODE:
            output_image_filename = f"{OUTPUT_IMAGE_BASENAME}_{SAVE_IMAGE_CNT}.jpg"
            print("write cap image to: " + output_image_filename)
            cv2.imwrite(output_image_filename, im2show)
            SAVE_IMAGE_CNT+=1
        if SAVE_VIDEO:
            writer.write(im2show)
    cap.release()
    cv2.destroyAllWindows()
    if SAVE_VIDEO:
        writer.release()