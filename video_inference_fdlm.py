"""inference fdlm on given video(or webcam)
"""
import os
import cv2

from scrfd import ScrfdFaceDetector

#VIDEO_SOURCE = 0  # webcam
VIDEO_SOURCE = os.path.join("data", "example", "sub_MUTA-FatBoyGang.mp4")


INT = False
DEBUG_INFO=False

SAVE_VIDEO = True
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
#        if bboxes.shape[0] != 1:
#            print("WARNING bboxes has shape: {}, select first box".format(bboxes.shape))
#        if lms.shape[0] == 0:
#            print("no face det")
#            continue
        fd.plot(im2show, bboxes, lms, num=False)
        # imshow
        cv2.imshow('im2show', im2show)
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