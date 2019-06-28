import cv2
import numpy as np
import time
from utils import convertToGRAY, convertToRGB, rounded_seconds, show_image_and_wait

face_cascade = cv2.CascadeClassifier('./cv2data/haarcascade_frontalface_alt.xml')
profile_cascade = cv2.CascadeClassifier('./cv2data/haarcascade_profileface.xml')

def detect_and_blur(classifier, rgb_img, scaleFactor=1.1, minNeighbours=5):
    result_image = np.copy(rgb_img)
    detections = classifier.detectMultiScale(convertToGRAY(result_image), scaleFactor=1.2, minNeighbors=minNeighbours)
    for d in detections:
        x1, x2, y1, y2 = (d[0], d[0]+ d[2], d[1], d[1]+ d[3])
        result_image[y1:y2, x1:x2] = cv2.blur(result_image[y1:y2, x1:x2], (23, 23))

    return result_image


def detect_and_blur_faces(rgb_img, scaleFactor=1.1, minNeighbours=5):
    result_image = np.copy(rgb_img)
    result_image = detect_and_blur(face_cascade, result_image)
    result_image = detect_and_blur(profile_cascade, result_image, scaleFactor=1.05)
    return result_image


def create_blurred_video(vid_path, output_path):
    print('\twill save protected video to {}'.format(output_path))
    # create video capture and get video info
    vidcap = cv2.VideoCapture(vid_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    size = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vidwrite = cv2.VideoWriter(output_path, fourcc, fps, size)

    # Read initial frame
    count = 1
    success, frame = vidcap.read()
    while success:
        t1 = time.time()
        result_frame = detect_and_blur_faces(frame)
        # Write video
        vidwrite.write(result_frame)
        t2 = time.time()
        print('\tframe {} done in {}'.format(count, rounded_seconds(t2 - t1)))
        # Read video
        success, frame = vidcap.read()
        count += 1

    vidcap.release()
    vidwrite.release()