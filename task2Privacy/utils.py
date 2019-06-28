import time
import cv2
import os

def rounded_seconds(t):
    return str(round(t, 3))


def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def convertToGRAY(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def get_new_save_path(save_dir, current_path):
    return os.path.join(save_dir, os.path.basename(current_path))


def show_image_and_wait(img): 
    window = cv2.imshow('Show Image', img)
    cv2.waitKey(0)
    cv2.destroyWindow(window)