#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image

from data.config import cfg
from s3fd import build_s3fd
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr


use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def to_pillow_img_arr(imgArr):
    pillowImg = Image.fromarray(imgArr)

    if pillowImg.mode == 'L':
        pillowImg = pillowImg.convert('RGB')

    return np.array(pillowImg)


def create_tensor_module(image):
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    return x
    

def blur_image(image, detections, thresh, scale):
    result_image = image.copy()
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            bounding_box = (detections[0, i, j, 1:] * scale).cpu().numpy()
            result_image = blur_detection(image, bounding_box)
            j += 1

    return result_image
    

def blur_detection(image, bounding_box):
    # not sure what pt stands for
    pt = bounding_box
    # get detected image area
    image_area = image[int(pt[1]):int(pt[3]), int(pt[0]):int(pt[2])]
    # blur the detected area
    blurred_image = cv2.GaussianBlur(image_area, (23, 23), 30)
    # apply the blur to the image
    image[int(pt[1]):int(pt[1]+blurred_image.shape[0]), int(pt[0]):int(pt[0]+blurred_image.shape[1])] = blurred_image
    return image


def resize_image(cv2Img):
    rgbImgArr = cv2.cvtColor(cv2Img, cv2.COLOR_BGR2RGB)
    img = to_pillow_img_arr(rgbImgArr)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1700 * 1200 / (img.shape[0] * img.shape[1]))
    resizedImg = cv2.resize(img, None, None, fx=max_im_shrink,
                      fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    tensor_scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])
    return resizedImg, tensor_scale


def detect(net, img, thresh):
    image, scale = resize_image(img)
    image_tensor = create_tensor_module(image)

    result = net(image_tensor)
    detections = result.data

    result_image = blur_image(img, detections, thresh, scale)
    return result_image


def detect_from_path(net, save_dir, img_path, thresh):
    print('detecting: {}'.format(img_path))
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    t1 = time.time()
    result_image = detect(net, img, thresh)
    cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), result_image)
    t2 = time.time()
    print('done. timer: {}'.format(t2 - t1))


def generate_images(net, save_dir, thresh):
    img_path = './img'
    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path) if x.endswith('jpg')]

    for path in img_list:
        detect_from_path(net, save_dir, path, thresh)


def generate_video(net, vid_path, thresh):
    frames = video_to_frames(vid_path)
    for index, frame in enumerate(frames):
        # frames[index] = detect(net, frame, thresh)
        print(index)



def build_net(model): 
    net = build_s3fd('test', cfg.NUM_CLASSES)

    if use_cuda:
        state_dict = torch.load(model)
        net.cuda()
        cudnn.benckmark = True
    else:
        #setting map_location to cpu will forcefully remap everything onto CPU
        state_dict = torch.load(model, map_location='cpu')

    net.load_state_dict(state_dict)
    net.eval()
    return net


def video_to_frames(vid_path, output_path):
    # create video capture and get video info
    vidcap, fps, size = get_video_capture(vid_path)
    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vidwrite = cv2.VideoWriter(output_path, fourcc, fps, size)
   
    frames = []
    success = True
    while success:
        # Read video
        success, frame = vidcap.read()
        # Write video
        vidwrite.write(frame)

    vidcap.release()
    vidwrite.release()
    return frames


def get_video_capture(vid_path):
    vidcap = cv2.VideoCapture(vid_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    size = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return vidcap, fps, size