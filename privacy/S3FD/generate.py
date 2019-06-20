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

from .data.config import cfg
from .s3fd import build_s3fd
from torch.autograd import Variable
from .utils.augmentations import to_chw_bgr


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
            result_image = blur_detection(result_image, bounding_box)
            j += 1

    return result_image
    

def blur_detection(image, bounding_box):
    # not sure what pt stands for
    pt = bounding_box
    # get detected image area
    image_area = image[int(pt[1]):int(pt[3]), int(pt[0]):int(pt[2])]
    # blur the detected area
    blurred_image = cv2.GaussianBlur(image_area, (23, 23), 30)
    if blurred_image is not None:
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


def detect_and_blur(net, img, thresh):
    image, scale = resize_image(img)
    image_tensor = create_tensor_module(image)

    result = net(image_tensor)
    detections = result.data

    result_image = blur_image(img, detections, thresh, scale)
    return result_image


def generate_image(net, save_dir, img_path, thresh):
    print('Protecting Image {}...'.format(img_path))
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    t1 = time.time()
    result_image = detect_and_blur(net, img, thresh)
    cv2.imwrite(get_new_save_path(save_dir, img_path), result_image)
    t2 = time.time()
    print('done. timer: {}'.format(t2 - t1))


def generate_video(net, vid_path, save_dir, thresh):
    output_path = get_new_save_path(save_dir, vid_path)
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
        print('\tframe{}'.format(count))
        result_frame = detect_and_blur(net, frame, thresh)
        # Write video
        vidwrite.write(result_frame)
        print('\tdone in {}'.format(time.time() - t1))
        # Read video
        success, frame = vidcap.read()
        count += 1

    vidcap.release()
    vidwrite.release()


def get_new_save_path(save_dir, current_path):
    return os.path.join(save_dir, os.path.basename(current_path))


def generate_images(net, save_dir, thresh):
    img_dir = './img'
    timeStart = time.time()
    print('Commencing protection of videos...')
    for filename in os.listdir(img_dir):
        if filename.lower().endswith('jpg'):
            img_path = os.path.join(img_dir, filename)
            time1 = time.time()
            print('Protecting image {}...'.format(img_path))
            generate_image(net, save_dir, img_path, thresh)
            time2 = time.time()
            print('Done. time taken: {}'.format(time2 - time1))
            print('Cumulative time spent: {}'.format(time2 - timeStart))

    timeFinal = time.time()
    print('All images have been protected.')
    print('Total time taken: {}'.format(timeStart-timeFinal))


def generate_videos(net, save_dir, thresh):
    timeStart = time.time()
    print('Commencing protection of videos...')
    video_dir = './video'
    video_out_dir = './video_out'
    for filename in os.listdir(video_dir):
        if filename.lower().endswith('mp4'):
            video_path = os.path.join(video_dir, filename)
            time1 = time.time()
            print('Protecting video {}...'.format(video_path))
            generate_video(net, video_path, save_dir, thresh)
            time2 = time.time()
            print('Done. time taken: {}'.format(time2 - time1))
            print('Cumulative time spent: {}'.format(time2 - timeStart))

    timeFinal = time.time()
    print('All videos have been protected.')
    print('Total time taken: {}'.format(timeStart-timeFinal))
    

def build_net(model): 
    net = build_s3fd('test', cfg.NUM_CLASSES)

    if use_cuda:
        print('Cuda is available. Computations will be done on GPU')
        state_dict = torch.load(model)
        net.cuda()
        cudnn.benckmark = True
    else:
        print('Cuda is not available. Computations will be done on CPU')
        #setting map_location to cpu will forcefully remap everything onto CPU
        state_dict = torch.load(model, map_location='cpu')

    net.load_state_dict(state_dict)
    net.eval()
    return net
