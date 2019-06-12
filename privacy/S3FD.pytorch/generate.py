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


def detect(net, img_path, thresh, save_dir):
    #img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1700 * 1200 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(img, None, None, fx=max_im_shrink,
                      fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    #image = cv2.resize(img, (640, 640))
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()
    
    print('detecting: {}'.format(img_path))
    t1 = time.time()
    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    result_image = blur_frame(img, detections, thresh, scale)

    t2 = time.time()
    print('done. timer: {}'.format(t2 - t1))
    cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), result_image)#img)


def blur_frame(frame, detections, thresh, scale):
    result_image = frame.copy()

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            #left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            j += 1
            #cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 20)
            sub_face = frame[int(pt[1]):int(pt[3]), int(pt[0]):int(pt[2])]
            sub_face = cv2.GaussianBlur(sub_face, (23, 23), 30)
            #merge
            result_image[int(pt[1]):int(pt[1]+sub_face.shape[0]), int(pt[0]):int(pt[0]+sub_face.shape[1])] = sub_face
            #face_file_name = "./blurredFace.jpg"
            #conf = "{:.3f}".format(score)
            #point = (int(left_up[0]), int(left_up[1] - 5))
            #cv2.putText(img, conf, point, cv2.FONT_HERSHEY_COMPLEX,
            #            0.6, (0, 255, 0), 1)

    return result_image


def build_net(model): 
    net = build_s3fd('test', cfg.NUM_CLASSES)
    #setting map_location to cpu will forcefully remap everything onto CPU
    net.load_state_dict(torch.load(model, map_location='cpu'))
    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    return net


def generate_images(net, save_dir, thresh):
    img_path = './img'
    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path) if x.endswith('jpg')]

    for path in img_list:
        detect(net, path, thresh, save_dir)
