#-*- coding:utf-8 -*-

import os
import argparse
from .generate import build_net, generate_images, generate_videos

parser = argparse.ArgumentParser(description='s3df demo')
parser.add_argument('--save_dir', type=str, default='tmp/',
                    help='Directory for detect result')
parser.add_argument('--model', type=str,
                    default='./weights/s3fd.pth', help='trained model')
parser.add_argument('--thresh', default=0.6, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


if __name__ == '__main__':
    net = build_net(args.model)
    #generate_images(net, args.save_dir, args.thresh)
    generate_videos(net, args.save_dir, args.thresh)
