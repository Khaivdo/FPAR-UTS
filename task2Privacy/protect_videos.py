import os
import time
import argparse
from S3FD.generate import build_net, generate_video
from utils import rounded_seconds, get_new_save_path
from cv2face import create_blurred_video

parser = argparse.ArgumentParser(description='Privacy protection')
parser.add_argument('-i', type=str, default='./input', help='Directory that contains the test video segments')
parser.add_argument('-o', type=str, default='./output', help='Output directory that will contain the corresponding protected videos')
parser.add_argument('-mode', type=str, default='S3FD', help='can either be S3FD or cv2haar')
args = parser.parse_args()

isCv2Face = args.mode == 'cv2haar'

def get_new_save_path(save_dir, current_path):
    return os.path.join(save_dir, os.path.basename(current_path))


def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def protect_videos(input_dir, output_dir, net=0, thresh=0):
    timeStart = time.time()
    print('Commencing protection of videos within directory {}...'.format(input_dir))
    create_dir_if_not_exists(output_dir)
    for dirpath, dirnames, filenames in os.walk(input_dir):
        print(dirpath)
        video_files = (file for file in filenames if file.lower().endswith('mp4'))
        new_dirpath = dirpath.replace(input_dir, output_dir)
        create_dir_if_not_exists(new_dirpath)
        for video_file in video_files:
            time1 = time.time()
            video_path = os.path.join(dirpath, video_file)
            print('Protecting video {}...'.format(video_path))
            if isCv2Face:
                output_path = get_new_save_path(new_dirpath, video_path)
                create_blurred_video(video_path, output_path)
            else:
                generate_video(net, video_path, new_dirpath, thresh)

            time2 = time.time()
            print('Done. time taken: {}'.format(rounded_seconds(time2 - time1)))
            print('Cumulative time spent: {}'.format(rounded_seconds(time2 - timeStart)))


    timeFinal = time.time()
    print('All videos have been protected.')
    print('Total time taken: {}'.format(rounded_seconds(timeFinal-timeStart)))


if __name__ == '__main__':
    if not os.path.exists(args.i):
        raise TypeError('Input directory does not exist!')

    if not isCv2Face:
        print('using default S3FD')
        thresh = 0.6
        model = './S3FD/weights/s3fd.pth'
        net = build_net(model)
        protect_videos(args.i, args.o, net=net, thresh=thresh)
    else:
        print('using cv2 haar')
        protect_videos(args.i, args.o)