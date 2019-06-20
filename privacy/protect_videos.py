import os
import time
import argparse
from S3FD.generate import build_net, generate_video

parser = argparse.ArgumentParser(description='Privacy protection')
parser.add_argument('-i', type=str, default='./input', help='Directory that contains the test video segments')
parser.add_argument('-o', type=str, default='./output', help='Output directory that will contain the corresponding protected videos')
args = parser.parse_args()


def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def protect_videos(net, input_dir, output_dir, thresh):
    timeStart = time.time()
    print('Commencing protection of videos within directory {}...'.format(input_dir))
    create_dir_if_not_exists(output_dir)
    for dirpath, dirnames, filenames in os.walk(input_dir):
        video_files = (file for file in filenames if file.lower().endswith('jpeg'))
        new_dirpath = dirpath.replace(input_dir, output_dir)
        create_dir_if_not_exists(new_dirpath)
        for video_file in video_files:
            time1 = time.time()
            video_path = os.path.join(dirpath, video_file)
            print('Protecting video {}...'.format(video_path))
            generate_video(net, video_path, new_dirpath, thresh)
            time2 = time.time()
            print('Done. time taken: {}'.format(time2 - time1))
            print('Cumulative time spent: {}'.format(time2 - timeStart))


    timeFinal = time.time()
    print('All videos have been protected.')
    print('Total time taken: {}'.format(timeStart-timeFinal))


if __name__ == '__main__':
    if not os.path.exists(args.i):
        raise TypeError('Input directory does not exist!')

    thresh = 0.6
    model = './S3FD/weight/s3fd.pth'
    net = build_net(model)
    protect_videos(net, args.i, args.o, thresh)