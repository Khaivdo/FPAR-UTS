import numpy as np
import cv2
import os
import argparse
import shutil

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--org', type=str, default='./org')
    parser.add_argument('--des', type=str, default='./des')
    args = parser.parse_args()
    input_path = './images/adversarials/'
    files=os.listdir(args.org)
    for file in files:
        print('Preparing...')
        in_path=os.path.join(args.org,file)
        out_path=os.path.join(args.des,file)
        cap = cv2.VideoCapture(in_path)
        nbFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        # codec = cap.get(cv2.CAP_PROP_FOURCC)
        out = cv2.VideoWriter(out_path, int(codec), int(fps), (int(width), int(height)))

        f = 0
        cmd = 'python p-fgsm.py --path=./images'
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame=frame[..., ::-1]
                cv2.imwrite(os.path.join('./images', str(f).zfill(6) + '.jpg'), frame)
                f = f + 1
                # cv2.imshow('test',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        os.system(cmd)
        pic_all = os.listdir(input_path)
        pic_all.sort()
        for pic_name in pic_all:
            img_name = input_path + pic_name
            frame = cv2.imread(img_name)
            out.write(frame)
        log_path=os.path.join(args.des,file[:-4])
        shutil.move('./log.txt',log_path+'.txt')
        pics= os.listdir('./images')
        for pic in pics:
            if '.jpg' in pic:
                os.remove(os.path.join('./images',pic))
        pics = os.listdir('./images/adversarials')
        for pic in pics:
            if '.jpg' in pic:
                os.remove(os.path.join('./images/adversarials', pic))

        cap.release()
        out.release()
        cv2.destroyAllWindows()
