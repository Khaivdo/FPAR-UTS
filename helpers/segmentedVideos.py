import cv2
import time
import os
import random
import numpy
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

activities=["chat", "write", "whiteboard", "wash", "clean", "drink", "dryer", "machine", "microwave",
            "mobile", "paper", "print", "read", "shake", "staple", "take", "typeset", "walk"]

def video_to_frames_train_val(cut_video, output_frames,  current_directory):
    train = os.path.join(current_directory + "/train")
    val = os.path.join(current_directory + "/val")
    try:
        os.chdir(train)
        if not os.path.isdir(output_frames):
            os.mkdir(output_frames)
        os.chdir(val)
        if not os.path.isdir(output_frames):
            os.mkdir(output_frames)
        # os.mkdir(output_frames)

    except OSError:
        pass
    os.chdir(current_directory)

    train_act = os.path.join(train, output_frames)
    val_act = os.path.join(val, output_frames)
    vidcap = cv2.VideoCapture(cut_video)
    count = 0
    success = True
    count1 = 0
    i = random.randint(1, 5)
    count2 = 1
    while success:
        # Read video
        success, image = vidcap.read()
        # Generate frame

        if success:
            print('Read a new frame from ', output_frames, ': ', success)

            # if count % 25 == 0:
            #     name = "/frame%#04d" % count1 + "%#05d.jpg" % (count+1)
            #     if not os.path.isfile(val_act + name):
            #         cv2.imwrite(val_act + "/frame%#04d" % count1 + "%#05d.jpg" % (count+1), image)
            #     elif os.path.isfile(val_act + name):
            #         count1 += 1
            #         cv2.imwrite(val_act + "/frame%#04d" % count1 + "%#05d.jpg" % (count + 1), image)

            if count % 5 == 0:
                name = "/frame%#05d" % count1 + "%#05d.jpg" % (count+1)
                if (count2==i):
                    if not os.path.isfile(val_act + name):
                        cv2.imwrite(val_act + "/frame%#05d" % count1 + "%#05d.jpg" % (count + 1), image)
                    elif os.path.isfile(val_act + name):
                        count1 += 1
                        cv2.imwrite(val_act + "/frame%#05d" % count1 + "%#05d.jpg" % (count + 1), image)
                else:
                    if not os.path.isfile(train_act + name):
                        cv2.imwrite(train_act + "/frame%#05d" % count1 + "%#05d.jpg" % (count + 1), image)
                    elif os.path.isfile(train_act + name):
                        count1 += 1
                        cv2.imwrite(train_act + "/frame%#05d" % count1 + "%#05d.jpg" % (count + 1), image)

                count2 += 1
                if count2 == 6:
                    count2 = 1
                    i = random.randint(1, 5)
        count += 1

def video_to_frames_test(cut_video, output_frames, current_directory):
    test = os.path.join(current_directory + "/test")

    try:
        os.chdir(test)
        if not os.path.isdir(output_frames):
            os.mkdir(output_frames)
        os.chdir(test)
        if not os.path.isdir(output_frames):
            os.mkdir(output_frames)

    except OSError:
        pass
    os.chdir(current_directory)

    test_act = os.path.join(test, output_frames)
    vidcap = cv2.VideoCapture(cut_video)
    count = 0
    success = True
    i = 0
    while success:
        # Read video
        success, image = vidcap.read()
        # Generate frame

        if success:
            print('Read a new frame from ', output_frames, ': ', success)

            if count % 25 == 0:
                name = "/frame%#04d" % i + "%#05d.jpg" % (count+1)
                if not os.path.isdir(test_act + name):
                    cv2.imwrite(test_act + "/frame%#04d" % i + "%#05d.jpg" % (count+1), image)
                else:
                    i += 1

        count += 1

if __name__=="__main__":
    # Get current working directory
    DIR = os.getcwd()
    dir_list = []
    dir_list2 = []
    list = [1, 2, 3, 4]
    list2 = []
    list3 = []
    Path_video_list = []

    try:
        os.mkdir("train")
        os.mkdir("val")
        os.mkdir("test")
    except OSError:
        pass

    for i in list:
        Path = os.path.join(DIR, "FPVO-segmented_videos%#01d" % i)
        if i < 4:
            list2 = numpy.add(numpy.multiply((i-1), 3), list[0:3])
        else:
            list2 = [10]

        list3 = [11, 12]
        for j in list2:
            Path_video = os.path.join(Path, "subject_%#01d_segments" % j)
            Path_video_list.append(Path_video)
            for video in os.listdir(Path_video):
                # Video to extract
                input_video = os.path.join(Path_video, video)
                # Folder to save extracted frames
                dir_list.append(input_video)
        if i == 4:
            for k in list3:
                Path_video = os.path.join(Path, "subject_%#01d_segments" % k)
                Path_video_list.append(Path_video)
                for video in os.listdir(Path_video):
                    # Video to extract
                    input_video = os.path.join(Path_video, video)
                    # Folder to save extracted frames
                    dir_list2.append(input_video)

    for vid in dir_list:
        # Folder to save extracted frames
        output_frames = vid.replace(".MP4", "")
        for i in range(0, 18):

            if activities[i] in output_frames:
                video_to_frames_train_val(vid, activities[i], DIR)

    # for vid in dir_list2:
    #     # Folder to save extracted frames
    #     output_frames = vid.replace(".MP4", "")
    #     for i in range(0, 18):
    #
    #         if activities[i] in output_frames:
    #             video_to_frames_test(vid, activities[i], DIR)


    #print(dir_list)