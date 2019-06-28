import cv2
import time
import os
import numpy
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def video_to_frames(cut_video, output_frames):
    try:
        os.mkdir(output_frames)
    except OSError:
        pass
    vidcap = cv2.VideoCapture(cut_video)
    count = 0
    success = True
    while success:
        # Read video
        success, image = vidcap.read()
        # Generate frame
        print ('Read a new frame from ', output_frames, ': ', success)
        if success:
            cv2.imwrite(output_frames + "/frame%#05d.jpg" % (count+1), image)
        count += 1

if __name__=="__main__":
    # Get current working directory
    DIR = os.getcwd()
    dir_list = []
    list = [1, 2, 3, 4]
    list2 = []
    for i in list:
        Path = os.path.join(DIR, "FPVO-segmented_videos%#01d" % i)
        if i < 4:
            list2 = numpy.add(numpy.multiply((i-1), 3), list[0:3])
        else:
            list2 = [10, 11, 12]

        for j in list2:
            Path_video = os.path.join(Path, "subject_%#01d_segments" % j)
            for video in os.listdir(Path_video):
                # Video to extract
                input_video = os.path.join(Path_video, video)
                # Folder to save extracted frames
                dir_list.append(input_video)

    for vid in dir_list:
        # Folder to save extracted frames
        output_frames = vid.replace(".MP4", "")
        video_to_frames(vid, output_frames)
    #print(dir_list)