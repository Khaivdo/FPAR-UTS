import cv2
import time
import os
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
        cv2.imwrite(output_frames + "/frame%#05d.jpg" % (count+1), image)

        print ('Read a new frame: ', success)
        count += 1

if __name__=="__main__":
    # Get current working directory
    DIR = os.getcwd()
    # Video to extract
    input_video = os.path.join(DIR,"subject1.mp4")
    # Folder to save extracted frames
    output_frames = os.path.join(DIR,"Frames")

    start_time = 66400
    end_time = 97700
    # Cut a new video from input video given start and end time
    ffmpeg_extract_subclip(input_video, (start_time/1000), (end_time/1000), targetname="out.mp4")
    cut_video = os.path.join(DIR,"out.mp4")
    # Extract frames from the new video
    video_to_frames(cut_video, output_frames)