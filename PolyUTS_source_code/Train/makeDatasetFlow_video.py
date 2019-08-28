import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import sys
import cv2
import re


def gen_split(root_dir, extractFrames):
    """
    Input:
        Dataset directory
        extractFrames: Training time can be reduced by 1/3 if frames are extracted
    Output:
        Lists of video paths, their ground truths and frame folders
    """
    Dataset = []
    Labels = []
    Frames = []
    New_dir = ['kenya_3', 'kenya_4', 'oxford_3', 'oxford_4', 'oxford_5']
    label_name = ['chat', 'clean', 'drink', 'dryer', 'machine', 'microwave', 'mobile', 'paper', 'print', 'read',
                  'shake', 'staple', 'take', 'typeset', 'walk', 'wash', 'whiteboard', 'write']
    for subject_folder in os.listdir(root_dir):
        dir = os.path.join(root_dir, subject_folder)
        for target in sorted(os.listdir(dir)):
            if target.endswith(".MP4"):                                             # Skip directory if there is any
                if subject_folder in New_dir:                                       # Videos from test_dataset_1
                    video_name = target.split(".")[0]
                    num_label = video_name.split("-")[0]
                    label = re.split('(\d+)',num_label)[-1]
                elif subject_folder not in New_dir:                                 # Videos from segmented folders
                    video_name = target.split(".")[0]
                    label = video_name.split("_")[-1]

                if label == 'wave' or label == 'open':                              # Activities not assessed
                    continue

                if extractFrames:                                                   # Extract frames
                    if not os.path.exists(os.path.join(dir, video_name)):
                        os.makedirs(os.path.join(dir, video_name))
                        vidcap = cv2.VideoCapture(os.path.join(dir, target))
                        success = True
                        count = 0
                        while success:
                            os.chdir(os.path.join(dir, video_name))
                            success, image = vidcap.read()
                            if success:
                                cv2.imwrite("%d.jpg" % (count), image)              # Save these frames
                                count += 1
                # else:                                          # Else remove all folders containing frames extracted
                #     shutil.rmtree(os.path.join(dir, video_name))

                for i in range(0, len(label_name)):
                    if label == label_name[i]:
                        label_id = i
                        break
                    if label == "dry":
                        label_id = 3
                        break
                Dataset.append(os.path.join(dir, target))
                Labels.append(label_id)
                Frames.append(os.path.join(dir, video_name))

    return Dataset, Labels, Frames


def optical_flow(prev_img, curr_img, spatial_transform, inpSeq):
    """
    Generate Optical Flow using OpenCV
    """
    flow = cv2.calcOpticalFlowFarneback(prev_img, curr_img, None, 0.702, 5, 10, 2, 7, 1.5,
                                        cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow = np.clip(flow, -15, 15)
    norm_flow = ((flow + 15) * 8.5).astype(int)
    flow_x = norm_flow[..., 0]
    flow_y = norm_flow[..., 1]
    # flow_x = Image.fromarray(flow_x)
    # flow_y = Image.fromarray(flow_y)
    # If TypeError: Cannot handle this data type occurs, please change to the following code
    flow_x = Image.fromarray(flow_x.astype(np.uint8))
    flow_y = Image.fromarray(flow_y.astype(np.uint8))
    inpSeq.append(spatial_transform(flow_x.convert('L'), inv=True, flow=True))
    inpSeq.append(spatial_transform(flow_y.convert('L'), inv=False, flow=True))

    return inpSeq


def read_img(numFrame, frames_idx, i):
    """
    If Frames are extracted, previous image and current image are returned from stored images
    """
    os.chdir(frames_idx)                                                        # Change current dir to frame dir
    if os.path.exists(os.path.join(frames_idx, "%d.jpg" % (i-1))):              # if numFrame >= stackSize
        prev_img = cv2.imread("%d.jpg" % (i-1))  # n >= 5
        prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        if os.path.exists(os.path.join(frames_idx, "%d.jpg" % i)):              # if numFrame > stackSize
            curr_img = cv2.imread("%d.jpg" % i)  # n >= 6
            curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        else:                                                                   # if numFrame = stackSize
            curr_img = prev_img
    else:                                                                       # if numFrame < stackSize
        i = numFrame - 1
        prev_img = cv2.imread("%d.jpg" % (int(np.floor(i - 1))))
        prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        curr_img = prev_img

    return prev_img, curr_img


def capture_img(numFrame, video_idx, i):
    """
    If Frames are captured, previous image and current image are returned from video directly
    """
    videoCapture = cv2.VideoCapture(video_idx)
    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, i - 1)                            # Capture from frame 0
    prev_success, prev_img = videoCapture.read()
    if prev_success:                                                            # if numFrame >= stackSize
        prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        videoCapture.set(cv2.CAP_PROP_POS_FRAMES, i)
        curr_success, curr_img = videoCapture.read()
        if curr_success:                                                        # if numFrame > stackSize
            curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        else:                                                                   # if numFrame = stackSize
            curr_img = prev_img

    else:                                                                       # if numFrame < stackSize
        i = numFrame - 1
        videoCapture.set(cv2.CAP_PROP_POS_FRAMES, i)  # Last frame captured
        prev_success, prev_img = videoCapture.read()
        prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        curr_img = prev_img

    return prev_img, curr_img


def start_frame(extractFrames, sequence, numSeg, frames_idx, video_idx, stackSize, phase):
    """
    Return number of frames in video, and position of the first frame/frames to start (single number or an array)
    """
    numFrame = 0
    frameStart = 0
    startFrame = 0
    if extractFrames is True:                                                   # CASE 1: Frames are extracted
        for image_path in os.listdir(frames_idx):                               # Count number of frames
            numFrame = numFrame + 1                                             # n frames: "0.jpg" to "n-1.jpg"

    else:
        videoCapture = cv2.VideoCapture(video_idx)                              # CASE 2: Frames are captured directly
        while True:                                                             # Count number of frames
            ret, _, = videoCapture.read()
            if ret is False:
                break                                                           # n frames
            numFrame = numFrame + 1

    if sequence is True:                                                        # Choose start frame
        if numFrame <= stackSize:
            frameStart = np.ones(numSeg)
        else:
            frameStart = np.linspace(1, numFrame - stackSize, numSeg)
    else:
        if numFrame <= stackSize:
            startFrame = 1
        else:
            if phase == 'train':
                startFrame = random.randint(1, numFrame - stackSize)
            else:
                startFrame = np.ceil((numFrame - stackSize) / 2)

    return frameStart, startFrame, numFrame


class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, sequence=False, stackSize=5,
                 train=True, numSeg=5, fmt='.jpg', phase='train', seqLen = 25, extractFrames=False):

        self.video, self.labels, self.frames = gen_split(
            root_dir, extractFrames)
        self.spatial_transform = spatial_transform
        self.train = train
        self.numSeg = numSeg
        self.sequence = sequence
        self.stackSize = stackSize
        self.fmt = fmt
        self.phase = phase
        self.seqLen = seqLen
        self.extractFrames = extractFrames

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        """
        Input:
            A video in dataset
        Output:
            A sequence of optical frames generated from the video and label of this sequence
        """

        self.spatial_transform.randomize_parameters()
        numFrame=0
        label = self.labels[idx]
        inpSeqSegs = []
        inpSeq = []

        if self.sequence is True:
            frameStart, startFrame, numFrame = start_frame(self.extractFrames, self.sequence, self.numSeg, self.frames[idx],
                                                           self.video[idx], self.stackSize, self.phase)
            for startFrame in frameStart:
                for k in range(self.stackSize):                         # 0 to (stackSize-1)
                    i = k + int(startFrame)                             # min i is 1, max i is stackSize
                    if self.extractFrames:
                        prev_img, curr_img = read_img(numFrame, self.frames[idx], i)
                    else:
                        prev_img, curr_img = capture_img(numFrame, i)
                    inpSeq = optical_flow(prev_img, curr_img, self.spatial_transform, inpSeq)
                inpSeqSegs.append(torch.stack(inpSeq, 0).squeeze())
            inpSeqSegs = torch.stack(inpSeqSegs, 0)

        else:
            _, startFrame, numFrame = start_frame(self.extractFrames, self.sequence, self.numSeg, self.frames[idx],
                                                  self.video[idx], self.stackSize, self.phase)
            for k in range(self.stackSize):
                i = k + int(startFrame)
                if self.extractFrames:
                    prev_img, curr_img = read_img(numFrame, self.frames[idx], i)
                else:
                    prev_img, curr_img = capture_img(numFrame, self.video[idx], i)
                inpSeq = optical_flow(prev_img, curr_img, self.spatial_transform, inpSeq)
            inpSeqSegs = torch.stack(inpSeq, 0).squeeze(1)

        return inpSeqSegs,  label
