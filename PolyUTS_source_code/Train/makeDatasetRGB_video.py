import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import sys
import cv2
import shutil
import Train.makeDatasetFlow_video as makeDatasetFlow


class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, sequence=False, stackSize=5,
                 train=True, numSeg=5, fmt='.jpg', phase='train', seqLen = 25, extractFrames=False):

        self.video, self.labels, self.frames = makeDatasetFlow.gen_split(
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
            A sequence of frames extracted from the video and label of this sequence
        """
        self.spatial_transform.randomize_parameters()
        label = self.labels[idx]
        numFrame=0
        inpSeqF = []

        if self.extractFrames:                                                  # If frames are extracted
            for image_path in os.listdir(self.frames[idx]):
                numFrame=numFrame+1                                             # Count number of frames
            os.chdir(self.frames[idx])

            # Read, convert and append frames to a sequence
            for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
                img = cv2.imread("%d.jpg" % (int(np.floor(i - 1))))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                inpSeqF.append(self.spatial_transform(img.convert('RGB')))

        else:
            videoCapture = cv2.VideoCapture(self.video[idx])                    # Capture frames directly from videos
            while True:
                ret, _, = videoCapture.read()                                   # Count number of frames captured
                if ret is False:
                    break
                numFrame = numFrame + 1

            # Capture, convert and append frames to a sequence
            for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
                videoCapture.set(cv2.CAP_PROP_POS_FRAMES, int(np.floor(i)))     # Capture frame at a certain position
                read, img = videoCapture.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                inpSeqF.append(self.spatial_transform(img.convert('RGB')))

        inpSeqF = torch.stack(inpSeqF, 0)

        return inpSeqF, label
