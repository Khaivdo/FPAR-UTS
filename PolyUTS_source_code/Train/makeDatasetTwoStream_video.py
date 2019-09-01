import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import sys
import cv2
import Train.makeDatasetFlow_video as makeDatasetFlow


class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, sequence=False, stackSize=5,
                 train=True, numSeg=5, fmt='.jpg', phase='train', seqLen=25, extractFrames=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.video, self.labels, self.frames = makeDatasetFlow.gen_split(
            root_dir, stackSize)
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

        self.spatial_transform.randomize_parameters()
        label = self.labels[idx]
        inpSeq = []
        inpSeqSegs = []

        # for Flow model
        if self.sequence is True:
            frameStart, startFrame, numFrame = makeDatasetFlow.start_frame(self.extractFrames, self.sequence,
                                                                           self.numSeg, self.frames[idx],
                                                                           self.video[idx], self.stackSize, self.phase)
            for startFrame in frameStart:
                for k in range(self.stackSize):
                    i = k + int(startFrame)
                    if self.extractFrames:                                      # If frames are extracted
                        prev_img, curr_img = makeDatasetFlow.read_img(numFrame, self.frames[idx], i)
                    else:                                                       # If frames are captured
                        prev_img, curr_img = makeDatasetFlow.capture_img(numFrame, self.video[idx], i)

                    # Optical Flow
                    inpSeq = makeDatasetFlow.optical_flow(prev_img, curr_img, self.spatial_transform, inpSeq)
                inpSeqSegs.append(torch.stack(inpSeq, 0).squeeze())
            inpSeqSegs = torch.stack(inpSeqSegs, 0)
        else:
            _, startFrame, numFrame = makeDatasetFlow.start_frame(self.extractFrames, self.sequence, self.numSeg,
                                                                  self.frames[idx],
                                                                  self.video[idx], self.stackSize, self.phase)
            for k in range(self.stackSize):
                i = k + int(startFrame)
                if self.extractFrames:                                          # If frames are extracted
                    prev_img, curr_img = makeDatasetFlow.read_img(numFrame, self.frames[idx], i)
                else:                                                           # If frames are captured
                    prev_img, curr_img = makeDatasetFlow.capture_img(numFrame, self.video[idx], i)

                # Optical Flow
                inpSeq = makeDatasetFlow.optical_flow(prev_img, curr_img, self.spatial_transform, inpSeq)
            inpSeqSegs = torch.stack(inpSeq, 0).squeeze(1)
        os.chdir(self.frames[idx])
        inpSeqF = []

        # for RGB model
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            if self.extractFrames:
                img = cv2.imread("%d.jpg" % (int(np.floor(i)) - 1))
            else:
                videoCapture = cv2.VideoCapture(self.video[idx])
                videoCapture.set(cv2.CAP_PROP_POS_FRAMES, int(np.floor(i) - 1))
                read, img = videoCapture.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            inpSeqF.append(self.spatial_transform(img.convert('RGB')))
        inpSeqF = torch.stack(inpSeqF, 0)
        return inpSeqSegs, inpSeqF, label
