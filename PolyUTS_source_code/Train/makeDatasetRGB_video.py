import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import sys
import cv2


def gen_split(root_dir, stackSize):
    Dataset = []
    Labels = []
    label_name = ['chat', 'clean', 'drink', 'dryer', 'machine', 'microwave', 'mobile', 'paper', 'print', 'read',
                  'shake', 'staple', 'take', 'typeset', 'walk', 'wash', 'whiteboard', 'write']
    for subject_folder in os.listdir(root_dir):
        dir = os.path.join(root_dir, subject_folder)
        for target in sorted(os.listdir(dir)):
            video_name = target.split(".")[0]
            label = video_name.split("_")[-1]
            if label == 'wave' or label == 'open':
                continue
            Dataset.append(os.path.join(dir, target))
            for i in range(0, len(label_name)):
                if label == label_name[i]:
                    label_id = i
                    break
                if label == "dry":
                    label_id = 3
                    break
            Labels.append(label_id)
    return Dataset, Labels

class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, sequence=False, stackSize=5,
                 train=True, numSeg=5, fmt='.jpg', phase='train', seqLen = 25):


        self.video,self.labels  = gen_split(
            root_dir, stackSize)
        self.spatial_transform = spatial_transform
        self.train = train
        self.numSeg = numSeg
        self.sequence = sequence
        self.stackSize = stackSize
        self.fmt = fmt
        self.phase = phase
        self.seqLen = seqLen

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        dir=self.video[idx]
        videoCapture = cv2.VideoCapture(self.video[idx])
        self.spatial_transform.randomize_parameters()
        numFrame=0
        while(True):
            ret,_,=videoCapture.read()
            if ret is False:
                break
            numFrame=numFrame+1

        label = self.labels[idx]

        inpSeqF = []
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            videoCapture.set(cv2.CAP_PROP_POS_FRAMES, int(np.floor(i)))
            read, img = videoCapture.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=Image.fromarray(img)
            inpSeqF.append(self.spatial_transform(img.convert('RGB')))
        inpSeqF = torch.stack(inpSeqF, 0)
        return inpSeqF, label
