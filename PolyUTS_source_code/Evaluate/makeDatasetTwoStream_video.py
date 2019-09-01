import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import sys
import cv2
import matplotlib.pyplot as plt
import re


#New_Dataset_save=r"C:\Users\enkmlam\Downloads\private_new\train\flow"
def gen_split(root_dir, stackSize):
    Dataset = []

    Labels = []
    New_dir = ['kenya_3', 'kenya_4', 'oxford_3', 'oxford_4', 'oxford_5']
    label_name = ['chat', 'clean', 'drink', 'dryer', 'machine', 'microwave', 'mobile', 'paper', 'print', 'read',
             'shake', 'staple', 'take', 'typeset', 'walk', 'wash', 'whiteboard', 'write']
    for subject_folder in os.listdir(root_dir):
        dir = os.path.join(root_dir,subject_folder)
        for target in sorted(os.listdir(dir)):
            if subject_folder in New_dir:  # Videos from test_dataset_1
                video_name = target.split(".")[0]
                num_label = video_name.split("-")[0]
                label = re.split('(\d+)', num_label)[-1]
            elif subject_folder not in New_dir:  # Videos from segmented folders
                video_name = target.split(".")[0]
                label = video_name.split("_")[-1]
            if label=='wave'or label=='open':
                continue
            Dataset.append(os.path.join(dir, target))
            for i in range(0,len(label_name)):
                if label==label_name[i]:
                    label_id=i
                    break
                if label=="dry":
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
        label_name = ['chat', 'clean', 'drink', 'dryer', 'machine', 'microwave', 'mobile', 'paper', 'print', 'read',
                      'shake', 'staple', 'take', 'typeset', 'walk', 'wash', 'whiteboard', 'write']
        dir=self.video[idx]
        videoCapture = cv2.VideoCapture(self.video[idx])
        self.spatial_transform.randomize_parameters()
        numFrame=0
        while(True):
            ret,_,=videoCapture.read()
            if ret is False:
                break
            numFrame=numFrame+1
        numFrame = numFrame -1
        label = self.labels[idx]
        inpSeqSegs = []
        if self.sequence is True:
            if numFrame <= self.stackSize:
                frameStart = np.ones(self.numSeg)
            else:
                frameStart = np.linspace(1, numFrame - self.stackSize, self.numSeg)
            for startFrame in frameStart:
                inpSeq = []
                for k in range(self.stackSize):
                    i = k + int(startFrame)

                    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, int(round(i)))
                    _, prev_img = videoCapture.read()
                    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
                    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, int(round(i))+1)
                    _, curr_img = videoCapture.read()
                    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(prev_img, curr_img, None, 0.702, 5, 10, 2, 7, 1.5,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                    flow = np.clip(flow, -15, 15)
                    norm_flow = ((flow + 15) * 8.5).astype(int)
                    flow_x = norm_flow[..., 0]
                    flow_y = norm_flow[..., 1]
                    #flow_x = Image.fromarray(flow_x)
                    #flow_y = Image.fromarray(flow_y)
                    # If TypeError: Cannot handle this data type occurs, please change to the following code
                    flow_x = Image.fromarray(flow_x.astype(np.uint8))
                    flow_y = Image.fromarray(flow_y.astype(np.uint8))
                    inpSeq.append(self.spatial_transform(flow_x.convert('L'), inv=True, flow=True))
                    inpSeq.append(self.spatial_transform(flow_y.convert('L'), inv=False, flow=True))
                inpSeqSegs.append(torch.stack(inpSeq, 0).squeeze())
            inpSeqSegs = torch.stack(inpSeqSegs, 0)
        else:
            if numFrame <= self.stackSize:
                startFrame = 1
            else:
                if self.phase == 'train':
                    startFrame = random.randint(1, numFrame - self.stackSize)
                else:
                    startFrame = np.ceil((numFrame - self.stackSize)/2)
            inpSeq = []
            for k in range(self.stackSize):
                i = k + int(startFrame)
                videoCapture.set(cv2.CAP_PROP_POS_FRAMES,int(round(i)))
                _, prev_img = videoCapture.read()
                prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
                videoCapture.set(cv2.CAP_PROP_POS_FRAMES, int(round(i))+1)
                _, curr_img = videoCapture.read()
                curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_img, curr_img, None, 0.702, 5, 10, 2, 7, 1.5,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                flow = np.clip(flow, -15, 15)
                norm_flow = ((flow + 15) * 8.5).astype(int)
                flow_x = norm_flow[..., 0]
                flow_y = norm_flow[..., 1]
                #flow_x = Image.fromarray(flow_x)
                #flow_y = Image.fromarray(flow_y)
                # If TypeError: Cannot handle this data type occurs, please change to the following code
                flow_x = Image.fromarray(flow_x.astype(np.uint8))
                flow_y = Image.fromarray(flow_y.astype(np.uint8))
                inpSeq.append(self.spatial_transform(flow_x.convert('L'), inv=True, flow=True))
                inpSeq.append(self.spatial_transform(flow_y.convert('L'), inv=False, flow=True))
            inpSeqSegs = torch.stack(inpSeq, 0).squeeze(1)

        inpSeqF = []
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            videoCapture.set(cv2.CAP_PROP_POS_FRAMES, int(np.floor(i)))
            read, img = videoCapture.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=Image.fromarray(img)
            inpSeqF.append(self.spatial_transform(img.convert('RGB')))
        inpSeqF = torch.stack(inpSeqF, 0)

        return inpSeqSegs, inpSeqF, label
