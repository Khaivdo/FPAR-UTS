import torch
import Train.resnetMod as resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from Train.MyConvLSTMCell import *


class attentionModel(nn.Module):
    """Create one branch of sRGB network
    Input:
        Input variables generated from images
    Output:
        Video feature descriptor and class category
    """

    def __init__(self, num_classes=18, mem_size=512):
        super(attentionModel, self).__init__()
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)    # Use pre-trained model and no Batch Normalization layer
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = MyConvLSTMCell(512, mem_size)  # Hidden unit of convLSTM is 512
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, inputVariable):
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        for t in range(inputVariable.size(0)):
            # Pass input variables through the resNet34
            # Save features generated at the end of the resNet and the final CNN layer
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h*w)

            # Classified activity and its probability
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]

            # Generate class activation map (CAM)
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)

            # Convert CAM to a probability map
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)

            # Apply spatial attention
            attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)

            # Pass to convLSTM to generate spatio-temporal encoding of features
            state = self.lstm_cell(attentionFeat, state)
        bz0, nc0, h0, w0 = state[1].size()
        X = state[1].view(bz0, nc0, h0 * w0)

        # Video feature descriptor obtained by spatial average pooling operation
        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)

        # Class category
        feats = self.classifier(feats1)
        return feats, feats1
