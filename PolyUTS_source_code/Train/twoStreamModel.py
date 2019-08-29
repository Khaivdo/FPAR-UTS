import torch
from Train.flow_resnet import *
from Train.objectAttentionModelConvLSTM import *
import torch.nn as nn


class twoStreamAttentionModel(nn.Module):
    """twoStream model
    Inputs:
        Fresh or pre-trained Flow and RGB models
    Outputs:
        twoStream model using concatenated features of the two previous models
    """
    def __init__(self, flowModel='', frameModel='', stackSize=5, memSize=512, num_classes=61):
        super(twoStreamAttentionModel, self).__init__()

        # Load fresh pre-trained Flow model on ImageNet
        self.flowModel = flow_resnet34(False, channels=2*stackSize, num_classes=num_classes)
        if flowModel != '':
            self.flowModel.load_state_dict(torch.load(flowModel))           # Else load personal pre-trained Flow model

        self.frameModel = attentionModel(num_classes, memSize)              # Load fresh RGB model
        if frameModel != '':
            self.frameModel.load_state_dict(torch.load(frameModel))         # Else load personal pre-trained EGB model

        self.fc2 = nn.Linear(512 * 2, num_classes, bias=True)               # Classifier with FC and Dropout layers
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(self.dropout, self.fc2)

    def forward(self, inputVariableFlow, inputVariableFrame):
        _, flowFeats = self.flowModel(inputVariableFlow)
        _, rgbFeats = self.frameModel(inputVariableFrame)
        twoStreamFeats = torch.cat((flowFeats, rgbFeats), 1)                # Concatenate features

        return self.classifier(twoStreamFeats)
