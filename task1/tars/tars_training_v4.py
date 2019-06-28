""" training functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import time
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd

from tars.utils import *
from tars.tars_data_loaders import *
def assigned_label(labels):
    label1= torch.tensor([0,1])
    label2_1= labels
    label2_2= labels
    labels1= labels.clone()
    labels2_1= labels.clone()
    labels2_2= labels.clone()
#    if  labels==0:
#        lable1= torch.tensor([0])
#    else:
#        lable1= torch.tensor([1])
    if (labels== 0 or labels== 1 or labels== 2 or labels== 3 or labels== 5 or labels== 8 or labels== 10 or labels== 14):
        labels1=torch.tensor([0])
    else:
        labels1=torch.tensor([1])
    
    
    if (labels== 0 ):
        labels2_1[0]=torch.tensor([0])
    elif (labels== 1 ):
        labels2_1[1]=torch.tensor([1])
    elif (labels== 2 ):
        labels2_1[2]=torch.tensor([2])
    elif (labels== 3 ):
        labels2_1[3]=torch.tensor([3])
    elif (labels== 5 ):
        labels2_1[4]=torch.tensor([4])
    elif (labels== 8 ):
        labels2_1[5]=torch.tensor([5])
    elif (labels== 10 ):
        labels2_1[6]=torch.tensor([6])
    elif (labels== 14):
        labels2_1[7]=torch.tensor([7])
    
    elif (labels== 4):
        labels2_2[0]=torch.tensor([0])
    elif (labels== 6):
        labels2_2[1]=torch.tensor([1])
    elif (labels== 7):
        labels2_2[2]=torch.tensor([2])
    elif (labels== 9):
        labels2_2[3]=torch.tensor([3])
    elif (labels== 11):
        labels2_2[4]=torch.tensor([4])
    elif (labels== 12):
        labels2_2[5]=torch.tensor([5])
    elif (labels== 13):
        labels2_2[6]=torch.tensor([6])
     
    
    return labels1, labels2_1, labels2_2
    
def train_model(model, dataloaders, dataset_sizes, criterion,criterion1, optimizer, scheduler, use_gpu, num_epochs=25, mixup = False, alpha = 0.1):
    print("MIXUP".format(mixup))
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc1 = 0.0
    best_acc2_1 = 0.0
    best_acc2_2 = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects1 = 0
            running_corrects2_1 = 0
            running_corrects2_2 = 0

            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                # get the inputs
                inputs, labels = data
                labels1, labels2_1, labels2_2= assigned_label(labels)
                #augementation using mixup
                if phase == 'train' and mixup:
                    inputs = mixup_batch(inputs, alpha)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels1 = Variable(labels1.type(torch.FloatTensor).cuda())
                    labels2_1 = Variable(labels2_1.cuda())
                    labels2_2 = Variable(labels2_2.cuda())
#                    labels1 = Variable(labels1.cuda())
                else:
                    inputs, labels1, labels2_1, labels2_2  = Variable(inputs), Variable(labels1), Variable(labels2_1), Variable(labels2_2)

                # zero the parameter gradients
#                optimizer.zero_grad()

                # forward
                outputs1,outputs2 = model(inputs,labels1)
#                print (outputs2)
##                print (labels1)
#                print ("&&&&&&&&&&&&&&&&&&&&&7777")
                
                
                
                if type(outputs1) == tuple:
                    outputs1, _ = outputs1
                if type(outputs2) == tuple:
                    outputs2, _ = outputs2
                if outputs1.data<.5:
                    preds1 = torch.tensor([0])
                else:
                    preds1 = torch.tensor([1])
#                _, preds1 = torch.max(outputs1.data, 1)
                _, preds2 = torch.max(outputs2.data, 1)
#                t = Variable(torch.FloatTensor([0.5]))
#                ind = (preds1 > t.cuda()).float()*1
                
                
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    
                    loss1 = criterion1(outputs1, labels1)
                    if labels1 == torch.tensor([0]):
                        loss2_1 = criterion(outputs2, labels2_1)
                    elif labels1 == torch.tensor([1])
                        loss2_2 = criterion(outputs2, labels2_2)
                    loss = loss1 +loss2_1 +loss2_2
                    loss.reshape(1)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                if phase == 'val':
                    
                    if preds1 == torch.tensor([0]):
                        loss2_1 = criterion(outputs2, labels2_1)
                    elif preds1 == torch.tensor([1])
                        loss2_2 = criterion(outputs2, labels2_2)
                    
                    loss1 = criterion1(outputs1, labels1)
                    
                    loss = loss1 +loss2_1 +loss2_2
                    loss.reshape(1)
                   
                # statistics
                running_loss += loss.item()
                running_corrects1 += torch.sum(preds1.type(torch.FloatTensor).cuda() == labels1.data)
                running_corrects2_1 += torch.sum(preds2 == labels2_1.data)
                running_corrects2_2 += torch.sum(preds2 == labels2_2.data)
                
            # Added these 2 lines to fix "AttributeError: 'float' object has no attribute 'cpu'"
#            if isinstance(running_loss, float): return np.array(running_loss)
#            if isinstance(running_corrects, float): return np.array(running_corrects)
            
            
            running_loss_ = running_loss
            epoch_loss = running_loss_ / float(dataset_sizes[phase])
            running_corrects_1 = running_corrects1.cpu().numpy()
            epoch_acc_labels1 = running_corrects_1 / float(dataset_sizes[phase])
            running_corrects2_1 = running_corrects2_1.cpu().numpy()
            epoch_acc_labels2_1 = running_corrects2_1 / float(dataset_sizes[phase])
            running_corrects2_2 = running_corrects2_2.cpu().numpy()
            epoch_acc_labels2_2 = running_corrects2_2 / float(dataset_sizes[phase])

            print('{} Loss: {:.4f} Group_Acc: {:.4f}  Activity_Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc_labels1, epoch_acc_labels2_1, epoch_acc_labels2_2))

            # deep copy the model
            if phase == 'val' and epoch_acc_labels1 > best_acc1 and epoch_acc_labels2_1 > best_acc2_1 and epoch_acc_labels2_2 > best_acc2_2:
                best_acc1 = epoch_acc_labels1
                best_acc2_1 = epoch_acc_labels2_1
                best_acc2_2 = epoch_acc_labels2_2
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Group_Acc: {:4f}'.format(best_acc1))
    print('Best val Label2_1_Activity_Acc: {:4f}'.format(best_acc2_1))
    print('Best val Label2_2_Activity_Acc: {:4f}'.format(best_acc2_2))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def model_evaluation(mode, model_conv, input_data_loc, input_shape, use_gpu, name):
    """A function which evaluates the model and outputs the csv of predictions of validation and train.

    mode:
    model_conv:
    input_data_loc:
    input_shape:
    use_gpu:

    Output:
    1) Prints score of train and validation (Write it to a log file)
    """
    print("[Evaluating the data in {}]".format(mode))
    data_loc = os.path.join(input_data_loc, mode)

    print("[Building dataloaders]")
    dataloaders, image_datasets = data_loader_predict(data_loc, input_shape, name)
    class_to_idx = image_datasets.class_to_idx
    imgs = [i[0] for i in image_datasets.imgs]
    print("total number of {} images: {}".format(mode, len(imgs)))
    original, predicted, probs = [], [], []
    for img, label in dataloaders:
        if use_gpu:
            inputs = Variable(img.cuda())
        else:
            inputs = Variable(img)
        bs, ncrops, c, h, w = inputs.data.size()
        output = model_conv(inputs.view(-1, c, h, w),2) # fuse batch size and ncrops
        if type(output) == tuple:
            output, _ = output
        else:
            output = output
        outputs = torch.stack([nn.Softmax(dim=0)(i) for i in output])
        outputs = outputs.mean(0)
        _, preds = torch.max(outputs, 0)
        probs.append(outputs.data.cpu().numpy())
        original.extend(label.numpy())
#        print (preds.data.cpu().numpy())
        predicted.extend([preds.data.cpu().numpy()])
    print("Accuracy_score {} : {} ".format(mode,  accuracy_score(original, predicted)))
    frame = pd.DataFrame(probs)
    frame.columns = ["class_{}".format(i) for i in frame.columns]
    frame["img_loc"] = imgs
    frame["original"] = original
    frame["predicted"] = predicted
    return frame, class_to_idx


def model_evaluation_test(mode, model_conv, test_input_data_loc, input_shape, use_gpu, name):
    dataloaders, image_datasets = data_loader_predict(test_input_data_loc, input_shape, name)
    imgs =[i[0] for i in image_datasets.imgs]
    print("total number of {} images: {}".format(mode, len(imgs)))
    predicted, probs = [], []
    for img, label in dataloaders:
        if use_gpu:
            inputs = Variable(img.cuda())
        else:
            inputs = Variable(img)
        bs, ncrops, c, h, w = inputs.data.size()
        output = model_conv(inputs.view(-1, c, h, w)) # fuse batch size and ncrops
        
        if type(output) == tuple:
            output, _ = output
        else:
            output = output
        outputs = torch.stack([nn.Softmax(dim=0)(i) for i in output])
        outputs = outputs.mean(0)
        _, preds = torch.max(outputs, 0)
        probs.append(outputs.data.cpu().numpy())
#        print ([preds.data.cpu().numpy()])
        predicted.extend([preds.data.cpu().numpy()])
    frame = pd.DataFrame(probs)
    frame.columns = ["class_{}".format(i) for i in frame.columns]
    frame["img_loc"] = imgs
    frame["predicted"] = predicted
    return frame
