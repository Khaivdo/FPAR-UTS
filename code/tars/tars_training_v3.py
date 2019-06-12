""" training functions
"""
import argparse
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

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class_names = ['11_chat', '12_clean', '13_drink', '14_dryer', '15_microwave', '16_print', '17_walk', '18_shake', '21_machine', '22_mobile', '23_paper', '24_read', '25_staple', '26_take', '27_typeset']
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def assigned_label(labels):
    label1= torch.tensor([0,1])
    label2= labels
    labels1= labels.clone()
    labels2= labels.clone()
#    if  labels==0:
#        lable1= torch.tensor([0])
#    else:
#        lable1= torch.tensor([1])
    if (labels== 0 or labels== 1 or labels== 2 or labels== 3 or labels== 5 or labels== 8 or labels== 10 or labels== 14 or labels== 15 or labels== 16):
        labels1=torch.tensor([0])
    else:
        labels1=torch.tensor([1])
    
    
    if (labels== 0  or labels== 4):
        labels2=torch.tensor([0])
    elif (labels== 1  or labels== 6):
        labels2=torch.tensor([1])
    elif (labels== 2  or labels== 7):
        labels2=torch.tensor([2])
    elif (labels== 3  or labels== 9):
        labels2=torch.tensor([3])
    elif (labels== 5  or labels== 11):
        labels2=torch.tensor([4])
    elif (labels== 8  or labels== 12):
        labels2=torch.tensor([5])
    elif (labels== 10  or labels== 13):
        labels2=torch.tensor([6])
    elif (labels== 14 or labels== 17):
        labels2=torch.tensor([7])
    elif (labels== 15):
        labels2=torch.tensor([8])
    elif (labels== 16):
        labels2=torch.tensor([9])
    return labels1, labels2
    
def train_model(model, dataloaders, dataset_sizes, criterion,criterion1, optimizer, scheduler, use_gpu, num_epochs=25, mixup = False, alpha = 0.1):
    print("MIXUP".format(mixup))
    since = time.time()
    parser = argparse.ArgumentParser(description='Training a pytorch model to classify different plants')
    parser.add_argument('-idl', '--input_data_loc', help='', default='data')
    parser.add_argument('-mo', '--model_name', default="resnet50")
    parser.add_argument('-f', '--freeze_layers', default=False, action='store_false', help='Bool type')
    parser.add_argument('-fi', '--freeze_initial_layers', default=True, action='store_false', help='Bool type')
    parser.add_argument('-ep', '--epochs', default=50, type=int)
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-is', '--input_shape', default=224, type=int)
    parser.add_argument('-sl', '--save_loc', default="models/" )
    parser.add_argument("-g", '--use_gpu', default=True, action='store_false', help='Bool type gpu')
    parser.add_argument("-p", '--use_parallel', default=True, action='store_false', help='Bool type to use_parallel')
    parser.add_argument("-mx", '--mixup', default=True, action='store_true' ,help='Use mixup data augementation')
    parser.add_argument("-mxal", '--mixup_alpha', default=0.1, type = float, help='Alpha to be used in mixup agumentation')
    args = parser.parse_args()
    lr = 0.01

    best_model_wts = model.state_dict()
    best_acc1 = 0.0
    best_acc2 = 0.65

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
            running_corrects2 = 0
            counter = 0
            pred_class = np.zeros(1,dataset_sizes[phase])
            true_class = np.zeros(1,dataset_sizes[phase])
            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                # get the inputs
                inputs, labels = data
#                true_class[i]=labels
                labels1, labels2 = assigned_label(labels)
                #augementation using mixup
                if phase == 'train' and mixup:
                    inputs = mixup_batch(inputs, alpha)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels1 = Variable(labels1.type(torch.FloatTensor).cuda())
                    labels2 = Variable(labels2.cuda())
                    labels = Variable(labels.cuda())
#                    labels1 = Variable(labels1.cuda())
                else:
                    inputs, labels1, labels,  = Variable(inputs), Variable(labels1), Variable(labels2), Variable(labels)

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
                pred_class[counter]= preds2.cpu().numpy()
                true_class[counter]= labels
                counter =counter+1
                
                
                loss1 = criterion1(outputs1, labels1)
                loss2 = criterion(outputs2, labels2)
#                print (loss1)
#                print (loss2)
                loss = loss1+loss2
                loss.reshape(1)
                # backward + optimize only if in training phase
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects1 += torch.sum(preds1.type(torch.FloatTensor).cuda() == labels1.data)
                running_corrects2 += torch.sum(preds2.type(torch.LongTensor).cuda() == labels2.data)
                
            # Added these 2 lines to fix "AttributeError: 'float' object has no attribute 'cpu'"
#            if isinstance(running_loss, float): return np.array(running_loss)
#            if isinstance(running_corrects, float): return np.array(running_corrects)
            
            
            running_loss_ = running_loss
            epoch_loss = running_loss_ / float(dataset_sizes[phase])
            running_corrects_1 = running_corrects1.cpu().numpy()
            epoch_acc_labels1 = running_corrects_1 / float(dataset_sizes[phase])
            running_corrects_2 = running_corrects2.cpu().numpy()
            epoch_acc_labels2 = running_corrects_2 / float(dataset_sizes[phase])

            print('{} Loss: {:.4f} Group_Acc: {:.4f}  Activity_Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc_labels1, epoch_acc_labels2))
            # deep copy the model
            if phase == 'val' and epoch_acc_labels2 > best_acc2:
                best_acc1 = epoch_acc_labels1
                best_acc2 = epoch_acc_labels2
                best_model_wts = model.state_dict()
                model_save_loc = args.save_loc+args.model_name+"_"+str(args.freeze_layers)+"_freeze"+"_"+str(args.freeze_initial_layers)+"_freeze_initial_layer"+"_"+str(lr)+"learing_rate"+".pth"
                print("[Save the best model]")
                torch.save(model.state_dict(), model_save_loc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Group_Acc: {:4f}'.format(best_acc1))
    print('Best val Activity_Acc: {:4f}'.format(best_acc2))

    
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
