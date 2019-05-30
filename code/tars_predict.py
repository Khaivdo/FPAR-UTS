#""" Given the pre-trained model and model_weights this will predict the output for the code
#
#python tars_predict.py -idl data/training_data -ll models/resnet18_True_freeze.pth -mo "resnet18" -nc 12 -is 224
#"""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

import pandas as pd
from collections import OrderedDict

import os
import argparse

from tars.tars_data_loaders import *
from tars.tars_training import *
from tars.tars_model import *

parser = argparse.ArgumentParser(description='Inference for a trained PyTorch Model')
parser.add_argument('-idl', '--input_data_loc', help='', default='data')
parser.add_argument('-is', '--input_shape', default=224, type=int)
parser.add_argument('-ll', '--load_loc', default="models/resnet18_True_freeze_True_freeze_initial_layer.pth" )
parser.add_argument('-mo', '--model_name', default="resnet18")
parser.add_argument('-sl', '--save_loc', default="submission/")
parser.add_argument("-g", '--use_gpu', default=True, action='store_false', help='Bool type gpu')
parser.add_argument("-gi", '--used_dataparallel', default=True, action='store_false', help='Bool type gpu')
parser.add_argument("-nc", '--num_classes', default=5, type=int)

args = parser.parse_args()

if not os.path.exists(args.save_loc):
    os.makedirs(args.save_loc)

print("use_gpu: {}, name_of_model: {}".format(args.use_gpu, args.model_name))
model_conv = all_pretrained_models(args.num_classes, use_gpu=args.use_gpu, name=args.model_name)

print("[Loading the pretrained model on this datasets]")
state_dict = torch.load(args.load_loc)

if args.used_dataparallel:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    print("[Loading Weights to the Model]")
    model_conv.load_state_dict(new_state_dict)

if not args.used_dataparallel:
    model_conv = nn.DataParallel(model_conv, device_ids=[0])
    model_conv.load_state_dict(state_dict)

model_conv = model_conv.eval()



#
#print("[Validating on Train data]")
#train_predictions, class_to_idx = model_evaluation("train", model_conv, args.input_data_loc, args.input_shape, args.use_gpu, args.model_name)
#idx_to_class = {class_to_idx[i]:i for i in class_to_idx.keys()}
##train_predictions["predicted_class_name"] = train_predictions["predicted"].apply(lambda x: idx_to_class[x])
##save_loc = args.save_loc+args.load_loc.rsplit("/")[-1].rsplit(".")[0]+"_train_"+".csv"
##train_predictions.to_csv(save_loc, index=False)
#
#print("[Validating on Validating data]")
#val_predictions, class_to_idx = model_evaluation("val", model_conv, args.input_data_loc, args.input_shape, args.use_gpu, args.model_name)
#idx_to_class = {class_to_idx[i]:i for i in class_to_idx.keys()}
##val_predictions["predicted_class_name"] = val_predictions["predicted"].apply(lambda x: idx_to_class[x])
##save_loc = args.save_loc+args.load_loc.rsplit("/")[-1].rsplit(".")[0]+"_val_"+".csv"
##val_predictions.to_csv(save_loc, index=False)

print("[Predictions on Test data]")
predictions = model_evaluation_test("test", model_conv, "data/test_all", args.input_shape, args.use_gpu, args.model_name)
print("[Predictions on Test data]")


dataloaders, image_datasets = data_loader_predict("data/train", args.input_shape, args.model_name)
class_to_idx = image_datasets.class_to_idx

path_input = "/media/ml/Scene_classification/code/pytorch_classifiers/data/test_all/normal/"
path_output = "/media/ml/Scene_classification/code/pytorch_classifiers/data/output_test_all/"
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
file = os.path.basename(predictions.iloc[10]['img_loc'])
img = cv2.imread(path_input + file) 
height , width , layers =  img.shape
#video = cv2.VideoWriter('/media/ml/Scene_classification/code/pytorch_classifiers/data/video.avi',-1,1,(1280,720))
#video = cv2.VideoWriter("/media/ml/Scene_classification/code/pytorch_classifiers/data/output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 8,(1280,720))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
video = cv2.VideoWriter("/media/ml/Scene_classification/code/pytorch_classifiers/data/output.avi", fourcc, 12.0, (width, height))
for i in range(len(predictions)):
    a=predictions.iloc[i]['predicted']
    file = os.path.basename(predictions.iloc[i]['img_loc'])
    img = cv2.imread(path_input + file) 
    
    cv2.putText(img, "Scene Type: ", (200, 50), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
    if a==0:
        text="Blocked"
        cv2.putText(img, text, (500, 50), font, 1.50, (255, 255, 0), 2, cv2.LINE_AA)
    elif a==1:
        text="Blured"
        cv2.putText(img, text, (500, 50), font, 1.50, (128, 128, 0), 2, cv2.LINE_AA)
    elif a==2:
        text="Changed View"
        cv2.putText(img, text, (500, 50), font, 1.50, (0, 255, 255), 2, cv2.LINE_AA)
    elif a==3:
        text="Normal"
        cv2.putText(img, text, (500, 50), font, 1.50, (255, 255, 255), 2, cv2.LINE_AA)
    elif a==4:
        text="Others"
        cv2.putText(img, text, (500, 50), font, 1.50, (0, 128, 128), 2, cv2.LINE_AA)
    
    
    
    
    cv2.imwrite(path_output + file,img)
    
    
    
    
    video.write(img)
    

#cv2.destroyAllWindows()
video.release()
    
#predictions["predicted_class_name"] = predictions["predicted"].apply(lambda x: idx_to_class[x])
#save_loc = args.save_loc+args.load_loc.rsplit("/")[-1].rsplit(".")[0]+"_test_"+".csv"
#predictions.to_csv(save_loc, index=False)
#
#path_input = "/media/ml/Scene_classification/datasets/test_all/"
#path_output = "/media/ml/Scene_classification/datasets/output_test_all/"
#import cv2
#use_gpu =True
#listing = os.listdir(path_input)    
#
#img=cv2.imread('xx.jpg')
#img=cv2.resize(img,(32,32))
#img = img.transpose((2,0,1))
#img=np.expand_dims(img,axis=0)
#img=img/255.0
#img=torch.FloatTensor(img)
#img=Variable(img)
#img = img.cuda()
#
#
#for file in listing:
#    img = cv2.imread(path_input + file)    
#    
#    
#    if use_gpu:
#        inputs = Variable(img.cuda())
#    else:
#        inputs = Variable(img)
#            
#    bs, ncrops, c, h, w = inputs.data.size()
#    output = model_conv(inputs.view(-1, c, h, w)) # fuse batch size and ncrops        
#                 
#    cv2.imwrite(path_output + file,im)
#    print ("   aaaa")
#    
#    
#    
#
#for img, label in dataloaders:
#        if use_gpu:
#            inputs = Variable(img.cuda())
#        else:
#            inputs = Variable(img)
#        bs, ncrops, c, h, w = inputs.data.size()
#        output = model_conv(inputs.view(-1, c, h, w)) # fuse batch size and ncrops
#        if type(output) == tuple:
#            output, _ = output
#        else:
#            output = output
