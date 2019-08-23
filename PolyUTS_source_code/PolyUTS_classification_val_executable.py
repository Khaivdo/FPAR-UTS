from __future__ import print_function, division
from Evaluate.spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip, FiveCrops)
from torch.autograd import Variable
from Evaluate.twoStreamModel import *
from sklearn.metrics import confusion_matrix
from Evaluate.makeDatasetTwoStream_video import *
import argparse
import sys
from prettytable import PrettyTable
import csv
from Evaluate.Confusion_matrix import *


def main_run(dataset, model_state_dict, dataset_dir, stackSize, seqLen, memSize):
    label = ['chat', 'clean', 'drink', 'dryer', 'machine', 'microwave', 'mobile', 'paper', 'print', 'read',
             'shake', 'staple', 'take', 'typeset', 'walk', 'wash', 'whiteboard', 'write']

    if  dataset=="whatever":
        num_classes=18
    elif dataset == 'gtea61':
        num_classes = 61
    elif dataset == 'gtea71':
      num_classes = 71
    elif dataset == 'gtea_gaze':
        num_classes = 44
    elif dataset == 'egtea':
        num_classes = 106

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    normalize = Normalize(mean=mean, std=std)

    testBatchSize = 1
    spatial_transform = Compose([Scale(256), CenterCrop(224), ToTensor(), normalize])

    vid_seq_test = makeDataset(dataset_dir,spatial_transform=spatial_transform, sequence=False, numSeg=1,
                               stackSize=stackSize, fmt='.jpg', phase='Test', seqLen=seqLen)

    test_loader = torch.utils.data.DataLoader(vid_seq_test, batch_size=testBatchSize,
                            shuffle=False, num_workers=0, pin_memory=True)

    model = twoStreamAttentionModel(stackSize=5, memSize=512, num_classes=num_classes)
    model.load_state_dict(torch.load(model_state_dict))


    for params in model.parameters():
        params.requires_grad = False

    model.train(False)
    model.cuda()

    test_samples = vid_seq_test.__len__()
    save_dir ="./val_result"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f=open(os.path.join(save_dir,"label.txt"),'w')
    print('Number of samples = {}'.format(test_samples))
    f.write('Number of samples = {}\n'.format(test_samples))
    print('Evaluating...')
    numCorrTwoStream = 0
    predicted_class = dict()
    true_class = dict()
    true_positive = dict()
    predicted_labels = []
    true_labels = []

    for j, (inputFlow, inputFrame, targets) in enumerate(test_loader):

        inputVariableFrame = Variable(inputFrame.permute(1, 0, 2, 3, 4).cuda())
        inputVariableFlow = Variable(inputFlow.cuda())
        with torch.no_grad():
            output_label = model(inputVariableFlow, inputVariableFrame)
        _, predictedTwoStream = torch.max(output_label.data, 1)
        if predictedTwoStream.item()==targets.item():
            numCorrTwoStream=numCorrTwoStream+1
        print(str(j+1)+", true label is "+str(label[targets.item()])+", predict is "+str(label[predictedTwoStream.item()]))
        f.write(str(j+1)+", true label is "+str(label[targets.item()])+", predict is "+str(label[predictedTwoStream.item()])+"\n")

        if predictedTwoStream.item() in predicted_class.keys():
            predicted_class[predictedTwoStream.item()]=int(predicted_class[predictedTwoStream.item()])+1
        else:
            predicted_class[predictedTwoStream.item()]=1
        if targets.item() in true_class.keys():
            true_class[targets.item()]=int(true_class[targets.item()])+1
        else:
            true_class[targets.item()]=1
        if predictedTwoStream.item() == targets.item():
            if predictedTwoStream.item() in true_positive.keys():
                true_positive[predictedTwoStream.item()]=true_positive[predictedTwoStream.item()]+1
            else:
                true_positive[predictedTwoStream.item()]=1

        predicted_labels.append(label[predictedTwoStream.item()])
        true_labels.append(label[targets.item()])
    f.close()
    test_accuracyTwoStream = (numCorrTwoStream / test_samples) * 100


    print("****************************************************************************************************")
    print('Test Accuracy = {}%'.format(test_accuracyTwoStream))
    number_class=0
    F_Score=dict()
    P_Score=dict()
    R_Score=dict()
    for class_id in true_class.keys():

        if class_id in predicted_class.keys():
            pass
        else:
            predicted_class[class_id]=0

        if class_id in true_positive.keys():
            pass
        else:
            true_positive[class_id]=0
        if predicted_class[class_id]==0:
            P=0
        else:
            P = float(true_positive[class_id] / predicted_class[class_id])
        R = float(true_positive[class_id] / true_class[class_id])
        if (P+R)==0:
            F=0
        else:
            F = 2 * (P * R) / (P + R)
        F_Score[class_id]=F
        P_Score[class_id]=P
        R_Score[class_id]=R
        number_class = number_class + 1



    with open(os.path.join(save_dir,"true_label.csv"), 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(true_labels)

    with open(os.path.join(save_dir,"predicted_label.csv"), 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(predicted_labels)

    with open(os.path.join(save_dir,'F_Score.csv'), 'w') as f:
        for key in F_Score.keys():
            f.write("%s,%s\n" % (key, F_Score[key]))

    with open(os.path.join(save_dir,'P_Score.csv'), 'w') as f:
        for key in P_Score.keys():
            f.write("%s,%s\n" % (key, P_Score[key]))

    with open(os.path.join(save_dir,'R_Score.csv'), 'w') as f:
        for key in R_Score.keys():
            f.write("%s,%s\n" % (key, R_Score[key]))

    table = PrettyTable(['Activity ID','Activity Name', 'Precision', 'Recall', 'F Score'])
    for i in range(0,18):
        if not i in P_Score.keys():
            P_Score[i]=0
            R_Score[i]=0
            F_Score[i]=0
        table.add_row([i, label[i], str(round(P_Score[i]*100,2))+'%', str(round(R_Score[i]*100,2))+'%',str(round(F_Score[i]*100,2))+"%"])
    table.add_row([' ','Total', 'Average Precision', 'Average Recall', 'Average F Score'])
    table.add_row([" ", " ", str(round(sum(P_Score.values())/number_class*100,2))+'%', str(round(sum(R_Score.values())/number_class*100,2))+"%",str(round(sum(F_Score.values())/number_class*100,2))+"%"])
    print(table)
    with open(os.path.join(save_dir,'Table.txt'), 'w') as w:
        w.write('Test Accuracy = {}%\n'.format(test_accuracyTwoStream))
        w.write(str(table))
    w.close()

    matrix = confusion_matrix(true_labels, predicted_labels,label)
    np.set_printoptions(precision=4)

# Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(matrix, classes=label , normalize=True,
                      title='Confusion Matrix ')
    plt.savefig(os.path.join(save_dir,'Confusion_Matrix.png'))
    plt.close()

def __main__():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='whatever', help='Dataset')
    parser.add_argument('--VIP_Cup19_Val_Data', type=str, default=r"C:\Users\enkmlam\Downloads\train_total",
                        help='Dataset directory')
    parser.add_argument('--modelStateDict', type=str, default=r"./Pretrained_Model/model_twoStream_state_dict.pth",
                        help='Model path')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--stackSize', type=int, default=5, help='Number of optical flow images in input')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')

    args = parser.parse_args()

    dataset = args.dataset
    model_state_dict=args.modelStateDict
    seqLen = args.seqLen
    stackSize = args.stackSize
    memSize = args.memSize

    dataset_dir=args.VIP_Cup19_Val_Data


    main_run(dataset, model_state_dict, dataset_dir, stackSize, seqLen, memSize)

if __name__ == '__main__':
    __main__()