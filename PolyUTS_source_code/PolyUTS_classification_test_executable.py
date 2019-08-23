from __future__ import print_function, division
from Test.spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip, FiveCrops)
from torch.autograd import Variable
from Test.twoStreamModel import *
from sklearn.metrics import confusion_matrix
from Test.makeDatasetTwoStream_video import *
import argparse



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
    save_dir ="./test_result"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f=open(os.path.join(save_dir,"label.txt"),'w')
    print('Number of samples = {}'.format(test_samples))
    f.write('Number of samples = {}\n'.format(test_samples))
    print('Evaluating...')

    for j, (inputFlow, inputFrame, video_name) in enumerate(test_loader):

        inputVariableFrame = Variable(inputFrame.permute(1, 0, 2, 3, 4).cuda())
        inputVariableFlow = Variable(inputFlow.cuda())
        with torch.no_grad():
            output_label = model(inputVariableFlow, inputVariableFrame)
        _, predictedTwoStream = torch.max(output_label.data, 1)

        print(str(j+1)+", video name is "+str(video_name)+", predict is "+str(label[predictedTwoStream.item()]))
        f.write(str(j+1)+", video name is "+str(video_name)+", predict is "+str(label[predictedTwoStream.item()])+"\n")

    f.close()

def __main__():



    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='whatever', help='Dataset')
    parser.add_argument('--VIP_Cup19_Test_Data', type=str, default=r"C:\Users\enkmlam\Downloads\video_segmented_total",
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

    dataset_dir=args.VIP_Cup19_Test_Data


    main_run(dataset, model_state_dict, dataset_dir, stackSize, seqLen, memSize)

if __name__ == '__main__':
    __main__()