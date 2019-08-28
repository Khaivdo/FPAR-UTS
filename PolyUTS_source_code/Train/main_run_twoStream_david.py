from __future__ import print_function, division
from Train.spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
from tensorboardX import SummaryWriter
import torch.nn as nn
from Train.twoStreamModel import *
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
from Train.makeDatasetTwoStream_video import *
import argparse
import time
import sys


def compile_dataset(trainDatasetDir, valDatasetDir, seqLen, stackSize, valBatchSize, trainBatchSize):
    """Compile training and validation dataset
    Inputs:
        Training and validation directories
    Outputs:
        Tensors of converted images
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Rescale the images and increase the data size using different methods
    normalize = Normalize(mean=mean, std=std)
    spatial_transform = Compose(
        [Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
         ToTensor(), normalize])

    # Prepare training dataset
    vid_seq_train = makeDataset(trainDatasetDir, spatial_transform=spatial_transform, sequence=False, numSeg=1,
                                stackSize=stackSize, fmt='.jpg', seqLen=seqLen, extractFrames=False)

    # Reshuffle data at every epoch and utilise subprocess
    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize, shuffle=True, num_workers=0,
                                               pin_memory=True)

    # Prepare validation dataset
    if valDatasetDir is not None:
        vid_seq_val = makeDataset(valDatasetDir,
                                  spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                                  sequence=False, numSeg=1, stackSize=stackSize, fmt='.jpg', phase='Test',
                                  seqLen=seqLen, extractFrames=False)

        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize, shuffle=False, num_workers=0,
                                                 pin_memory=True)
        valSamples = vid_seq_val.__len__()
    trainSamples = vid_seq_train.__len__()

    print('Number of samples in the dataset: training = {} | validation = {}'.format(trainSamples / 2,
                                                                                     valSamples / 2))
    return train_loader, val_loader, trainSamples, valSamples


def twoStream_model(flowModel, rgbModel, stackSize, memSize, num_classes):
    """
    Initialize twoStream model for training process
    """
    model = twoStreamAttentionModel(flowModel=flowModel, frameModel=rgbModel, stackSize=stackSize, memSize=memSize,
                                    num_classes=num_classes)
    for params in model.parameters():
        params.requires_grad = False

    model.train(False)
    train_params = []

    for params in model.classifier.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.frameModel.lstm_cell.parameters():
        train_params += [params]
        params.requires_grad = True

    for params in model.frameModel.resNet.layer4[0].conv1.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.frameModel.resNet.layer4[0].conv2.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.frameModel.resNet.layer4[1].conv1.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.frameModel.resNet.layer4[1].conv2.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.frameModel.resNet.layer4[2].conv1.parameters():
        params.requires_grad = True
        train_params += [params]
    #
    for params in model.frameModel.resNet.layer4[2].conv2.parameters():
        params.requires_grad = True
        train_params += [params]
    #
    for params in model.frameModel.resNet.fc.parameters():
        params.requires_grad = True
        train_params += [params]

    base_params = []
    for params in model.flowModel.layer4.parameters():
        base_params += [params]
        params.requires_grad = True

    model.cuda()

    return model, train_params, base_params


def main_run(dataset, flowModel, rgbModel, stackSize, seqLen, memSize, trainDatasetDir, valDatasetDir, outDir,
             trainBatchSize, valBatchSize, lr1, numEpochs, decay_step, decay_factor):
    """
        Main
    """
    if dataset == 'VIPCup':
        num_classes = 18
    else:
        print('Dataset not found')
        sys.exit()

    # Create a directory to save models and log files and ignore it if the dir exists
    cur_dir = os.getcwd()
    model_folder = os.path.join(cur_dir, outDir,  'twoStream')  # Dir for saving models and log files
    model_save_path = os.path.join(cur_dir, 'Training_Model')

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if os.path.exists(model_folder):
        pass
    else:
        os.makedirs(model_folder)

    train_loader, val_loader, trainSamples, valSamples = compile_dataset(trainDatasetDir, valDatasetDir, seqLen,
                                                                         stackSize, valBatchSize, trainBatchSize)
    model, train_params, base_params = twoStream_model(flowModel, rgbModel, stackSize, memSize, num_classes)

    min_accuracy = 0
    train_min_accuracy = 0

    # Choose loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer_fn = torch.optim.SGD([
        {'params': train_params},
        {'params': base_params, 'lr': 1e-4},
    ], lr=lr1, momentum=0.9, weight_decay=5e-4)

    optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_fn, step_size=decay_step, gamma=decay_factor)
    train_iter = 0

    for epoch in range(numEpochs):
        start = time.time()
        writer = SummaryWriter(model_folder)                                    # Save results to txt files
        train_log_loss = open((model_folder + '/train_log_loss.txt'), 'w')
        train_log_acc = open((model_folder + '/train_log_acc.txt'), 'w')
        val_log_loss = open((model_folder + '/val_log_loss.txt'), 'w')
        val_log_acc = open((model_folder + '/val_log_acc.txt'), 'w')
        optim_scheduler.step()                                                # Update the learning rate
        epoch_loss = 0
        numCorrTrain = 0
        iterPerEpoch = 0
        model.classifier.train(True)                                          # Train the Flow4 and classifier layers
        model.flowModel.layer4.train(True)
        for j, (inputFlow, inputFrame, targets) in enumerate(train_loader):
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()                                # Zero out the gradients to update model parameters
            inputVariableFlow = Variable(inputFlow.cuda())
            inputVariableFrame = Variable(inputFrame.permute(1, 0, 2, 3, 4).cuda())
            labelVariable = Variable(targets.cuda())
            output_label = model(inputVariableFlow, inputVariableFrame)       # Pass input variables to twoStream module
            loss = loss_fn(F.log_softmax(output_label, dim=1), labelVariable)
            loss.backward()                                                   # Back-propagation
            optimizer_fn.step()                                               # Update the model's parameters
            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += (predicted == targets.cuda()).sum().item()
            epoch_loss += loss.data.item()
        avg_loss = epoch_loss / iterPerEpoch
        trainAccuracy = (numCorrTrain / trainSamples) * 100
        print('Average training loss after {} epoch = {} '.format(epoch + 1, avg_loss))
        print('Training accuracy after {} epoch = {}% '.format(epoch + 1, trainAccuracy))
        writer.add_scalar('train/epoch_loss', avg_loss, epoch + 1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch + 1)
        train_log_loss.write('Training loss after {} epoch = {}\n'.format(epoch + 1, avg_loss))
        train_log_acc.write('Training accuracy after {} epoch = {}\n'.format(epoch + 1, trainAccuracy))
        if valDatasetDir is not None:                                          # Validation
            if (epoch + 1) % 1 == 0:
                model.train(False)                                             # Skip training model
                val_loss_epoch = 0
                val_iter = 0
                numCorr = 0
                for j, (inputFlow, inputFrame, targets) in enumerate(val_loader):
                    val_iter += 1
                    inputVariableFlow = Variable(inputFlow.cuda())
                    inputVariableFrame = Variable(inputFrame.permute(1, 0, 2, 3, 4).cuda())
                    labelVariable = Variable(targets.cuda())
                    output_label = model(inputVariableFlow, inputVariableFrame)
                    loss = loss_fn(F.log_softmax(output_label, dim=1), labelVariable)
                    val_loss_epoch += loss.data.item()
                    _, predicted = torch.max(output_label.data, 1)
                    numCorr += (predicted == labelVariable.data).sum().item()
                val_accuracy = (numCorr / valSamples) * 100
                avg_val_loss = val_loss_epoch / val_iter
                print('Val Loss after {} epochs, loss = {}'.format(epoch + 1, avg_val_loss))
                print('Val Accuracy after {} epochs = {}%'.format(epoch + 1, val_accuracy))
                writer.add_scalar('val/epoch_loss', avg_val_loss, epoch + 1)
                writer.add_scalar('val/accuracy', val_accuracy, epoch + 1)
                val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))
                val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))
                if val_accuracy > min_accuracy:
                    save_path_model = (model_folder + '/model_twoStream_state_dict.pth')
                    torch.save(model.state_dict(), save_path_model)
                    torch.save(model.state_dict(), (model_save_path + '/model_twoStream_state_dict.pth'))
                    val_log_loss.write(' (Saved) \n')
                    min_accuracy = val_accuracy
                if val_accuracy == min_accuracy and trainAccuracy > train_min_accuracy:
                    save_path_model = (model_folder + '/model_twoStream_state_dict.pth')
                    torch.save(model.state_dict(), save_path_model)
                    torch.save(model.state_dict(), (model_save_path + '/model_twoStream_state_dict.pth'))
                    val_log_loss.write(' (Saved) \n')
                    min_accuracy = val_accuracy
                    train_min_accuracy = trainAccuracy
        else:
            if (epoch + 1) % 10 == 0:
                save_path_model = (model_folder + '/model_twoStream_state_dict.pth')
                torch.save(model.state_dict(), (model_save_path+ '/model_twoStream_state_dict.pth'))
                torch.save(model.state_dict(), save_path_model)
        print("Time taken:" +str(time.time() - start))
        print("Saved : Two Stream module with Val_acc of %" + str(round(min_accuracy, 3))
              + " and Train_acc of %" + str(round(train_min_accuracy, 3)))
        train_log_loss.close()
        train_log_acc.close()
        val_log_acc.close()
        val_log_loss.close()
        writer.export_scalars_to_json(model_folder + "/all_scalars.json")
        writer.close()


def __main__(train_dir,val_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='VIPCup', help='Dataset')
    parser.add_argument('--flowModel', type=str, default='./experiments/flow/model_flow_state_dict.pth',
                        help='Flow model')
    parser.add_argument('--rgbModel', type=str, default='./experiments/rgb/stage2/model_rgb_state_dict.pth',
                        help='RGB model')
    parser.add_argument('--trainDatasetDir', type=str, default='./train',
                        help='Train set directory')
    parser.add_argument('--valDatasetDir', type=str, default='./cv',
                        help='Validation set directory')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--stackSize', type=int, default=5, help='Number of opticl flow images in input')
    parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--valBatchSize', type=int, default=32, help='Validation batch size')
    parser.add_argument('--numEpochs', type=int, default=250, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--stepSize', type=float, default=1, help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.99, help='Learning rate decay rate')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')

    args = parser.parse_args()

    dataset = args.dataset
    flowModel = args.flowModel
    rgbModel = args.rgbModel
    trainDatasetDir = train_dir
    valDatasetDir = val_dir
    outDir = args.outDir
    stackSize = args.stackSize
    seqLen = args.seqLen
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize
    numEpochs = args.numEpochs
    lr1 = args.lr
    decay_step = args.stepSize
    decay_factor = args.decayRate
    memSize = args.memSize

    main_run(dataset, flowModel, rgbModel, stackSize, seqLen, memSize, trainDatasetDir, valDatasetDir, outDir,
             trainBatchSize, valBatchSize, lr1, numEpochs, decay_step, decay_factor)


if __name__ == '__main__':
    __main__()
