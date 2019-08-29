from __future__ import print_function, division
from Train.flow_resnet import *
from Train.spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                      RandomHorizontalFlip)
from tensorboardX import SummaryWriter
import torch.nn as nn
from torch.autograd import Variable
from Train.makeDatasetFlow_video import *
import argparse
import sys
import os
import time


def compile_dataset(trainDir1, valDir, stackSize, trainBatchSize, valBatchSize):
    """"
    Compile training and validation dataset
    """
    # Rescale the images and increase the data size using different methods
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    spatial_transform = Compose(
        [Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
         ToTensor(), normalize])

    # Prepare training dataset
    vid_seq_train = makeDataset(trainDir1, spatial_transform=spatial_transform, sequence=False,
                                stackSize=stackSize, fmt='.jpg', extractFrames=False)

    # Reshuffle data at every epoch and utilise subprocess
    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                                               shuffle=True, sampler=None, num_workers=4, pin_memory=True)
    # Prepare validation dataset
    if valDir is not None:
        vid_seq_val = makeDataset(valDir,
                                  spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                                  sequence=False, stackSize=stackSize, fmt='.jpg', phase='Test', extractFrames=False)

        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                                 shuffle=False, num_workers=2, pin_memory=True)
        valInstances = vid_seq_val.__len__()

    trainInstances = vid_seq_train.__len__()

    print('Number of samples in the dataset: training = {} | validation = {}'.format(trainInstances, valInstances))

    return train_loader, val_loader


def main_run(dataset, trainDir1, valDir, outDir, stackSize, trainBatchSize, valBatchSize, numEpochs, lr1,
             decay_factor, decay_step):
    """
        Main
    """

    if dataset == "VIPCup":
        num_classes = 18
    else:
        print("Dataset not found")
        sys.exit()

    min_accuracy = 0

    # Create a directory to save models and log files and ignore it if the dir exists
    cur_dir = os.getcwd()
    model_folder = os.path.join(cur_dir, outDir, 'flow')
    if os.path.exists(model_folder):
        pass
    else:
        os.makedirs(model_folder)

    # Load compiled dataset
    train_loader, val_loader = compile_dataset(trainDir1, valDir, stackSize, trainBatchSize, valBatchSize)

    if torch.cuda.device_count() > 1:
        cuda1 = torch.device('cuda:1')
    else:
        cuda1 = torch.device('cuda:0')

    model = flow_resnet34(True, channels=2*stackSize, num_classes=num_classes)          # Load fresh flow module
    model.train(True)                                                                   # Retrain the module
    train_params = list(model.parameters())
    model.cuda(cuda1)                                                                   # Copy CPU data to cuda1

    # Choose loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer_fn = torch.optim.SGD(train_params, lr=lr1, momentum=0.9, weight_decay=5e-4)

    # Change the learning rate after a certain number of epoch
    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=decay_step, gamma=decay_factor)

    train_iter = 0
    train_min_accuracy = 0

    for epoch in range(numEpochs):
        start = time.time()
        writer = SummaryWriter(model_folder)
        train_log_loss = open((model_folder + '/train_log_loss.txt'), 'a')
        train_log_acc = open((model_folder + '/train_log_acc.txt'), 'a')
        val_log_loss = open((model_folder + '/val_log_loss.txt'), 'a')
        val_log_acc = open((model_folder + '/val_log_acc.txt'), 'a')
        optim_scheduler.step()                                                  # Update the learning rate
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        model.train(True)
        writer.add_scalar('lr', optimizer_fn.param_groups[0]['lr'], epoch+1)    # Record lr info to .json file
        for i, (inputs, targets) in enumerate(train_loader):
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()                                # Zero out the gradients to update model parameters
            inputVariable = Variable(inputs.cuda(cuda1))
            labelVariable = Variable(targets.cuda(cuda1))
            trainSamples += inputs.size(0)
            output_label, _ = model(inputVariable)                  # Train Flow module from converted frames
            loss = loss_fn(output_label, labelVariable)
            loss.backward()                                         # Back-propagation
            optimizer_fn.step()            # Put optim_"scheduler.step()" after "optimizer_fn.step()" (pytorch > 1.1.0v)

            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += (predicted == targets.cuda(cuda1)).sum().item()
            epoch_loss += loss.item()
        avg_loss = epoch_loss/iterPerEpoch
        trainAccuracy = (numCorrTrain / trainSamples) * 100

        print('Train: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch + 1, avg_loss, trainAccuracy))
        writer.add_scalar('train/epoch_loss', avg_loss, epoch+1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch+1)
        train_log_loss.write('Training loss after {} epoch = {}\n'.format(epoch+1, avg_loss))
        train_log_acc.write('Training accuracy after {} epoch = {}\n'.format(epoch+1, trainAccuracy))
        if valDir is not None:
            if (epoch+1) % 1 == 0:
                model.train(False)                                      # Skip training model in validation dataset
                val_loss_epoch = 0
                val_iter = 0
                val_samples = 0
                numCorr = 0
                for j, (inputs, targets) in enumerate(val_loader):
                    val_iter += 1
                    val_samples += inputs.size(0)
                    with torch.no_grad():                # Set all the requires_grad flag to false to avoid retraining
                        inputVariable = Variable(inputs.cuda(cuda1))
                        labelVariable = Variable(targets.cuda(non_blocking=True).to(cuda1))
                    output_label, _ = model(inputVariable)
                    val_loss = loss_fn(output_label, labelVariable)
                    val_loss_epoch += val_loss.item()
                    _, predicted = torch.max(output_label.data, 1)
                    numCorr += (predicted == targets.cuda(cuda1)).sum().item()

                val_accuracy = (numCorr / val_samples) * 100
                avg_val_loss = val_loss_epoch / val_iter

                print('Validation: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch + 1, avg_val_loss, val_accuracy))
                writer.add_scalar('val/epoch_loss', avg_val_loss, epoch + 1)
                writer.add_scalar('val/accuracy', val_accuracy, epoch + 1)
                val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))
                val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))

                if val_accuracy > min_accuracy:                                 # Save the model with highest accuracy
                    save_path_model = (model_folder + '/model_flow_state_dict.pth')
                    torch.save(model.state_dict(), save_path_model)
                    val_log_loss.write(' (Saved) \n')
                    min_accuracy = val_accuracy
                    train_min_accuracy = trainAccuracy

            else:
                if (epoch+1) % 10 == 0:
                    save_path_model = (model_folder + '/model_flow_state_dict.pth')
                    torch.save(model.state_dict(), save_path_model)

        print("Time taken: "+str(time.time()-start))
        print("Saved : Flow module with Val_acc of %" + str(round(min_accuracy, 3))
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
    parser.add_argument('--trainDatasetDir1', type=str, default=r"C:\Users\enkmlam\Downloads\Addional_dataset",
                        help='Train set directory')
    parser.add_argument('--valDatasetDir', type=str, default=r"C:\Users\enkmlam\Downloads\video_segmented_total",
                        help='Validation set directory')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--stackSize', type=int, default=5, help='Length of sequence')
    parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--valBatchSize', type=int, default=32, help='Validation batch size')
    parser.add_argument('--numEpochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--stepSize', type=float, default=[50, 100], nargs="+", help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.5, help='Learning rate decay rate')

    args = parser.parse_args()
    dataset = args.dataset
    trainDatasetDir1 = train_dir
    valDatasetDir = val_dir
    outDir = args.outDir
    stackSize = args.stackSize
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize
    numEpochs = args.numEpochs
    lr1 = args.lr
    stepSize = args.stepSize
    decayRate = args.decayRate

    print("Flow training Start...")

    main_run(dataset, trainDatasetDir1, valDatasetDir, outDir, stackSize, trainBatchSize, valBatchSize, numEpochs, lr1,
             decayRate, stepSize)


if __name__ == '__main__':
    __main__()
