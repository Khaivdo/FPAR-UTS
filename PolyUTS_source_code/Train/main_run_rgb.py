from __future__ import print_function, division
from Train.objectAttentionModelConvLSTM import *
from Train.spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                      RandomHorizontalFlip)
from tensorboardX import SummaryWriter
from Train.makeDatasetRGB_video import *
import argparse
import sys
import time


def compile_dataset(train_data_dir1, val_data_dir, seqLen, trainBatchSize, valBatchSize):
    """Compile training and validation dataset
    Inputs:
        Training and validation directories
    Outputs:
        Tensors of converted images
    """
    # Rescale the images and increase the data size using different methods
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    spatial_transform = Compose(
        [Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
         ToTensor(), normalize])

    # Prepare training dataset
    vid_seq_train = makeDataset(train_data_dir1, spatial_transform=spatial_transform, seqLen=seqLen, fmt='.jpg',
                                extractFrames=False)

    # Reshuffle data at every epoch and utilise subprocess
    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                                               shuffle=True, num_workers=0, pin_memory=True)

    # Prepare validation dataset
    if val_data_dir is not None:
        vid_seq_val = makeDataset(val_data_dir,
                                  spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                                  seqLen=seqLen, fmt='.jpg', extractFrames=False)

        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                                 shuffle=False, num_workers=0, pin_memory=True)
        valInstances = vid_seq_val.__len__()
    trainInstances = vid_seq_train.__len__()

    print('Number of samples in the dataset: training = {} | validation = {}'.format(trainInstances / 2,
                                                                                     valInstances / 2))
    return train_loader, val_loader


def rgb_training_stages(stage, num_classes, memSize, stage1_dict):
    """Initialize RGB model for training process
    Inputs:
        - RGB training stage
        - Number of activity
        - Hidden state size of convLSTM
    Outputs:
         Setup RGB model and its training parameters
    """
    train_params = []

    # Only the classifier and the convLSTM layers are trained in the first stage
    if stage == 1:
        model = attentionModel(num_classes=num_classes, mem_size=memSize)
        model.train(False)
        for params in model.parameters():
            params.requires_grad = False

    # For second stage, the CNN layers of the resNet layer 4 and the fully connected layer are also trained
    # Set "params.requires_grad = True" so we can retrain the gradient
    else:
        model = attentionModel(num_classes=num_classes, mem_size=memSize)
        model.load_state_dict(torch.load(stage1_dict))
        model.train(False)
        for params in model.parameters():
            params.requires_grad = False
        for params in model.resNet.layer4[0].conv1.parameters():
            params.requires_grad = True
            train_params += [params]
        for params in model.resNet.layer4[0].conv2.parameters():
            params.requires_grad = True
            train_params += [params]
        for params in model.resNet.layer4[1].conv1.parameters():
            params.requires_grad = True
            train_params += [params]
        for params in model.resNet.layer4[1].conv2.parameters():
            params.requires_grad = True
            train_params += [params]
        for params in model.resNet.layer4[2].conv1.parameters():
            params.requires_grad = True
            train_params += [params]
        for params in model.resNet.layer4[2].conv2.parameters():
            params.requires_grad = True
            train_params += [params]
        for params in model.resNet.fc.parameters():
            params.requires_grad = True
            train_params += [params]
        model.resNet.layer4[0].conv1.train(True)
        model.resNet.layer4[0].conv2.train(True)
        model.resNet.layer4[1].conv1.train(True)
        model.resNet.layer4[1].conv2.train(True)
        model.resNet.layer4[2].conv1.train(True)
        model.resNet.layer4[2].conv2.train(True)
        model.resNet.fc.train(True)

    # LSTM
    for params in model.lstm_cell.parameters():
        params.requires_grad = True
        train_params += [params]
    # Classifier
    for params in model.classifier.parameters():
        params.requires_grad = True
        train_params += [params]

    model.lstm_cell.train(True)
    model.classifier.train(True)
    model.cuda()  # Copy CPU data to GPU

    return model, train_params


def main_run(dataset, stage, train_data_dir1, val_data_dir, stage1_dict, out_dir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decay_factor, decay_step, memSize):
    """
        Main
    """
    if dataset == "VIPCup":
        num_classes = 18
    else:
        print("Dataset not found")
        sys.exit()

    # Create a directory to save models and log files and ignore it if the dir exists
    cur_dir = os.getcwd()
    model_folder = os.path.join(cur_dir, out_dir, 'rgb', 'stage' + str(stage))
    if os.path.exists(model_folder):
        pass
    else:
        os.makedirs(model_folder)

    train_loader, val_loader = compile_dataset(train_data_dir1, val_data_dir, seqLen, trainBatchSize, valBatchSize)

    # Initialize RGB model and training parameters
    model, train_params = rgb_training_stages(stage, num_classes, memSize, stage1_dict)

    # Choose loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer_fn = torch.optim.Adam(train_params, lr=lr1, weight_decay=4e-5, eps=1e-4)

    # Change the learning rate after a certain number of epoch
    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=decay_step, gamma=decay_factor)

    train_iter = 0
    min_accuracy = 0
    train_min_accuracy = 0

    for epoch in range(numEpochs):
        start = time.time()
        writer = SummaryWriter(model_folder)  # Save results to txt files
        train_log_loss = open((model_folder + '/train_log_loss.txt'), 'a')
        train_log_acc = open((model_folder + '/train_log_acc.txt'), 'a')
        val_log_loss = open((model_folder + '/val_log_loss.txt'), 'a')
        val_log_acc = open((model_folder + '/val_log_acc.txt'), 'a')
        optim_scheduler.step()  # Update the learning rate
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        model.lstm_cell.train(True).cuda()  # Train the LSTM and classifier layers if it is stage 1
        model.classifier.train(True).cuda()

        # Add scalar data to summarize the changes in learning rate
        writer.add_scalar('lr', optimizer_fn.param_groups[0]['lr'], epoch + 1)

        if stage == 2:  # Train the CNN layers of resNet layer 4 and the FC layer if it is stage 2
            model.resNet.layer4[0].conv1.train(True)
            model.resNet.layer4[0].conv2.train(True)
            model.resNet.layer4[1].conv1.train(True)
            model.resNet.layer4[1].conv2.train(True)
            model.resNet.layer4[2].conv1.train(True)
            model.resNet.layer4[2].conv2.train(True)
            model.resNet.fc.train(True)

        for i, (inputs, targets) in enumerate(train_loader):  # Loop through all training samples
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()  # Zero out the gradients to update model parameters

            # Convert input frames to training variables
            inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).cuda())
            labelVariable = Variable(targets.cuda())
            trainSamples += inputs.size(0)
            output_label, _ = model(inputVariable)  # Pass input variables to RGB module
            loss = loss_fn(output_label, labelVariable)
            loss.backward()  # Back-propagation
            optimizer_fn.step()  # Update the model's parameters
            _, predicted = torch.max(output_label.data, 1)

            # Update number of activities being classified correctly and training loss
            numCorrTrain += (predicted == targets.cuda()).sum().item()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / iterPerEpoch
        trainAccuracy = (numCorrTrain / trainSamples) * 100

        print('Train: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch + 1, avg_loss, trainAccuracy))

        # Summarize changes in average loss amd training accuracy
        writer.add_scalar('train/epoch_loss', avg_loss, epoch + 1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch + 1)

        # Write the recorded results to the txt files
        train_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_loss))
        train_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, trainAccuracy))

        if val_data_dir is not None:  # Validation
            if (epoch + 1) % 1 == 0:
                model.train(False)  # For validation dataset, skip training model
                val_loss_epoch = 0
                val_iter = 0
                val_samples = 0
                numCorr = 0
                for j, (inputs, targets) in enumerate(val_loader):
                    val_iter += 1
                    val_samples += inputs.size(0)
                    with torch.no_grad():  # Temporarily set all the requires_grad flag to false

                        inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).cuda())
                        # Overlap the compute of the model and the transfer of the ground-truth
                        labelVariable = Variable(targets.cuda(non_blocking=True))

                    output_label, _ = model(inputVariable)  # Implement the newly trained model
                    val_loss = loss_fn(output_label, labelVariable)
                    val_loss_epoch += val_loss.item()
                    _, predicted = torch.max(output_label.data, 1)
                    numCorr += (predicted == targets.cuda()).sum().item()
                val_accuracy = (numCorr / val_samples) * 100
                avg_val_loss = val_loss_epoch / val_iter
                print('Val: Epoch = {} | Loss {} | Accuracy = {}'.format(epoch + 1, avg_val_loss, val_accuracy))
                writer.add_scalar('val/epoch_loss', avg_val_loss, epoch + 1)
                writer.add_scalar('val/accuracy', val_accuracy, epoch + 1)
                val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))
                val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))

                # Save the model with highest accuracy
                if val_accuracy > min_accuracy:
                    save_path_model = (model_folder + '/model_rgb_state_dict.pth')
                    torch.save(model.state_dict(), save_path_model)
                    val_log_loss.write(' (Saved) \n')
                    min_accuracy = val_accuracy
                    train_min_accuracy = trainAccuracy

        # If there is no validation dataset, then save a new model after every 10 epochs
        else:
            if (epoch + 1) % 10 == 0:
                save_path_model = (model_folder + '/model_rgb_state_dict.pth')
                torch.save(model.state_dict(), save_path_model)
        print("Time taken: " + str(time.time() - start))
        print("Saved : RGB stage " + str(stage) + " with Val_acc of %" + str(round(min_accuracy, 3))
              + " and Train_acc of %" + str(round(train_min_accuracy, 3)))

        train_log_loss.close()
        train_log_acc.close()
        val_log_acc.close()
        val_log_loss.close()
        writer.export_scalars_to_json(model_folder + "/all_scalars.json")
        writer.close()


def __main__(stage_input, train_dir, val_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='VIPCup', help='Dataset')
    parser.add_argument('--stage', type=int, default=1, help='Training stage')

    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--stage1Dict', type=str, default='./experiments/rgb/stage1/model_rgb_state_dict.pth',
                        help='Stage 1 model path')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--valBatchSize', type=int, default=64, help='Validation batch size')
    parser.add_argument('--numEpochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--stepSize1', type=float, default=[25, 75, 150], nargs="+", help='Learning rate decay step')
    parser.add_argument('--stepSize2', type=float, default=[25, 75], nargs="+", help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')

    args = parser.parse_args()
    dataset = args.dataset
    stage = stage_input
    outDir = args.outDir
    stage1Dict = args.stage1Dict
    seqLen = args.seqLen
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize

    # Train stage 1 for 300 epochs with learning rate decayed by a factor of 0.1 after epochs 25, 75, 150
    if stage == 1:
        numEpochs = int(300)
        lr1 = float(1e-3)
        stepSize = args.stepSize1

    # Train stage 2 for 150 epochs with learning rate decayed by a factor of 0.1 after epochs 25, 75
    if stage == 2:
        numEpochs = int(150)
        lr1 = float(1e-4)
        stepSize = args.stepSize2

    decayRate = args.decayRate
    memSize = args.memSize
    trainDatasetDir1 = train_dir
    valDatasetDir = val_dir
    print("RGB training stage " + str(stage_input) + " Start...")

    main_run(dataset, stage, trainDatasetDir1, valDatasetDir, stage1Dict, outDir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decayRate, stepSize, memSize)


if __name__ == '__main__':
    __main__()
