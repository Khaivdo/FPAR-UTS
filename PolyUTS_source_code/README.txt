ReadMe file
===========

# Introduction

This is the implementation of STANet for human activity recognition based on egocentric videos.

# Task1-Activity recognition from raw body-camera videos

## Prerequisites

- ​Windows or Unix
- ​Python 3.7.1
- ​Anaconda 4.6.8
​- OpenCV 3.4.1
​- Pytorch 1.0.1
- ​tensorboardX 1.7
- ​prettytable 0.7.2   
- Tqdm  
- Pillow

Due to the difference between different OpenCV version, you may have to change 
'flow_x=Image.fromarry(flowx)' to 'flow_x=Image.fromarry(flowx.astype(np.uint8))'
'flow_y=Image.fromarry(flowy)' to 'flow_x=Image.fromarry(flowy.astype(np.uint8))'

## Download pre-trained model
    https://drive.google.com/drive/folders/1ZxIsNGeGrldX0DYqEjPFkRBfdCzewYx3?usp=sharing

## Performance Evaluation on Test Dataset

1.  Open "PolyUTS_source_code" folder with Command Prompt.

2.  Run "PolyUTS_classification_test_executable.py" with "--VIP_Cup19_Test_Data" and the path of test dataset which ***only contains videos***. "--VIP_Cup19_Test_Data" is the name of the first argument.

    For example:

        python PolyUTS_classification_test_executable.py --VIP_Cup19_Test_Data C:\Users\Hello\Downloads\videos

3.  The video name and the predicted label for each video segment will be shown on the interface, which will also be saved in one “.txt” file named "label.txt" under the "test_result" folder.

## Performance Evaluation on Validation Dataset

1.  Download the video segments from website:
    http://www.eecs.qmul.ac.uk/~andrea/fpvo

2.  Extract the "FPVO-segmented_videos1.zip" under the current folder. And get the "FPVO-segmented_videos1" folder, ***which contains three sub-folders***, including "subject_1_segments", "subject_2_segments", and "subject_3_segments".

3.  Open "PolyUTS_source_code" folder with Command Prompt.

4.  Run "PolyUTS_classification_val_executable.py" with "--VIP_Cup19_Val_Data" and the path of validation dataset.  "--VIP_Cup19_Val_Data" is the name of the first argument.

    For example:

        python PolyUTS_classification_val_executable.py --VIP_Cup19_Val_Data  C:\Users\Hello\Downloads\FPVO-segmented_videos1

5.  The true label and the predicted label for each video segment will show on the interface, which will also be saved in one “.txt” file named "label.txt" under the "val_result"folder.

6.  The Precision, Recall and F score for each classes will be listed in a table which will be shown on the interface and will also be saved in one “.txt” file named "Table.txt" under the "val_result" folder.

7.  The resulted Confusion_Matrix will be saved in "Confusion_Matrix.png" under the "val_result" folder.

## Train Model

1.  Download the video segments from website:
    http://www.eecs.qmul.ac.uk/~andrea/fpvo

2.  Extract the "FPVO-segmented_videos1.zip", "FPVO-segmented_videos2.zip", "FPVO-segmented_videos3.zip", "FPVO-segmented_videos4.zip" under the current folder. And get the "FPVO-segmented_videos1", "FPVO-segmented_videos2", "FPVO-segmented_videos3", "FPVO-segmented_videos4" folder which contain three sub-folder for each "FPVO-segmented_videosX". Create two new empty folders named "Val" and "Train" and copy "subject_1_segments" to "subject_3_segments" to "Val" folder, "subject_4_segments" to "subject_12_segments" to "Train" folder.

3.  Open "PolyUTS_source_code" folder with Command Prompt.

4.  Run "PolyUTS_training.py" with "--VIP_Cup19_Train_Data" plus the path of training dataset and "--VIP_Cup19_Val_Data" plus the path of validation dataset.

    For example:

        Python PolyUTS_training.py --VIP_Cup19_Train_Data  C:\Users\Hello\Downloads\Train  --VIP_Cup19_Val_Data C:\Users\Hello\Downloads\Val

5.  The training loss and the training accuracy for each video segment will show on the interface.

6.  After training, model will be save as "model_twoStream_state_dict.pth" under "Training_Model" folder.

# Task2-Privacy protection in body-camera videos

## Introduction

​This is the implementation for Task 2, privacy protection, based on P-FGSM. Each frame of a video segment will be degraded by an adversarial noise, which can mislead a general ResNet50 classifier for recognizing human faces, monitors, and mobile phones.

## Setup

    pip install -r requirements.txt

## Generate adversarial videos

1.  Create a folder with the videos that you want to create their protected version in <test-directory-folder>

2.  Create a folder that you want to store the generated protected videos in <protected-directory-folder>

3.  Generate adversarial video executing

python protect.py --org=<test-directory-folder> --des=<protected-directory-folder>

For example:

python protect.py --org=./org --des=./des

In the competition, the pre-trained ResNet50 classifier is used (stored in task2/models/model.pth), there are total 4 classes, among which '0'stands for mobile phone, '1' stands for monitor, '2' stands for negative, while '3' stands for human face.

## Output and format

1. New generated protected videos inside the location <protected-directory-folder>:
      <protected-directory-folder>/<videoname>.MP4


2.  videoname.txt file inside <protected-directory-folder> with the following order of columns (format):
      - frame number
      - number of iterations to converge
      - original class
      - original class probability
      - final class
      - final class probability
      - target class
      - target class probability

# Task3-Activity recognition from privacy-protected body-camera videos

For Task3, you may change the path of the testing video to where you stored the protected videos. The procedure is same as Task1-"Performance Evaluation on Test Dataset".
For example:
python PolyUTS_classification_test_executable.py --VIP_Cup19_Test_Data C:\Users\Hello\Downloads\protected_videos
