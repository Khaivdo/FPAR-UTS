# FPAR Task 2 - Privacy Implementation
This directory contains the implementation of Task 2 for the IEEE VIP Cup 2019. The primary goal of the code is to detect and blur faces within a specified directory.

We considered the usability of two different implementation methods:
- S3FD (Single Shot Scale-invariant Face Detector) with pytorch. This was originally implemented [in this Github repository](https://github.com/yxlijun/S3FD.pytorch).
- Face Detection using OpenCV's Haar Cascades. 

## How to use
The entry point of the program is `protect_videos.py`, and can be executed by running:
```
 python protect_videos.py
```

The program will walk through a specified directory (defaulted to `./input` if no directory is included as an argument) and finds any `.mp4` files.

It will then process each frame of each video, detect and blur any faces and then write the output video to a specified directory (defaulted to `./output` if no directory is included as an argument).

The program will ensure that the directory tree structure of the input folder remains intact. for example, if the directory tree is:

```
input
├── video1.mp4
├── video2.mp4
├── folder1
│   ├── video11.mp4
│   └── video12.mp4
└── folder2
    ├── video21.mp4
    └── video22.mp4
```

then the resulting output directory will be as follows:

```
output
├── video1.mp4
├── video2.mp4
├── folder1
│   ├── video11.mp4
│   └── video12.mp4
└── folder2
    ├── video21.mp4
    └── video22.mp4
```

with the only difference being that the videos have now been process and blurred as required.

### Arguments

`protect_videos.py` takes in a number of arguments:

| Argument | Default | Description |
| --- | --- | --- |
| -h, --help | n/a | shows a help message displaying a description of each argument |
| -i, I | `'./input'` | Directory that contains the test video segments |
| -o, O | `'./output'` | Output directory that will contain the corresponding protected videos  |
| -mode, MODE | `'S3FD'` | This will specify which implementation to use during the program's execution. It can either be `'S3FD'` or `'cv2haar'` |

## Results from Testing

Both implementation methods were tested using the entire VIP Cup 2019 dataset, which can be found [here](http://www.eecs.qmul.ac.uk/~andrea/fpvo).

In terms of time taken, the Haar Cascades approach was the most efficient, taking around 2.5 hours (~0.2s per frame) and not requiring the use of a GPU.

S3FD required a GPU (NVIDIA Quadro P5000 was used) in order to reach a computation speed of ~0.5s per frame, and took around 5 hours to complete.

When viewing the resulting protected videos, it could be seen that S3FD had given the best outcome, successfully blurring majority of the faces. There were rare cases where it was not able to blur a face that was only half visible (for example when a person was half out of view from the camera).

The Open CV Haar classifiers (Front face, and side face) were able to detect well defined faces (weren't able to detect faces where there were objects in front of the face). Additionally, it couldn't detect abd blur when the person was at an angle from the camera (roughly 45 degrees).

Ultimately, S3FD gives the best outcome for now.