import os
import Train.main_run_rgb as rgb
import Train.main_run_flow as flow
import Train.main_run_twoStream_david as twoStream
import argparse

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--VIP_Cup19_Test_Data', type=str, default=r"C:\Users\enkmlam\Downloads\FPVO-segmented_videos2 - Copy\FPVO-segmented_videos2",
                        help='Train set directory')
    parser.add_argument('--VIP_Cup19_Val_Data', type=str, default=r"C:\Users\enkmlam\Downloads\FPVO-segmented_videos1 - Copy\FPVO-segmented_videos1",
                        help='Validation set directory')
    args = parser.parse_args()

    train_dir = args.VIP_Cup19_Test_Data
    val_dir=args.VIP_Cup19_Val_Data

    rgb.__main__(1,train_dir,val_dir)
    rgb.__main__(2, train_dir, val_dir)
    flow.__main__(train_dir, val_dir)
    twoStream.__main__(train_dir, val_dir)


if __name__ == '__main__':
    __main__()