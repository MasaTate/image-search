import cv2
import numpy as np
import argparse
import glob
import os
from tqdm import tqdm
import random


def main(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    img_files = [*glob.glob(os.path.join(args.root_path, '**', '*.jpg'), recursive=True),
                *glob.glob(os.path.join(args.root_path, '**', '*.jpeg'), recursive=True),
                *glob.glob(os.path.join(args.root_path, '**', '*.png'), recursive=True)]

    akaze = cv2.AKAZE_create()

    print("extracting features...")
    for img_path in tqdm(img_files):
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        kp, des = akaze.detectAndCompute(img, None)
        img_kp = cv2.drawKeypoints(img, kp, None, flags=4)
        cv2.imwrite(os.path.join(args.result_path+os.path.splitext(os.path.basename(img_path))[0]+ "_keypoints.png"), img_kp)
        print(os.path.join(args.result_path,os.path.splitext(os.path.basename(img_path))[0]+ "_keypoints.png"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", help="input image root dir path", required=True)
    parser.add_argument("--result_path", help="output dir path", required=True)
    args = parser.parse_args()

    main(args)