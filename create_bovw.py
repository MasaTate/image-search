import cv2
import numpy as np
import argparse
import glob
import os
from tqdm import tqdm
import csv
import pathlib
from collections import deque


def main(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    img_files = [*glob.glob(os.path.join(args.root_path, '**', '*.jpg'), recursive=True),
                *glob.glob(os.path.join(args.root_path, '**', '*.jpeg'), recursive=True),
                *glob.glob(os.path.join(args.root_path, '**', '*.png'), recursive=True)]

    codebook = np.load(args.codebook_path)
    channel, dim = codebook.shape
    print("book size:", channel, ",", dim)

    akaze = cv2.AKAZE_create()

    hist_all = deque()
    print("creating bag of visual words...")
    for img_path in tqdm(img_files):

        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        kp, des = akaze.detectAndCompute(img, None)

        if not type(des) == type(None):
            hist = deque(desToHist(des, codebook))
            hist.appendleft(pathlib.Path(img_path).resolve())
            hist_all.append(hist)
        else:
            img_files.remove(img_path)

    hist_all = list(hist_all)
    with open(os.path.join(args.result_path, "bovw_sklearn_gray"+"_{}".format(channel)+".csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(hist_all)
    
def desToHist(descriptor: np.ndarray, book):
    K, D = book.shape
    hist = [0,] * K
    assert len(descriptor.shape) == 2
    for n in range(descriptor.shape[0]):
        des = np.broadcast_to(descriptor[n],(K, D))
        tmp = np.power(des - book, 2).sum(axis=1)
        index = np.argmin(tmp)
        hist[index] += 1
    
    # normalize
    hist_norm = [float(i)/sum(hist) for i in hist]
    
    return hist_norm



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", help="input image root dir path", required=True)
    parser.add_argument("--result_path", help="output dir path", required=True)
    parser.add_argument("--codebook_path", help="codebook path", required=True)

    args = parser.parse_args()

    main(args)