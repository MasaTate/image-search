import cv2
import numpy as np
from utils.utils import randomIntList, getDate, strToDate
import argparse
import glob
import os
from tqdm import tqdm
import csv

def main(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    codebook = np.load(args.codebook_path)
    channel, dim = codebook.shape
    print("book size:", channel, ",", dim)
    
    akaze = cv2.AKAZE_create()
    #query_img = cv2.imread(args.query_path)
    query_img = cv2.imread(args.query_path,cv2.IMREAD_GRAYSCALE)
    query_kp, query_des = akaze.detectAndCompute(query_img, None)
    query_des  = np.array(query_des)
    query_hist = desToHist(query_des, codebook)

    rows = []
    with open(args.bovw_path) as f:
        reader = csv.reader(f)
        for row in tqdm(reader):
            rows.append(row)

    rows_num = []
    for row in rows:
        rows_num.append([float(num) for num in row[1:]])
    bovw = np.array(rows_num)
    
    scores = histIntersect(bovw, query_hist)
    #scores = cosSim(bovw, query_hist)

    y = np.argsort(scores)[::-1]

    out_index = []
    for i in range(len(y)):
        start_flag = True
        end_flag = True
        if args.start_date is not None:
            start_flag = False
            im_date = getDate(rows[y[i][0]])
            if im_date is not None and strToDate(im_date) >= strToDate(args.start_date):
                start_flag = True
        if args.end_date is not None:
            end_flag = False
            im_date = getDate(rows[y[i][0]])
            if im_date is not None and strToDate(im_date) <= strToDate(args.end_date):
                end_flag = True

        if start_flag and end_flag:
            out_index.append(y[i])
        
        if not len(out_index) < args.num:
            break

    for i in range(len(out_index)):
        print(scores[out_index[i]], rows[out_index[i]][0])
    return

    
def desToHist(descriptor: np.ndarray, book):
    K, D = book.shape
    hist = [0,] * K
    for n in range(descriptor.shape[0]):
        des = np.broadcast_to(descriptor[n],(K, D))
        tmp = np.power(des - book, 2).sum(axis=1)
        index = np.argmin(tmp)
        hist[index] += 1
    
    # normalize
    hist_norm = np.array([float(i)/sum(hist) for i in hist])

    return hist_norm

def histIntersect(bovw, query_hist):
    N, D = bovw.shape
    scores = np.zeros((N))
    for i in range(N):
        eval = np.vstack((bovw[i], query_hist))
        eval = np.amin(eval, axis=0)
        scores[i] = eval.sum()
    
    return scores

def cosSim(bovw, query_hist):
    N, D = bovw.shape
    #bovw_norm = bovw * bovw
    #bovw_norm = bovw.sum(axis=1)
    #print(bovw_norm)
    #scores = np.divide(np.dot(bovw, query_hist.T) / np.linalg.norm(query_hist, ord=2), np.broadcast_to(bovw_norm.reshape(1, N), (D, N)))
    scores = np.dot(bovw, query_hist.T)
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_path", help="output dir path", required=True)
    parser.add_argument("--codebook_path", help="codebook path", required=True)
    parser.add_argument("--bovw_path", help="bovw path", required=True)
    parser.add_argument("--query_path", help="query image path", required=True)
    parser.add_argument("--num", help="number of images to retrieve", type=int, default=10)
    parser.add_argument("--start_date", help="Input start date like yyyy:mm:dd", default=None)
    parser.add_argument("--end_date", help="Input end date like yyyy:mm:dd", default=None)

    args = parser.parse_args()

    main(args)