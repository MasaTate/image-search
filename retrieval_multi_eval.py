import cv2
import numpy as np
import argparse
import glob
import os
from tqdm import tqdm
import csv
import random

def main(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    codebook = np.load(args.codebook_path)
    channel, dim = codebook.shape
    print("book size:", channel, ",", dim)
    
    akaze = cv2.AKAZE_create()

    rows = []
    with open(args.bovw_path) as f:
        reader = csv.reader(f)
        for row in tqdm(reader):
            rows.append(row)
    rows_num = []
    for row in rows:
        rows_num.append([float(num) for num in row[1:]])
    bovw = np.array(rows_num)

    category = os.listdir(args.root_path)
    category = [f for f in category if os.path.isdir(os.path.join(args.root_path, f))]
    acc_dict = {}

    for c in tqdm(category):
        img_files = [*glob.glob(os.path.join(args.root_path, c, '*.jpg'), recursive=True),
                *glob.glob(os.path.join(args.root_path, c, '*.jpeg'), recursive=True),
                *glob.glob(os.path.join(args.root_path, c, '*.png'), recursive=True)]
        
        num_sample = 20
        if len(img_files) < num_sample*2:
            samples = random.sample(img_files, len(img_files))
            for i in range(num_sample*2 - len(img_files)):
                samples.append(random.choice(img_files))
        else:
            samples = random.sample(img_files, num_sample*2)

        count_per_category = 0
        denom = 0
        iteration = iter(samples)
        for s_1, s_2 in zip(iteration, iteration):

            query_img_1 = cv2.imread(s_1,cv2.IMREAD_GRAYSCALE)
            query_img_2 = cv2.imread(s_2,cv2.IMREAD_GRAYSCALE)
            _, query_des_1 = akaze.detectAndCompute(query_img_1, None)
            _, query_des_2 = akaze.detectAndCompute(query_img_2, None)
            if type(query_des_1) == type(None) or type(query_des_2) == type(None): 
                continue
            
            denom += 1
            query_des_1 = np.array(query_des_1)
            query_des_2 = np.array(query_des_2)
            
            query_hist_1 = desToHist(query_des_1, codebook)
            query_hist_2 = desToHist(query_des_2, codebook)

            #query_hist = np.add(query_hist_1, query_hist_2)
            query_hist = np.amax([query_hist_1, query_hist_2], axis=0)
            scores = histIntersect(bovw, query_hist)
            #scores = cosSim(bovw, query_hist)

            y = np.argsort(scores)[::-1]

            correct_count = 0

            eval_count = 0
            index = 0
            while(eval_count<args.num):
                if rows[y[index]][0] == s_1 or rows[y[index]][0] == s_2:
                    index += 1
                    continue

                eval_count += 1
                if rows[y[index]][0].split('/')[-2] == c:
                    correct_count += 1
                index += 1
            #print(c, ":", correct_count/args.num)
            count_per_category += correct_count

        acc_dict[c] = count_per_category /args.num /denom
        print(acc_dict)

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

    parser.add_argument("--root_path", help="root dir path", required=True)
    parser.add_argument("--result_path", help="output dir path", required=True)
    parser.add_argument("--codebook_path", help="codebook path", required=True)
    parser.add_argument("--bovw_path", help="bovw path", required=True)
    parser.add_argument("--num", help="number of images to retrieve", type=int, default=10)

    args = parser.parse_args()

    main(args)