import cv2
import numpy as np
from utils.utils import randomIntList
import argparse
import glob
import os
from tqdm import tqdm
import random
from sklearn import cluster


def main(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    img_files = [*glob.glob(os.path.join(args.root_path, '**', '*.jpg'), recursive=True),
                *glob.glob(os.path.join(args.root_path, '**', '*.jpeg'), recursive=True),
                *glob.glob(os.path.join(args.root_path, '**', '*.png'), recursive=True)]

    akaze = cv2.AKAZE_create()

    des_all = []
    print("extracting features...")
    for img_path in tqdm(img_files):
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        kp, des = akaze.detectAndCompute(img, None)

        if not type(des) == type(None):
            des_all.extend(des)
        else:
            img_files.remove(img_path)

    des_all = np.array(des_all)

    n_cluster = 1024
    print("creating codebook... (dim={})".format(n_cluster))


    #codebook = desToBook(des_all, n_cluster)
    #codebook = fastDesToBook(des_all, n_cluster)
    #codebook = fastDesToBookPlus(des_all, n_cluster)

    km = cluster.MiniBatchKMeans(n_cluster, init="k-means++", batch_size=n_cluster, max_iter=5000)
    y_km = km.fit_predict(des_all)
    codebook = km.cluster_centers_
    

    np.save(os.path.join(args.result_path,"codebook_sklearn_gray_{}".format(n_cluster)), codebook)

# Mini-Batch K-Means++ Clustering
def fastDesToBookPlus(descriptor: np.ndarray, K: int, batch=1024, iter=5000):
    N,D = descriptor.shape
    book = np.zeros((K, D))

    # initialize
    init_size = 3 * batch
    if init_size < K:
        init_size = 3 * K
    init_rand = randomIntList(init_size, 0, N-1)
    init_des = descriptor[init_rand]

    center_index = randomIntList(1, 0, init_size-1)
    book = init_des[center_index[0]]
    book = book[np.newaxis, :]

    print("initializing...")
    for i in tqdm(range(1, K)):
        
        dist = ((init_des[:, :, np.newaxis] - book.T[np.newaxis, :, :])**2).sum(axis=1)
        dist = dist.min(axis=1)
        prob_dist = dist / dist.sum()

        center_index.append(np.random.choice(np.array(init_des.shape[0]), size=1, replace=False, p = prob_dist))

        book = np.r_[book, init_des[center_index[i]]]

    label_count = np.zeros((K), dtype=int)
    print("clustering...")
    for i in tqdm(range(iter)):
        # randomly pick examples
        rand_batch = randomIntList(batch, 0, N-1)
        des_batch = descriptor[rand_batch]

        label = np.zeros((batch), dtype=int)
        for n in range(batch):
            des = np.broadcast_to(des_batch[n],(K, D))
            tmp = ((des - book)**2).sum(axis=1)
            index = np.argmin(tmp)
            label[n] = index

        for n in range(batch):
            label_count[label[n]] += 1
            eta = 1 / label_count[label[n]]
            book[label[n]] = (1-eta) * book[label[n]] + eta * des_batch[n]

    return book
    
# Mini-Batch K-Means Clustering
def fastDesToBook(descriptor: np.ndarray, K: int, batch=1000, iter=2000):
    N,D = descriptor.shape
    book = np.zeros((K, D))
    
    # initialize
    if N >= K:
        rand_list = randomIntList(K, 0, N-1)
        for i in range(K):
            book[i] = descriptor[rand_list[i]]
    else:
        rand_list = randomIntList(N, 0, K-1)
        for i in range(N):
            book[rand_list[i]] = descriptor[i]
    
    
    label_count = np.zeros((K), dtype=int)
    for i in tqdm(range(iter)):
        # randomly pick examples
        rand_batch = randomIntList(batch, 0, N-1)
        des_batch = np.zeros((batch, D))
        for j in range(batch):
            des_batch[j] = descriptor[rand_batch[j]]

        label = np.zeros((batch), dtype=int)
        for n in range(batch):
            des = np.broadcast_to(des_batch[n],(K, D))
            tmp = np.power(des - book, 2).sum(axis=1)
            index = np.argmin(tmp)
            label[n] = index

        for n in range(batch):
            label_count[label[n]] += 1
            eta = 1 / label_count[label[n]]
            book[label[n]] = (1-eta) * book[label[n]] + eta * des_batch[n]

    return book

# K-Means Clustering
def desToBook(descriptor: np.ndarray, K: int):
    N,D = descriptor.shape
    book = np.zeros((K, D))
    
    # K-means
    # initialize
    if N >= K:
        rand_list = randomIntList(K, 0, N-1)
        for i in range(K):
            book[i] = descriptor[rand_list[i]]
    else:
        rand_list = randomIntList(N, 0, K-1)
        for i in range(N):
            book[rand_list[i]] = descriptor[i]
    
    prev_book = np.zeros((K, D))
    count=0
    while(True):
        prev_book = book.copy()
        
        label = np.zeros((N))
        for n in range(N):
            des = np.broadcast_to(descriptor[n],(K, D))
            tmp = np.power(des - book, 2).sum(axis=1)
            index = np.argmin(tmp)
            label[n] = index
            
        for k in range(K):
            avg = np.average(descriptor[np.where(label==k)[0]],axis=0)
            book[k] = avg
            
        count += 1
        if np.allclose(prev_book, book):
            break
    
    #print(count)
    return book



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", help="input image root dir path", required=True)
    parser.add_argument("--result_path", help="output dir path", required=True)
    args = parser.parse_args()

    main(args)