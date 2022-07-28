import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import datasets
from create_codebook import fastDesToBook, fastDesToBookPlus
from create_bovw import desToHist
import numpy as np
from utils.utils import KMeans_pp


iris = datasets.load_iris()
data = iris.data
n_clusters = 4

des = np.array(data)
#book = fastDesToBook(des, n_clusters, 20, 5000)
book = fastDesToBookPlus(des, n_clusters, 20, 5000)
print(book)

label_all = np.zeros((des.shape[0]))
for i in range(des.shape[0]):
    label_all[i] = np.argmax(desToHist(des[i].reshape(1, 4), book), axis=0)

#model = KMeans_pp(n_clusters, 5000)
#model.fit(des)
#label_all = model.labels_
#book = model.cluster_centers_


km = cluster.MiniBatchKMeans(n_clusters, init="k-means++", batch_size=20, max_iter=5000)

y_km = km.fit_predict(des)
print(km.cluster_centers_)
colors = ["red", "blue", "green", "black", "cyan", "white", "magenta", "yellow"]

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)

for i in range(n_clusters):
    ax.scatter(data[:,0][label_all==i], data[:,1][label_all==i], color=colors[i])
    ax.scatter(book[i,0], book[i,1], marker="x", color=colors[i], s=300)
    ax.scatter(km.cluster_centers_[i, 0], km.cluster_centers_[i, 1], color=colors[i], marker="+", s=300)

fig.savefig("test_cluster.png")
