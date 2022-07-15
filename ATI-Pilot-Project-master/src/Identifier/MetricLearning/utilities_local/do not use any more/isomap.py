#!/usr/bin/env python 3.7
# coding: utf-8

import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

'''
实现了Isomap，使用的是k近邻，Floyd算法

总结：
首先用sklearn生成100个散点的S形状的数据来进行测试，得到效果图，与sklearn中的Isomap进行对比
'''


def myMDS(D, d):
    # dist = pdist(D, 'euclidean')
    # dist = squareform(dist)          # 转化为方阵
    # dist2 = dist**2
    dist2 = D ** 2  # dist2 = [dist^2]
    m = dist2.shape[0]
    disti2 = 1 / m * dist2.sum(axis=1, keepdims=True)  # (m, 1)
    distj2 = 1 / m * dist2.sum(axis=0, keepdims=True)  # (1, m)
    distij2 = 1 / (m ** 2) * dist2.sum()  # (1, 1)
    B = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            B[i][j] = -0.5 * (dist2[i][j] - disti2[i][0] - distj2[0][j] + distij2)
    A, U = np.linalg.eig(B)  # np.linalg.eig获得的A是特征值，T是特征向量矩阵，且T的列向量是特征向量
    top_A_idx = A.argsort()[::-1][:d]  # 获得最大的k个特征值的索引
    top_A = A[top_A_idx]  # 获得最大的k个特征值
    top_U = U[:, top_A_idx]
    lambd = np.diag(top_A)
    Z = np.dot(top_U, np.sqrt(lambd))
    return Z


def get_k_maxtria(D, k):
    '''
    构建近邻矩阵
    输入：样本集D，近邻参数k
    输出：近邻矩阵k_dist
    '''
    dist = pdist(D, 'euclidean')  # 获得距离矩阵
    dist = squareform(dist)  # 转化为方阵
    inf = float('inf')
    m = dist.shape[0]
    k_dist = np.ones([m, m]) * inf
    for i in range(m):
        topk = np.argpartition(dist[i], k)[:k + 1]
        k_dist[i][topk] = dist[i][topk]
    return k_dist


def Floyd(X):
    '''
    Floyd算法：
    输入: 距离矩阵X：(m,m)
        算法核心 path[i,j]:=min{path[i,k]+path[k,j],path[i,j]}
        时间复杂度：O（n^3)，空间复杂度：O(n^2)
    '''
    m = X.shape[0]
    for k in range(m):
        for i in range(m):
            for j in range(m):
                X[i][j] = min(X[i][j], X[i][k] + X[k][j])
    return X


def myIsomap(D, k, d):
    '''
    Isomap
    输入：样本集D，近邻参数k，降维后的维度d
    输出：降维后的数据Z
    '''
    k_dist = get_k_maxtria(D, k)
    count = k_dist.shape[0]
    dist = Floyd(k_dist)
    Z = myMDS(dist, d)
    return Z



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import datasets
from sklearn.manifold import Isomap

# Next line to silence pyflakes. This import is needed.
Axes3D

n_points = 100
X, color = datasets.make_s_curve(n_points, random_state=0)
n_neighbors = 10
n_components = 2

fig = plt.figure(figsize=(15, 8))
plt.suptitle("Manifold Learning with %i points, %i neighbors"
             % (1000, n_neighbors), fontsize=14)

ax = fig.add_subplot(251, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)

Y = myIsomap(X, 15, 2)
ax = fig.add_subplot(252)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
plt.xlabel("myIsomap with %i neighbors" % (15))
plt.suptitle("Isomap Learning with %i points" % (100), fontsize=14)

Y = myIsomap(X, 50, 2)
ax = fig.add_subplot(252 + 1)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
plt.xlabel("myIsomap with %i neighbors" % (50))


Y = Isomap(n_neighbors=15, n_components=2).fit_transform(X)
ax = fig.add_subplot(252 + 2)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
plt.xlabel("sklearn Isomap with %i neighbors" % (15))


Y = Isomap(n_neighbors=50, n_components=2).fit_transform(X)
ax = fig.add_subplot(252 + 3)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
plt.xlabel("sklearn Isomap with %i neighbors" % (50))
plt.show()
