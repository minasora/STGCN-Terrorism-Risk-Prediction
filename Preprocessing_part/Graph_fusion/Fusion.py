# graph fusion 思路来自 Deep Temporal Multi-Graph Convolutional Network for Crime Prediction_39th
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns

def normalize_adj(A):#矩阵相乘https://www.cnblogs.com/Mrfanl/p/11252015.html
    n = A.shape[0]
    d = np.sum(A, axis=1) #每个节点的度
    D_inv = np.linalg.inv(np.diag(d))  #D：度矩阵， 矩阵求逆https://www.py.cn/faq/python/18293.html
    A_norm = np.matmul(D_inv,A) + np.eye(n)
    return A_norm


A_distance = np.genfromtxt('Adj_distance.csv', dtype = str, delimiter = ','
                     , skip_header = 0).astype(float)
A_similarity = np.genfromtxt('Adj_similarity.csv', dtype = str, delimiter = ','
                     , skip_header = 0).astype(float)
A_self = np.genfromtxt('Adj_self.csv', dtype = str, delimiter = ','
                     , skip_header = 0).astype(float)

A_distance_norm = normalize_adj(A_distance)

A_similarity_norm = normalize_adj(A_similarity)

A_self = normalize_adj(A_self)

graph_num = 3 ###决定几张图融合

if graph_num==2:
    #两图融合，感觉故事不得够
    print("Two graph fusion")
    W1, W2= softmax(np.array([1.0/2, 1.0/2])) #计算不同图的权重
    Adj = np.multiply(A_distance,W1) + np.multiply(A_similarity_norm,W2)
if graph_num==3:
    ### 3图融合，似乎自身关联太严重
    print("Three graph fusion")
    W1, W2, W3= softmax(np.array([1.0/3, 1.0/3, 1.0/3]))
    Adj = np.multiply(A_distance,W1) + np.multiply(A_similarity_norm,W2) + np.multiply(A_self,W3)

### 画Adj https://blog.csdn.net/qq_42898981/article/details/102836530
plt.subplots(figsize=(9, 9))
#sns.heatmap(pearson_matrix, annot=True, vmax=1, square=True, cmap="Blues")
sns.heatmap(Adj, cmap="Blues")
plt.show()

np.savetxt('Adj_total.csv', Adj, fmt='%s', delimiter=',')###最后fusion的Adj
