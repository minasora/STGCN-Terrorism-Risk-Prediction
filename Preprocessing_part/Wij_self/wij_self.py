import torch
import numpy as np


wij_self = np.eye(34) ###34矩阵自身关联
np.savetxt('Adj_self.csv', wij_self, fmt='%s', delimiter=',')###无索引Adj
