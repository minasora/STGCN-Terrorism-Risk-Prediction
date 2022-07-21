import random

import numpy as np
import pandas as pd
import torch


def delete_space(data):  # list format data
    for i in range(len(data)):
        data[i] = "".join(data[i].split())
    return data


df = pd.read_csv("dataset/Afg_data_with_province.csv", header=None).values
df = df[:, 1:]
per = np.random.permutation(df.shape[1])
df = df[:, per]

# Select several provinces to plot con-currence fig
df = df[:, -10:]
prov = df[0, :]
prov = delete_space(prov)
data = df[1:, :].astype(int)

coocc = data.T.dot(data)  # concurrence among labels/provinces
np.fill_diagonal(coocc, 0)  # 对角线维0
# coocc = np.triu(coocc, k=0) #下三角为0
coocc = np.column_stack((prov, coocc))
title = np.insert(df[0, :], 0, "Province")
title = delete_space(title)
coocc = np.row_stack((title, coocc)).astype(str)

np.savetxt(str("dataset/concorrence matrix.csv"), coocc, delimiter=',',
           fmt='%s')

np.savetxt(str("dataset/concorrence matrix.txt"), coocc, delimiter='\t',
           fmt='%s')
