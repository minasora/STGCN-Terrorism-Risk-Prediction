# -*- coding: utf-8 -*-
import random
import sys

import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch
import torch.nn.functional as F
from geopandas import GeoDataFrame
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from load_data import *
from load_data import *
from stgcn import *
from stgcn import *
from utils import *
from utils import *

############## initial ############################
matrix_path = "dataset/Adj_total.csv"
data_path = "dataset/Afg_data.csv"
save_path = "save/model.pt"
torch.manual_seed(2333)
np.random.seed(2333)
random.seed(2333)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CFG:
    model_name = 'STGCN_LSTM'  # ['FNN','LSTM','STGCN','STGCN_LSTM']
    epochs = 10
    batch_size = 32
    lr = 1e-6
    day_slot = 34
    n_train, n_val, n_test = 120, 15, 15
    n_his = 30
    n_pred = 1
    n_route = 34
    Ks, Kt = 3, 3
    blocks = [[1, 32, 64], [64, 32, 128]]
    drop_prob = 0
    loss_name = 'MSE'  # ['BCE','Focal_loss']...
    optimzer_name = 'RMSprop'  # ['SGD','ADAMP','RMSprop']
    aug = True  # if do the change
    year = [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]


W = load_matrix(matrix_path)
L = scaled_laplacian(W)
Lk = cheb_poly(L, CFG.Ks)
Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
train, val, test = load_data(data_path, CFG.n_train * CFG.day_slot, CFG.n_val * CFG.day_slot)
scaler = MinMaxScaler()  # minmax标准化
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)
x_train, y_train = data_transform_our(train, CFG.n_his, CFG.n_pred, device)
x_val, y_val = data_transform_our(val, CFG.n_his, CFG.n_pred, device)
x_test, y_test = data_transform_our(test, CFG.n_his, CFG.n_pred, device)
if CFG.model_name == 'FNN':
    best_model = FNN(n_his=CFG.n_his, n_route=CFG.n_route).to(device)
elif CFG.model_name == 'LSTM':
    best_model = model = LSTM(batch_size=CFG.batch_size, n_his=CFG.n_his, n_route=CFG.n_route).to(device)
elif CFG.model_name == 'STGCN':
    best_model = STGCN(CFG.Ks, CFG.Kt, CFG.blocks, CFG.n_his, CFG.n_route, Lk, CFG.drop_prob).to(device)
elif CFG.model_name == 'STGCN_LSTM':
    best_model = STGCN_LSTM(CFG.Ks, CFG.Kt, CFG.blocks, CFG.n_his, CFG.n_route, Lk, CFG.drop_prob,
                            batch_size=CFG.batch_size, n_his=CFG.n_his, n_route=CFG.n_route).to(device)
else:
    assert NotImplementedError  # wait for implement
train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_data, CFG.batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, CFG.batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data, CFG.batch_size)
if CFG.loss_name == 'MSE':
    loss = nn.MSELoss()
else:
    assert NotImplementedError  # wait for implement

############## main: vis ############################
# 0：地图
best_model.load_state_dict(torch.load(save_path))
afg_geo = GeoDataFrame.from_file('Afghanistan_Districts.geojson')
afg_geo['value'] = 0
afg_geo.plot()  # 地图
# 1：Ground truth 着色地图
l, y_pred, y = evaluate_model(best_model, loss, test_iter)  # 获得预测和真实


def minmaxscaler(data):  # 标准化
    min = np.amin(data)
    max = np.amax(data)
    return (data - min) / (max - min)


y = y.cpu().detach().numpy()
y_pred = y_pred.cpu().detach().numpy()
y_pred = minmaxscaler(y_pred)
k = afg_geo['PROV_34_NA'].unique()
k.sort()
print(k)
dictionary = dict(zip(k, y[0]))
print(dictionary)
new_df = afg_geo['PROV_34_NA'].apply(lambda x: dictionary[x])
afg_geo['value'] = new_df
afg_geo.plot(column='value', cmap='Blues', )
# 2：预测值 连续值 着色地图
dictionary = dict(zip(k, y_pred[0]))
new_df = afg_geo['PROV_34_NA'].apply(lambda x: dictionary[x])
afg_geo['value'] = new_df
afg_geo.plot(column='value', cmap='Blues', )
# 3：预测值 分类 (0.5) 着色地图
dictionary = dict(zip(k, y_pred[0]))
for k, v in dictionary.items():
    if v > 0.5:
        dictionary[k] = 1
    else:
        dictionary[k] = 0
new_df = afg_geo['PROV_34_NA'].apply(lambda x: dictionary[x])
afg_geo['value'] = new_df
afg_geo.plot(column='value', cmap='Blues', )
