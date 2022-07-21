import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from load_data import *
from load_data import *
from stgcn import *
from stgcn import *
from utils import *
from utils import *

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
    epochs = 50
    batch_size = 100
    lr = 1e-3
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
# scaler = StandardScaler() #z-score正态标准化,不适用于0-1数据集
# scaler = MinMaxScaler() #minmax标准化,0-1数据集没有必要了
# train = scaler.fit_transform(train)
# val = scaler.transform(val)
# test = scaler.transform(test)


# x_train, y_train = data_transform(train, n_his, n_pred, day_slot, device)
# x_val, y_val = data_transform(val, n_his, n_pred, day_slot, device)
# x_test, y_test = data_transform(test, n_his, n_pred, day_slot, device)
x_train, y_train = data_transform_our(train, CFG.n_his, CFG.n_pred, device)
x_val, y_val = data_transform_our(val, CFG.n_his, CFG.n_pred, device)
x_test, y_test = data_transform_our(test, CFG.n_his, CFG.n_pred, device)

train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_data, CFG.batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, CFG.batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data, CFG.batch_size)

if CFG.model_name == 'FNN':
    model = FNN(n_his=CFG.n_his, n_route=CFG.n_route).to(device)
    best_model = FNN(n_his=CFG.n_his, n_route=CFG.n_route).to(device)
elif CFG.model_name == 'LSTM':
    model = LSTM(batch_size=CFG.batch_size, n_his=CFG.n_his, n_route=CFG.n_route).to(device)
    best_model = model = LSTM(batch_size=CFG.batch_size, n_his=CFG.n_his, n_route=CFG.n_route).to(device)
elif CFG.model_name == 'STGCN':
    model = STGCN(CFG.Ks, CFG.Kt, CFG.blocks, CFG.n_his, CFG.n_route, Lk, CFG.drop_prob).to(device)
    best_model = STGCN(CFG.Ks, CFG.Kt, CFG.blocks, CFG.n_his, CFG.n_route, Lk, CFG.drop_prob).to(device)
elif CFG.model_name == 'STGCN_LSTM':
    model = STGCN_LSTM(CFG.Ks, CFG.Kt, CFG.blocks, CFG.n_his, CFG.n_route, Lk, CFG.drop_prob, batch_size=CFG.batch_size,
                       n_his=CFG.n_his, n_route=CFG.n_route).to(device)
    best_model = STGCN_LSTM(CFG.Ks, CFG.Kt, CFG.blocks, CFG.n_his, CFG.n_route, Lk, CFG.drop_prob,
                            batch_size=CFG.batch_size, n_his=CFG.n_his, n_route=CFG.n_route).to(device)
else:
    assert NotImplementedError  # wait for implement

if CFG.optimzer_name == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
elif CFG.optimzer_name == 'RMSprop':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=CFG.lr)
else:
    assert NotImplementedError  # wait for implement
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

if CFG.loss_name == 'MSE':
    loss = nn.MSELoss()
else:
    assert NotImplementedError  # wait for implement

min_val_loss = np.inf  # 正无穷
train_loss_his = []  # 保存训练误差
val_loss_his = []  # 保存测试误差，方便绘图
epoch_his = []  # 保存epoch

for epoch in range(1, CFG.epochs + 1):
    l_sum, n = 0.0, 0
    model.train()
    for x, y in train_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    scheduler.step()
    val_loss = evaluate_model(model, loss, val_iter)
    # print(np.array(val_loss).shape)
    print(val_loss[0])
    # print(min_val_loss.shape)
    if val_loss[0] < min_val_loss:
        min_val_loss = val_loss[0]
        torch.save(model.state_dict(), save_path)
    print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)
    train_loss_his.append(l_sum / n)
    val_loss_his.append(val_loss[0])
    epoch_his.append(epoch)

plt.figure()
plt.plot(epoch_his, train_loss_his, 'g', label='Train Loss')
plt.plot(epoch_his, val_loss_his, 'k', label='Test Loss')
plt.grid(True)
plt.xlabel('epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.legend(loc="upper right", fontsize=15)
plt.show()

best_model.load_state_dict(torch.load(save_path))

if CFG.aug:
    def evaluate_model(model, loss, data_iter):
        def get_year(i):
            return (4590 + i) // 365

        model.eval()
        print(1)
        l_sum, n = 0.0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(data_iter):
                t = get_year(i)
                if CFG.year[t] == 1:
                    y_pred = model(x).view(len(x), -1) * 2
                else:
                    y_pred = model(x).view(len(x), -1)
                l = loss(y_pred, y)
                l_sum += l.item() * y.shape[0]
                n += y.shape[0]
            return l_sum / n, y_pred, y


    def evaluate_metric(model, data_iter, scaler):
        model.eval()

        def get_year(i):
            return (4590 + i) // 365

        with torch.no_grad():
            # mae, mape, mse = [], [], []
            mae, mse, mre, y_true = [], [], [], []
            for i, (x, y) in enumerate(data_iter):
                y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)

                t = get_year(i)
                if CFG.year[t] == 1:
                    y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1) * 2
                else:
                    y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
                # print(x.shape)
                # print(y.shape)
                # print(y_pred.shape)
                d = np.abs(y - y_pred)
                mae += d.tolist()
                # mape += (d / y).tolist()
                mre += d.tolist()
                y_true += y.tolist()
                mse += (d ** 2).tolist()
            MAE = np.array(mae).mean()
            # MAPE = np.array(mape).mean()
            MRE = np.array(mre).sum() / np.array(y_true).sum()
            RMSE = np.sqrt(np.array(mse).mean())
            # return MAE, MAPE, RMSE
            return MAE, MRE, RMSE
else:
    def evaluate_model(model, loss, data_iter):
        model.eval()
        l_sum, n = 0.0, 0
        with torch.no_grad():
            for x, y in data_iter:
                y_pred = model(x).view(len(x), -1)
                l = loss(y_pred, y)
                l_sum += l.item() * y.shape[0]
                n += y.shape[0]
            return l_sum / n, y_pred, y


    def evaluate_metric(model, data_iter, scaler):
        model.eval()
        with torch.no_grad():
            # mae, mape, mse = [], [], []
            mae, mse, mre, y_true = [], [], [], []
            for x, y in data_iter:
                y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
                y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
                # print(x.shape)
                # print(y.shape)
                # print(y_pred.shape)
                d = np.abs(y - y_pred)
                mae += d.tolist()
                # mape += (d / y).tolist()
                mre += d.tolist()
                y_true += y.tolist()
                mse += (d ** 2).tolist()
            MAE = np.array(mae).mean()
            # MAPE = np.array(mape).mean()
            MRE = np.array(mre).sum() / np.array(y_true).sum()
            RMSE = np.sqrt(np.array(mse).mean())
            # return MAE, MAPE, RMSE
            return MAE, MRE, RMSE

l, y_pred, y = evaluate_model(best_model, loss, test_iter)
# MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
# print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
MAE, MRE, RMSE = evaluate_metric(best_model, test_iter, scaler)
print("test loss:", l, "\nMAE:", MAE, ", MRE:", MRE, ", RMSE:", RMSE)
TP, TN, FP, FN, precision, recall, acc, f1 = evaluate_metric_classification(best_model, test_iter, scaler)
print("TP:", TP, ", TN", TN, ", FP", FP, ", FN", FN)
print("precision:", precision, ", recall", recall, ", acc", acc, ", f1", f1)

plt.axis('off')
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

############vis
# import geopandas as gp
# from geopandas import GeoDataFrame
# import matplotlib.pyplot as plt
# afg_geo = GeoDataFrame.from_file('Afghanistan_Districts.geojson')
# afg_geo['value'] = 0
# afg_geo.plot()
# l,y_pred,y = evaluate_model(best_model, loss, test_iter) # 获得预测和真实
# # def minmaxscaler(data): # 标准化
# #     min = np.amin(data)
# #     max = np.amax(data)
# #     return (data - min)/(max-min)
# y = y.cpu().detach().numpy()
# y_pred = y_pred.cpu().detach().numpy()
# # y_pred = minmaxscaler(y_pred)
# k = afg_geo['PROV_34_NA'].unique()
# k.sort()
# print(k)
# dictionary = dict(zip(k, y[0]))
# print(dictionary)
# new_df = afg_geo['PROV_34_NA'].apply(lambda x: dictionary[x])
# afg_geo['value'] = new_df
# afg_geo.plot(column='value',cmap='Reds',)
# dictionary = dict(zip(k, y_pred[0]))
# new_df = afg_geo['PROV_34_NA'].apply(lambda x: dictionary[x])
# afg_geo['value'] = new_df
# afg_geo.plot(column='value',cmap='Reds',)
# dictionary = dict(zip(k, y_pred[0]))
# for k,v in dictionary.items():
#     if v > 0.5:
#         dictionary[k] = 1
#     else:
#         dictionary[k] = 0
# new_df = afg_geo['PROV_34_NA'].apply(lambda x: dictionary[x])
# afg_geo['value'] = new_df
# afg_geo.plot(column='value',cmap='Reds',)
