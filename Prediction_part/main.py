import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from load_data import *
from load_data import *
from stgcn import *
from stgcn import *
from utils import *
from utils import *

matrix_path = "dataset/Adj_similarity.csv"
# Adj_total Adj_distance Adj_self Adj_similarity
data_path = "dataset/Afg_data.csv"
save_path = "save/model.pt"
torch.manual_seed(2333)
np.random.seed(2333)
random.seed(2333)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CFG:
    # ['FNN','LSTM','STGCN','STGCN_LSTM', 'Single_STGCN', 'STGCN_LSTM_OUT'
    # "STGCN_Attention", "STGCN_LSTM_SA","GRU"]
    #  ['STGCN_LSTM', "STGCN", "STGCN_Attention"]
    model_name = ["STGCN"]
    epochs = 15
    batch_size = 128
    lr = 1e-3
    # train : val : test = 8 : 1 : 1
    n_train, n_val = int(pd.read_csv(data_path).shape[0] * 0.8), int(pd.read_csv(data_path).shape[0] * 0.1)
    n_his = 30
    n_pred = [1, 5, 10]
    n_vertice = 34
    Ks, Kt = 3, 3
    # blocks = [[1, 32, 64], [64, 32, 128]]
    blocks = [[1, 16, 32], [32, 16, 64]]
    blocks_singe = [1, 16, 32]
    drop_prob = 0
    loss_name = 'Focal'  # ['MSE','Focal','BCE']...
    optimzer_name = 'RMSprop'  # ['SGD','ADAMP','RMSprop']


for mm in CFG.model_name:
    print(mm)
    performance = list(['model', 'future', "acc", "pre", "recall", "f1", "roc_auc"])
    for future in CFG.n_pred:  # All the tests use 30 days as the historical time window, to forecast attack risk in the next 1, 5, 10 days.
        print("Forecasting ahead days: ", future)
        W = load_matrix(matrix_path)
        L = scaled_laplacian(W)
        Lk = cheb_poly(L, CFG.Ks)
        Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
        train, val, test = load_data(data_path, CFG.n_train, CFG.n_val)

        x_train, y_train = data_transform_our(train, CFG.n_his, future, device)
        x_val, y_val = data_transform_our(val, CFG.n_his, future, device)
        x_test, y_test = data_transform_our(test, CFG.n_his, future, device)
        train_data = torch.utils.data.TensorDataset(x_train, y_train)
        train_iter = torch.utils.data.DataLoader(train_data, CFG.batch_size, shuffle=True)
        val_data = torch.utils.data.TensorDataset(x_val, y_val)
        val_iter = torch.utils.data.DataLoader(val_data, CFG.batch_size)
        test_data = torch.utils.data.TensorDataset(x_test, y_test)
        test_iter = torch.utils.data.DataLoader(test_data, CFG.batch_size)

        if mm == 'FNN':
            model = FNN(n_his=CFG.n_his, n_vertice=CFG.n_vertice).to(device)
            best_model = FNN(n_his=CFG.n_his, n_vertice=CFG.n_vertice).to(device)
        elif mm == 'LSTM':
            model = LSTM(batch_size=CFG.batch_size, n_his=CFG.n_his, n_vertice=CFG.n_vertice).to(device)
            best_model = model = LSTM(batch_size=CFG.batch_size, n_his=CFG.n_his, n_vertice=CFG.n_vertice).to(device)
        elif mm == 'GRU':
            model = GRU(batch_size=CFG.batch_size, n_his=CFG.n_his, n_vertice=CFG.n_vertice).to(device)
            best_model = GRU(batch_size=CFG.batch_size, n_his=CFG.n_his, n_vertice=CFG.n_vertice).to(device)
        elif mm == 'STGCN':
            model = STGCN(CFG.Ks, CFG.Kt, CFG.blocks, CFG.n_his, CFG.n_vertice, Lk, CFG.drop_prob).to(device)
            best_model = STGCN(CFG.Ks, CFG.Kt, CFG.blocks, CFG.n_his, CFG.n_vertice, Lk, CFG.drop_prob).to(device)
        elif mm == 'STGCN_LSTM':
            model = STGCN_LSTM(CFG.Ks, CFG.Kt, CFG.blocks, CFG.n_his, CFG.n_vertice, Lk, CFG.drop_prob,
                               batch_size=CFG.batch_size, n_his=CFG.n_his, n_vertice=CFG.n_vertice).to(device)
            best_model = STGCN_LSTM(CFG.Ks, CFG.Kt, CFG.blocks, CFG.n_his, CFG.n_vertice, Lk, CFG.drop_prob,
                                    batch_size=CFG.batch_size, n_his=CFG.n_his, n_vertice=CFG.n_vertice).to(device)
        elif mm == 'Single_STGCN':
            model = single_layer_STGCN(CFG.Ks, CFG.Kt, CFG.blocks_singe, CFG.n_his, CFG.n_vertice, Lk,
                                       CFG.drop_prob).to(device)
            best_model = single_layer_STGCN(CFG.Ks, CFG.Kt, CFG.blocks_singe, CFG.n_his, CFG.n_vertice, Lk,
                                            CFG.drop_prob).to(device)
        elif mm == 'STGCN_LSTM_OUT':
            model = STGCN_LSTM_OUT(CFG.Ks, CFG.Kt, CFG.blocks, CFG.n_his, CFG.n_vertice, Lk, CFG.drop_prob).to(device)
            best_model = STGCN_LSTM_OUT(CFG.Ks, CFG.Kt, CFG.blocks, CFG.n_his, CFG.n_vertice, Lk, CFG.drop_prob).to(
                device)
        elif mm == 'STGCN_Attention':
            model = STGCN_Attention(CFG.Ks, CFG.Kt, CFG.blocks, CFG.n_his, CFG.n_vertice, Lk, CFG.drop_prob,
                                    batch_size=CFG.batch_size, n_his=CFG.n_his, n_vertice=CFG.n_vertice).to(device)
            best_model = STGCN_Attention(CFG.Ks, CFG.Kt, CFG.blocks, CFG.n_his, CFG.n_vertice, Lk, CFG.drop_prob,
                                         batch_size=CFG.batch_size, n_his=CFG.n_his, n_vertice=CFG.n_vertice).to(device)
        elif mm == 'STGCN_LSTM_SA':
            model = STGCN_LSTM_SA(CFG.Ks, CFG.Kt, CFG.blocks, CFG.n_his, CFG.n_vertice, Lk, CFG.drop_prob,
                                  batch_size=CFG.batch_size, n_his=CFG.n_his, n_vertice=CFG.n_vertice).to(device)
            best_model = STGCN_LSTM_SA(CFG.Ks, CFG.Kt, CFG.blocks, CFG.n_his, CFG.n_vertice, Lk, CFG.drop_prob,
                                       batch_size=CFG.batch_size, n_his=CFG.n_his, n_vertice=CFG.n_vertice).to(device)

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
        if CFG.loss_name == 'Focal':
            loss = FocalLossV1(alpha=0.82, gamma=3, reduction='mean')
        if CFG.loss_name == 'BCE':
            loss = nn.BCEWithLogitsLoss()  # 等于sigmoid+BCElossl

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
            if val_loss[0] < min_val_loss:
                min_val_loss = val_loss[0]
                torch.save(model.state_dict(), save_path)
            print("epoch", epoch, ", train loss:", round(l_sum / n, 4),
                  ", validation loss:", round(val_loss[0], 4))
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
        plt.savefig("result/" + mm + "-" + str(future) + '.png'
                    , dpi=330, bbox_inches='tight')
        plt.show()

        if matrix_path != "dataset/Adj_self.csv":
            best_model.load_state_dict(torch.load(save_path))
            model = best_model
            model.eval()
        elif matrix_path == "dataset/Adj_self.csv":
            model.eval()
        with torch.no_grad():
            y_real_list, y_pre_list = [], []
            for x, y in test_iter:
                y = y.cpu().numpy()
                # print(y)
                y_real_list += y.tolist()
                y_pred = torch.sigmoid(model(x)).view(len(x), -1).cpu().numpy()
                # print(y_pred)
                y_pre_list += y_pred.tolist()
        y_real_list = np.array(y_real_list)
        y_pre_list = np.array(y_pre_list)
        y_pre_list = np.where(y_pre_list >= 0.5, 1, y_pre_list)
        y_pre_list = np.where(y_pre_list < 0.5, 0, y_pre_list)

        # evaluate: micro label-based metrix
        from sklearn.metrics import roc_auc_score, hamming_loss, multilabel_confusion_matrix, confusion_matrix

        GT = y_real_list  # groud truth
        pred = y_pre_list  # predicted
        quantity = label_quantity(GT, pred)  # tp, fp, tn, fn
        acc = label_accuracy_micro(GT, pred)
        pre = label_precision_micro(GT, pred)
        recall = label_recall_micro(GT, pred)
        f1 = label_f1_micro(GT, pred)
        roc_auc = roc_auc_score(GT, pred, average="micro")
        hamming = hamming_loss(GT, pred)

        confusion_matrix = multilabel_confusion_matrix(GT, pred)
        np.save("confusion_matrix.npy", confusion_matrix)
        GT = pd.DataFrame(GT)
        pred = pd.DataFrame(pred)
        filename = "result/" + mm + "-" + str(future) + "-prediction.xlsx"  # excel result of each model
        with pd.ExcelWriter(filename) as writer:
            GT.to_excel(writer, sheet_name='Ground_truth')
            pred.to_excel(writer, sheet_name=str("Prediction-" + mm + "-" + str(future)))

        performance = np.row_stack((performance, [mm, future, acc, pre, recall, f1, roc_auc]))

    pd.DataFrame(performance).to_csv(str("result/" + mm + "-performance.csv")
                                     , index=False, header=False)

# # ROC curve
# from sklearn.metrics import roc_curve, auc
# from itertools import cycle
# lw = 1
# GT = y_real_list #groud truth
# pred = y_pre_list #predicted
# n_classes = y_real_list.shape[1]
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(GT[:, i], pred[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(GT.ravel(), pred.ravel()) #np.ravel to 1-D array
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# # Finally average it and compute AUC
# mean_tpr /= n_classes

# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# # Plot all ROC curves
# plt.figure(figsize=(15, 10))
# plt.plot(
#     fpr["micro"],
#     tpr["micro"],
#     label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
#     color="deeppink",
#     linestyle=":",
#     linewidth=4,
# )

# plt.plot(
#     fpr["macro"],
#     tpr["macro"],
#     label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
#     color="navy",
#     linestyle=":",
#     linewidth=4,
# )

# colors = cycle(["aqua", "darkorange", "cornflowerblue"])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(
#         fpr[i],
#         tpr[i],
#         color=color,
#         lw=lw,
#         label="ROC curve of Province {0} (area = {1:0.2f})".format(i, roc_auc[i]),
#     )

# plt.plot([0, 1], [0, 1], "k--", lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# # plt.title("Some extension of Receiver operating characteristic to multiclass")
# plt.legend(loc="best")
# plt.show()
