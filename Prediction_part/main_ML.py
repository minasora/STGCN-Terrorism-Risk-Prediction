# -*- coding: utf-8 -*-
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from load_data import *
from load_data import *
from stgcn import *
from stgcn import *
from utils import *
from utils import *

data_path = "dataset/Afg_data.csv"


class CFG:
    # ['RF', 'XGB', 'Logit', 'SVM', 'DT', 'KNN'] # SVM cannot run
    model_name = 'XGB'
    # train : val : test = 8 : 1 : 1
    n_train, n_val = int(pd.read_csv(data_path).shape[0] * 0.8), int(pd.read_csv(data_path).shape[0] * 0.1)
    n_his = 30
    n_pred = [1, 5, 10]
    n_vertice = 34


print(CFG.model_name)
performance = list(['model', 'future', "acc", "pre", "recall", "f1", "roc_auc", "hamming"])
for future in CFG.n_pred:  # All the tests use 30 days as the historical time window, to forecast attack risk in the next 1, 5, 10 days.
    print("Forecasting ahead days: ", future)
    train, val, test = load_data(data_path, CFG.n_train, CFG.n_val)
    x_train, y_train = data_transform_our_numpy(train, CFG.n_his, future)
    x_val, y_val = data_transform_our_numpy(val, CFG.n_his,
                                            future)  # validation set not used in classical machine learning models
    x_test, y_test = data_transform_our_numpy(test, CFG.n_his, future)
    x_train = x_train.reshape(x_train.shape[0], (x_train.shape[1] * x_train.shape[2] * x_train.shape[3]))
    x_test = x_test.reshape(x_test.shape[0], (x_test.shape[1] * x_test.shape[2] * x_test.shape[3]))

    ### model train
    weight_for_RF = [{0: 1, 1: 100} for i in range(34)]
    RF = RandomForestClassifier(class_weight=weight_for_RF, n_jobs=-1, random_state=233)
    # RF = RandomForestClassifier(n_jobs=-1, random_state=233)
    XGB = XGBClassifier(scale_pos_weight=20, n_jobs=-1)
    Logit = MultiOutputClassifier(LogisticRegression(solver="liblinear",
                                                     random_state=0, n_jobs=-1))
    SVM = MultiOutputClassifier(SVC(class_weight="balanced", kernel='linear'))
    DT = DecisionTreeClassifier(random_state=0, class_weight="balanced")
    KNN = KNeighborsClassifier(n_jobs=-1)
    # models ['RF', 'XGB', 'Logit', 'SVM', 'DT']
    if CFG.model_name == 'RF':
        model = RF
    elif CFG.model_name == 'XGB':
        model = XGB
    elif CFG.model_name == 'Logit':
        model = Logit
    elif CFG.model_name == 'SVM':
        model = SVM
    elif CFG.model_name == 'DT':
        model = DT
    elif CFG.model_name == 'KNN':
        model = KNN
    model.fit(x_train, y_train)

    y_pre_list = model.predict(x_test)
    # y_proba_list = model.predict_proba(x_test)

    ### calculate metrix
    y_real_list = y_test
    y_pre_list = np.where(y_pre_list >= 0.5, 1, y_pre_list)
    y_pre_list = np.where(y_pre_list < 0.5, 0, y_pre_list)

    from sklearn.metrics import roc_auc_score, hamming_loss, multilabel_confusion_matrix, confusion_matrix
    from sklearn.metrics import roc_curve, auc

    GT = y_real_list  # groud truth
    pred = y_pre_list  # predicted
    quantity = label_quantity(GT, pred)  # tp, fp, tn, fn
    acc = label_accuracy_micro(GT, pred)
    pre = label_precision_micro(GT, pred)
    recall = label_recall_micro(GT, pred)
    f1 = label_f1_micro(GT, pred)
    roc_auc = roc_auc_score(GT, pred, average="micro")
    hamming = hamming_loss(GT, pred)

    method1 = []
    method2 = []
    # compare different ROC methods
    for i in range(GT.shape[1]):  # the two methods are same
        method1.append(roc_auc_score(GT[:, i], pred[:, i]))
        fpr, tpr, thresholds = roc_curve(GT[:, i], pred[:, i])
        method2.append(auc(fpr, tpr))

    GT = pd.DataFrame(GT)
    pred = pd.DataFrame(pred)
    filename = "result/ML/" + CFG.model_name + "-" + str(future) + "-prediction.xlsx"  # excel result of each model
    with pd.ExcelWriter(filename) as writer:
        GT.to_excel(writer, sheet_name='Ground_truth')
        pred.to_excel(writer, sheet_name=str("Prediction-" + CFG.model_name + "-" + str(future)))

    performance = np.row_stack((performance, [CFG.model_name, future, acc, pre, recall, f1, roc_auc, hamming]))

pd.DataFrame(performance).to_csv(str("result/ML/" + CFG.model_name + "-performance.csv")
                                 , index=False, header=False)
