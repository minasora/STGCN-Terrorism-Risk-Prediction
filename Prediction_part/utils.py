import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from ..utils import _log_api_usage_once

epsilon = np.finfo('float').eps


# version 1: use torch.autograd
class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean', ):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        '''
        # compute loss
        logits = logits.float()  # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)  # 沿列计算，即计算矩阵每一行总和，计算矩阵的度
    L = np.diag(d) - A  # np.diag() 将度d转化为沿对角线的度矩阵
    # L = L.astype(float) 一般不需要转化，因为L必定是float型，如果是int就要转换
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])  # scaled_laplacian: L=2*L/eigval_max-I_n, in cheb_poly
    lam = np.linalg.eigvals(L).max().real  # 求其特征根，及最大的特征根
    return 2 * L / lam - np.eye(n)  ###np.eye(n) 生成I_n


def cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]  ###LL：两个Array元素的list
    for i in range(2, Ks):  ###轮子在这默认cheb_poly k>2 值得商榷
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)


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


def label_quantity(gt, predict):
    tp = np.sum(np.logical_and(gt, predict), axis=0)
    fp = np.sum(np.logical_and(1 - gt, predict), axis=0)
    tn = np.sum(np.logical_and(1 - gt, 1 - predict), axis=0)
    fn = np.sum(np.logical_and(gt, 1 - predict), axis=0)
    return np.stack([tp, fp, tn, fn], axis=0).astype("float")


def label_accuracy_macro(gt, predict):
    quantity = label_quantity(gt, predict)
    tp_tn = np.add(quantity[0], quantity[2])  # 不合并，位相加，仍是1*34
    tp_fp_tn_fn = np.sum(quantity, axis=0)
    return np.mean(tp_tn / (tp_fp_tn_fn + epsilon))


def label_accuracy_micro(gt, predict):
    quantity = label_quantity(gt, predict)
    sum_tp, sum_fp, sum_tn, sum_fn = np.sum(quantity, axis=1)  # 所有样本相加
    return (sum_tp + sum_tn) / (
            sum_tp + sum_fp + sum_tn + sum_fn + epsilon)


def label_precision_macro(gt, predict):
    quantity = label_quantity(gt, predict)
    tp = quantity[0]
    tp_fp = np.add(quantity[0], quantity[1])
    return np.mean(tp / (tp_fp + epsilon))


def label_precision_micro(gt, predict):
    quantity = label_quantity(gt, predict)
    sum_tp, sum_fp, sum_tn, sum_fn = np.sum(quantity, axis=1)
    return sum_tp / (sum_tp + sum_fp + epsilon)


def label_recall_macro(gt, predict):
    quantity = label_quantity(gt, predict)
    tp = quantity[0]
    tp_fn = np.add(quantity[0], quantity[3])
    return np.mean(tp / (tp_fn + epsilon))


def label_recall_micro(gt, predict):
    quantity = label_quantity(gt, predict)
    sum_tp, sum_fp, sum_tn, sum_fn = np.sum(quantity, axis=1)
    return sum_tp / (sum_tp + sum_fn + epsilon)


def label_f1_macro(gt, predict, beta=1):
    quantity = label_quantity(gt, predict)
    tp = quantity[0]
    fp = quantity[1]
    fn = quantity[3]
    return np.mean((1 + beta ** 2) * tp / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + epsilon))


def label_f1_micro(gt, predict, beta=1):
    quantity = label_quantity(gt, predict)
    tp = np.sum(quantity[0])
    fp = np.sum(quantity[1])
    fn = np.sum(quantity[3])
    return (1 + beta ** 2) * tp / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + epsilon)
