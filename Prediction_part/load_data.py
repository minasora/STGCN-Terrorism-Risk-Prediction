import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from pandas import concat


def load_matrix(file_path):
    return pd.read_csv(file_path, header=None).values.astype(float)


def load_data(file_path, len_train, len_val):
    df = pd.read_csv(file_path, header=None).values.astype(float)
    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val:]
    return train, val, test


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def data_transform_our(data, n_his, n_pred, device):
    n_prov = data.shape[1]
    data_frame = np.array(series_to_supervised(data, n_his, n_pred))
    col_len = data_frame.shape[1]
    his_len = int(col_len - n_pred * n_prov)
    x = data_frame[:, :his_len]
    y = data_frame[:, -n_prov:]  # only one time slot in the seq output
    x = x.reshape(x.shape[0], 1, n_his, n_prov)
    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)


def data_transform_our_numpy(data, n_his, n_pred):
    n_prov = data.shape[1]
    data_frame = np.array(series_to_supervised(data, n_his, n_pred))
    col_len = data_frame.shape[1]  # x+y一共多少列
    his_len = int(col_len - n_pred * n_prov)
    x = data_frame[:, :his_len]
    y = data_frame[:, -n_prov:]  # only one time slot in the seq output
    x = x.reshape(x.shape[0], 1, n_his, n_prov)
    return x, y
