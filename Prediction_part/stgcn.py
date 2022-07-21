import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x


class temporal_conv_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(temporal_conv_layer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        #    self.lstm = lstm_layer(batch_size, c_in, n_his)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1),
                                  1)  # in_channel, out_channel, kenerl size, stride, 可以看到kenerl size第二项为1，说明是单节点卷积
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)


class spatio_conv_layer(nn.Module):
    def __init__(self, ks, c, Lk):
        super(spatio_conv_layer, self).__init__()
        self.Lk = Lk
        self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        return torch.relu(x_gc + x)


class st_conv_block(nn.Module):
    def __init__(self, ks, kt, n, c, p, Lk):  # c : bs， bottleneck结构
        super(st_conv_block, self).__init__()

        self.tconv1 = temporal_conv_layer(kt, c[0], c[1],
                                          "GLU")  # c[0]=1. 为论文里Ci，也就是vertice value的维度，经过第一个1D-CNN卷积，变为32维
        self.sconv = spatio_conv_layer(ks, c[1], Lk)
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_s = self.sconv(x_t1)
        x_t2 = self.tconv2(x_s)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)


class fully_conv_layer(nn.Module):
    def __init__(self, c):
        super(fully_conv_layer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)  # in_channel, out_channel, kenerl size

    def forward(self, x):
        return self.conv(x)


class output_layer(nn.Module):
    def __init__(self, c, T, n):  # 128*22*34 c=128, 1D-CNN通道数，T：22，缩小的时间序列长度，n：34，不变的节点数
        super(output_layer, self).__init__()
        self.tconv1 = temporal_conv_layer(T, c, c,
                                          "GLU")  # 和STGCN主干不一样，此时不再变化通道数量，1D-CNN卷积宽度也为T，等于直接对单个节点每个通道的所有步长缩小为1维
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = temporal_conv_layer(1, c, c, "sigmoid")
        self.fc = fully_conv_layer(c)  # 维度变化为c→1

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2)


class STGCN(nn.Module):
    def __init__(self, ks, kt, bs, T, n, Lk, p):  # n : n_vertice = 34
        super(STGCN, self).__init__()
        self.st_conv1 = st_conv_block(ks, kt, n, bs[0], p, Lk)
        self.st_conv2 = st_conv_block(ks, kt, n, bs[1], p, Lk)
        self.output = output_layer(bs[1][2], T - 4 * (kt - 1),
                                   n)  # 128*22*34， bs[1][2]=128，对应为bottleneck通道数、T经过卷积减少的宽度、节点数

    def forward(self, x):
        x_st1 = self.st_conv1(x)
        x_st2 = self.st_conv2(x_st1)
        return self.output(x_st2)


class FNN(nn.Module):
    def __init__(self, n_his, n_vertice):
        super(FNN, self).__init__()
        self.linear1 = nn.Linear(n_his * n_vertice, 128)
        self.linear2 = nn.Linear(128, n_vertice)

    def forward(self, x):
        size = (x.shape[0], x.shape[-1] * x.shape[-2] * x.shape[-3])
        x = x.reshape(size)
        x_st1 = self.linear1(x)
        x_st2 = self.linear2(x_st1)
        return x_st2.view(x_st2.shape[0], 1, 1, x_st2.shape[-1])


class lstm_layer(nn.Module):
    def __init__(self, batch_size, n_vertice, n_his):
        super(lstm_layer, self).__init__()
        self.batch_size = batch_size
        self.n_vertice = n_vertice
        self.n_his = n_his
        # self.LSTM1 = nn.LSTM(n_his, n_vertice)
        self.LSTM1 = nn.LSTM(input_size=n_vertice, hidden_size=n_vertice)  # input_size hidden_size num_layers

    def forward(self, x):
        k = x.shape
        x_gc = x
        batch_size = x.shape[0]
        n_vertice = self.n_vertice
        n_his = self.n_his
        x1 = x.view((n_his, batch_size, n_vertice))  # (seq, batch, feature)
        x2 = self.LSTM1(x1)[0]  # outshape: # (seq, batch, hidden_size)
        x2 = x2.view((k[0], 1, -1, k[-1]))
        return torch.relu(x_gc + x2)
        # return x2


class STGCN_LSTM(nn.Module):
    def __init__(self, ks, kt, bs, T, n, Lk, p, batch_size, n_vertice, n_his):
        super(STGCN_LSTM, self).__init__()
        self.st_LSTM = lstm_layer(batch_size, n_vertice, n_his)
        self.st_STGCN = STGCN(ks, kt, bs, T, n, Lk, p)

    def forward(self, x):
        # k = x.shape
        x = self.st_LSTM(x)
        # x = x.view((k[0],1,-1,k[-1]))
        return self.st_STGCN(x)


class multihead_attention(nn.Module):
    # mm = multihead_attention(batch_size=x.shape[0], n_vertice=34).to(device)
    # yy = mm(x)
    # embed_dim will be split across num_heads (i.e. each head will have dimension embed_dim // num_heads)
    def __init__(self, batch_size, n_vertice):
        super(multihead_attention, self).__init__()
        self.batch_size = batch_size
        self.embed_dim = n_vertice  # embed_dim
        self.num_heads = 2  # num_heads
        self.mulatt = nn.MultiheadAttention(embed_dim=self.embed_dim,
                                            num_heads=self.num_heads)

    def forward(self, x):
        k = x.shape
        x_gc = x
        batch_size = x.shape[0]
        n_vertice = self.embed_dim
        n_his = x.shape[2]
        x1 = x.view((n_his, batch_size, n_vertice))  # (seq, batch, feature)
        # self-attention Q=K=V, outshape: seq * batchsize * embed_dim(vertice)
        x2 = self.mulatt(x1, x1, x1, need_weights=False)[0]
        # reshape to 4-Dimension shape
        x2 = x2.view((k[0], 1, -1, k[-1]))
        return torch.relu(x_gc + x2)
        # return x2


class STGCN_Attention(nn.Module):
    def __init__(self, ks, kt, bs, T, n, Lk, p, batch_size, n_vertice, n_his):
        super(STGCN_Attention, self).__init__()
        self.mul_attn = multihead_attention(batch_size, n_vertice)
        self.st_STGCN = STGCN(ks, kt, bs, T, n, Lk, p)

    def forward(self, x):
        x = self.mul_attn(x)
        return self.st_STGCN(x)


class lstm_sa_layer(nn.Module):
    def __init__(self, batch_size, n_vertice):
        super(lstm_sa_layer, self).__init__()
        self.batch_size = batch_size
        self.embed_dim = n_vertice  # embed_dim
        self.num_heads = 2  # num_heads
        self.mulatt = nn.MultiheadAttention(embed_dim=self.embed_dim,
                                            num_heads=self.num_heads)
        self.LSTM1 = nn.LSTM(input_size=n_vertice, hidden_size=128)
        self.LSTM2 = nn.LSTM(input_size=128, hidden_size=n_vertice)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        k = x.shape
        x_gc = x
        batch_size = x.shape[0]
        n_vertice = self.embed_dim
        n_his = x.shape[2]
        x1 = x.view((n_his, batch_size, n_vertice))  # (seq, batch, feature)
        # Res: Input + LSTM
        x2 = self.LSTM1(x1)[0]
        x2 = self.LSTM2(x2)[0]
        x2 = x2.view((k[0], 1, -1, k[-1]))
        x2 = torch.relu(x_gc + x2)
        x2 = self.norm(x2)
        # Res: Input+SA self-attention Q=K=V, outshape: seq * batchsize * embed_dim(vertice)
        x3 = x2.view((n_his, batch_size, n_vertice))  # (seq, batch, feature)
        x3 = self.mulatt(x3, x3, x3, need_weights=False)[0]
        x3 = x3.view((k[0], 1, -1, k[-1]))
        x3 = torch.relu(x2 + x3)
        x3 = self.norm(x3)
        return self.dropout(x3)


class STGCN_LSTM_SA(nn.Module):
    def __init__(self, ks, kt, bs, T, n, Lk, p, batch_size, n_vertice, n_his):
        super(STGCN_LSTM_SA, self).__init__()
        self.lstm_sa = lstm_sa_layer(batch_size, n_vertice)
        self.st_STGCN = STGCN(ks, kt, bs, T, n, Lk, p)

    def forward(self, x):
        x = self.lstm_sa(x)
        return self.st_STGCN(x)


class LSTM(nn.Module):
    def __init__(self, batch_size, n_vertice, n_his):
        super(LSTM, self).__init__()
        self.batch_size = batch_size
        self.n_vertice = n_vertice
        self.n_his = n_his
        self.LSTM1 = nn.LSTM(n_vertice, 128)
        self.LSTM2 = nn.LSTM(128, n_vertice)

    def forward(self, x):
        batch_size = x.shape[0]
        n_vertice = self.n_vertice
        n_his = self.n_his
        x = x.view((n_his, batch_size, n_vertice))
        x = self.LSTM1(x)[0]
        x = self.LSTM2(x)[0]
        return x[-1, :, :]


class GRU(nn.Module):
    def __init__(self, batch_size, n_vertice, n_his):
        super(GRU, self).__init__()
        self.batch_size = batch_size
        self.n_vertice = n_vertice
        self.n_his = n_his
        self.GRU1 = nn.GRU(n_vertice, 128)
        self.GRU2 = nn.GRU(128, n_vertice)

    def forward(self, x):
        batch_size = x.shape[0]
        n_vertice = self.n_vertice
        n_his = self.n_his
        x = x.view((n_his, batch_size, n_vertice))
        x1 = self.GRU1(x)[0]
        x2 = self.GRU2(x1)[0]
        return x2[-1, :, :]


class single_layer_STGCN(nn.Module):
    def __init__(self, ks, kt, bs, T, n, Lk, p):
        super(single_layer_STGCN, self).__init__()
        self.st_conv1 = st_conv_block(ks, kt, n, bs, p, Lk)
        self.output = output_layer(bs[2], T - 2 * (kt - 1), n)

    def forward(self, x):
        x_st1 = self.st_conv1(x)
        return self.output(x_st1)


class STGCN_LSTM_OUT(nn.Module):  # 不要STGCN自己的输出层，改为LSTM输出
    def __init__(self, ks, kt, bs, T, n, Lk, p):
        super(STGCN_LSTM_OUT, self).__init__()
        self.st_conv1 = st_conv_block(ks, kt, n, bs[0], p, Lk)
        self.st_conv2 = st_conv_block(ks, kt, n, bs[1], p, Lk)
        self.tconv = temporal_conv_layer(1, bs[1][-1], 1, "sigmoid")  # kt, Cin, Cout
        self.LSTM_out1 = nn.LSTM(input_size=n, hidden_size=128)
        self.LSTM_out2 = nn.LSTM(input_size=128, hidden_size=n)
        self.linear1 = nn.Linear(n, n)

    def forward(self, x):
        # k = x.shape #batchsize * 1(channel of Vertice value) * n_his * n_vertice
        x_st1 = self.st_conv1(x)
        x_st2 = self.st_conv2(x_st1)
        x_t = self.tconv(x_st2)
        k1 = x_t.shape
        x_t = x_t.view((k1[2], k1[0], k1[3]))  # (seq, batch, feature)
        x_res = x_t
        x_lstm1 = self.LSTM_out1(x_t)[0]
        x_lstm2 = self.LSTM_out2(x_lstm1)[0]
        x_lstm2 = x_lstm2[-1, :, :]
        x_lstm2 = torch.relu(x_res + x_lstm2)
        x_out = self.linear1(x_lstm2)
        return x_out
