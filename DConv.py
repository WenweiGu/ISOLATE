import torch.nn as nn
from torch.nn.utils import weight_norm


# 裁剪模块，裁剪掉多余的padding
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# 相当于一个残差网络模块
class DCConv(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, padding, dilation, dropout=0.2):
        super(DCConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, self.kernel_size,
                                           stride=self.stride, padding=self.padding, dilation=self.dilation))
        self.chomp1 = Chomp1d(self.padding)
        self.bn_1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, self.kernel_size,
                                           stride=self.stride, padding=self.padding, dilation=self.dilation))
        self.chomp2 = Chomp1d(self.padding)
        self.bn_2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn_1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn_2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
