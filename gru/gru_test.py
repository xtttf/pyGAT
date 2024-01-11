# coding=utf-8
import pdb

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class RNN(nn.Module):

    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Linear(64, 1),
        )
        self.hidden = None

    def forward(self, x):
        r_out, self.hidden = self.rnn(x)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out)
        return out


class TrainSet(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


n = 4  # 前n个的数据作为参考
# 数据集建立

rnn = torch.load('gru_2.pkl')

loss_func = nn.MSELoss()  # 损失函数
result_set = [0, 1]
wrong_result_set = [1]


def get_result(num):
    res = result_set[0]
    df = abs(num - result_set[0])
    for item in result_set:
        if abs(num - item) < df:
            df = abs(num - item)
            res = item
    return res


def accuracy_rate(_output, labels):
    result = []
    for item in _output[0]:
        result.append(item[0])
    _acc = 0
    _wrong, _correct = 0, 0
    for index in range(0, len(result)):
        res = get_result(result[index])
        if labels[index] in wrong_result_set:
            _wrong += 1
        if res == labels[index]:
            if res in wrong_result_set:
                _correct += 1
            _acc += 1
    return 1.0 * _acc / len(result), _wrong, _correct


data = np.genfromtxt("test")

data_numpy = np.array(data[:, [0, 4, 6, 20, 23]])
data_numpy_mean = np.mean(data_numpy)
data_tensor = torch.Tensor(data_numpy)
test_set = TrainSet(data_tensor)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

tot_num = 0
tot_loss = 0
tot_acc, tot_wrong, tot_correct = 0, 0, 0

for tx, ty in test_loader:
    output = rnn(torch.unsqueeze(tx, dim=0))
    loss = loss_func(torch.squeeze(output), ty)
    tot_loss += loss
    sub_acc, sub_wrong, sub_correct = accuracy_rate(output, ty)
    tot_num += 1
    tot_acc += sub_acc
    tot_wrong += sub_wrong
    tot_correct += sub_correct

print("{}\t{}".format(tot_acc / tot_num, 1.0 * tot_correct / tot_wrong))
