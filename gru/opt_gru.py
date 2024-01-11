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


n = 6  # 前n个的数据作为参考
LR = 0.0001  # 模型学习率,为优化器使用
EPOCH = 200
# 数据集建立
# data = np.genfromtxt("opt_dataset")
#
# data_numpy = np.array(data)
# data_numpy_mean = np.mean(data_numpy)
# data_tensor = torch.Tensor(data_numpy)
# train_set = TrainSet(data_tensor)
# train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
# rnn = RNN(n)
# optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # 优化算法
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


def accuracy_rate(_output, labels, _tx):
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
        elif labels[index] == 1.0:
            print("{}\t{}".format(float(tx[index][2]), float(tx[index][6])))
    return 1.0 * _acc / len(result), _wrong, _correct


# max_acc = 0
# cnt_correct_rate = 0
# cnt_loss = 0
#
# for step in range(EPOCH):
#     loss = 0
#     acc = 0
#     m = 0
#     wrong, correct = 0, 0
#     for tx, ty in train_loader:
#         output = rnn(torch.unsqueeze(tx, dim=0))
#         sub_loss = loss_func(torch.squeeze(output), ty)
#         optimizer.zero_grad()  # clear gradients for this training step
#         sub_loss.backward()  # back propagation, compute gradients
#         optimizer.step()
#         sub_acc, sub_wrong, sub_correct = accuracy_rate(output, ty)
#         acc += sub_acc
#         wrong += sub_wrong
#         correct += sub_correct
#         loss += sub_loss
#         m += 1
#     loss = loss / m
#     acc = acc / m
#     correct_rate = 1.0 * correct / wrong
#     if acc > max_acc:
#         max_acc = acc
#         cnt_correct_rate = correct_rate
#         cnt_loss = loss
#         torch.save(rnn, 'gru_1.pkl')
#     print("epoch: {} loss: {} train acc: {} correct rate: {}".format(step, loss, acc, correct_rate))
#
# print("best epoch loss: {} acc: {} correct rate: {}".format(cnt_loss, max_acc, correct_rate))

rnn = torch.load('gru_3.pkl')
data = np.genfromtxt("opt_test")
for index in range(len(data)):
    cnt = data[index]
    tmp = cnt[7]
    cnt[7] = cnt[6]
    cnt[6] = tmp
    data[index] = cnt

data_numpy = np.array(data)
data_tensor = torch.Tensor(data_numpy)
test_set = TrainSet(data_tensor)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

tot_num = 0
tot_loss = 0
tot_acc, tot_wrong, tot_correct = 0, 0, 0
base = 0

for tx, ty in test_loader:
    output = rnn(torch.unsqueeze(tx[:, [0, 1, 2, 3, 4, 5]], dim=0))
    loss = loss_func(torch.squeeze(output), ty)
    tot_loss += loss
    sub_acc, sub_wrong, sub_correct = accuracy_rate(output, ty, tx)
    tot_num += 1
    tot_acc += sub_acc
    tot_wrong += sub_wrong
    tot_correct += sub_correct
    base += len(ty)

print("{}\t{}".format(tot_acc / tot_num, 1.0 * tot_correct / tot_wrong))
