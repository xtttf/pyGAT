# coding=utf-8
import pdb

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
path = os.path.dirname(__file__)

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


n = 8 # 前n个的数据作为参考
LR = 0.001  # 模型学习率,为优化器使用
EPOCH = 20
# 数据集建立
attack_type = 'attack5'
data = np.genfromtxt(os.path.join(path, 'gru_data', 'highd', attack_type, 'data_train_select2'))
data = data[:80000]

data_numpy = np.array(data)
data_numpy_mean = np.mean(data_numpy)#求平均值
data_tensor = torch.Tensor(data_numpy)
train_set = TrainSet(data_tensor)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
rnn = RNN(n)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # 优化算法
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


#p为标记为1的数据，t为标记为0的数据（即正确数据）
#tp为1被判为1，fn为1被判为0
#fp为0被判为1，tn为0被判为0
#准确率为(tp+tn)/(tp+fn+fp+tn)
#召回率（将错误数据成功找出）为tp/(tp+fn)即tp/p
#虚警率为fp/(tp+fp)
def accuracy_rate(_output, labels):
    result = []
    for item in _output[0]:
        result.append(item[0])
    _tp, _fn, _fp, _tn = 0, 0, 0, 0

    for index in range(0, len(result)):
        res = get_result(result[index])
        if labels[index] in wrong_result_set:
            if res == labels[index]:
                _tp += 1
            else:
                _fn += 1
        else:
            if res == labels[index]:
                _tn += 1
            else:
                _fp += 1
        
    return _tp, _fn, _fp, _tn, len(result)


max_acc = 0
cnt_correct_rate = 0
cnt_xjl_rate = 0
cnt_loss = 0

for step in range(EPOCH):
    loss = 0
    m = 0
    mm = 0
    tp, fn, fp, tn = 0, 0, 0, 0
    for tx, ty in train_loader:
        output = rnn(torch.unsqueeze(tx, dim=0))
        sub_loss = loss_func(torch.squeeze(output), ty)
        optimizer.zero_grad()  # clear gradients for this training step
        sub_loss.backward()  # back propagation, compute gradients
        optimizer.step()
        sub_tp, sub_fn, sub_fp, sub_tn, sub_m = accuracy_rate(output, ty)

        tp += sub_tp
        fn += sub_fn
        fp += sub_fp
        tn += sub_tn
        loss += sub_loss
        m += 1
        mm += sub_m

    
    loss = loss / m
    acc = (tp+tn)/(tp+fn+fp+tn)
    recall = tp/(tp+fn)#召回率
    falsealarm = fp/(tn+fp)#虚警率

    if acc > max_acc:
        max_acc = acc
        cnt_correct_rate = recall
        cnt_loss = loss
        cnt_xjl_rate = falsealarm
        torch.save(rnn, os.path.join(path, 'gru_data', 'highd', attack_type, 'result_select2.pkl'))
    print(tp, fn, fp, tn, mm)
    print("epoch: {} loss: {} train acc: {} recall: {} falsealarm: {}".format(step, loss, acc, recall, falsealarm))

print("best epoch loss: {} acc: {} recall: {} falsealarm: {}".format(cnt_loss, max_acc, cnt_correct_rate, cnt_xjl_rate))

data = np.genfromtxt(os.path.join(path, 'gru_data', 'highd', attack_type, 'data_test_select2'))
data = data[80000:100000]

data_numpy = np.array(data)
data_numpy_mean = np.mean(data_numpy)
data_tensor = torch.Tensor(data_numpy)
test_set = TrainSet(data_tensor)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

tot_num = 0
tot_loss = 0
tot_tp, tot_fn, tot_fp, tot_tn = 0, 0, 0, 0
tot_n = 0

for tx, ty in test_loader:
    output = rnn(torch.unsqueeze(tx, dim=0))
    loss = loss_func(torch.squeeze(output), ty)
    tot_loss += loss
    sub_tp, sub_fn, sub_fp, sub_tn, sub_m = accuracy_rate(output, ty)
    tot_tp += sub_tp
    tot_fn += sub_fn
    tot_fp += sub_fp
    tot_tn += sub_tn
    tot_num += 1
    tot_n += sub_m
tot_acc = (tot_tp+tot_tn)/(tot_tp+tot_fn+tot_fp+tot_tn)
tot_recall = tot_tp/(tot_tp+tot_fn)#召回率
tot_falsealarm = tot_fp/(tot_tn+tot_fp)#虚警率
    

print(tot_tp, tot_fn, tot_fp, tot_tn, tot_n)
print("test loss: {} acc: {} recall: {} falsealarm: {}".format(tot_loss / tot_num, tot_acc, tot_recall, tot_falsealarm))

