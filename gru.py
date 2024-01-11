# coding=utf-8
import pdb

import torch
import torch.nn as nn
import numpy as np
import time
from torch.nn.utils.rnn import pack_padded_sequence
import math

# 1：数据集

# 超参数
HIDDEN_SIZE = 100  # 隐藏层
BATCH_SIZE = 256
N_LAYER = 2  # RNN的层数
N_EPOCHS = 100  # train的轮数
N_CHARS = 65536  # 这个就是要构造的字典的长度
USE_GPU = False
Base = 1000


def get_dataset():
    filename = "dataset"
    return np.genfromtxt("./data/{}".format(filename), dtype=np.dtype(float))


def create_tensor(tensor):  # 是否使用GPU
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor


dataset = get_dataset()
n = len(dataset)
train_set = dataset[:int(n*0.8)]  # train数据
test_set = dataset[int(n*0.8):]  # test数据
N_COUNTRY = 3


# 2：构造模型
class RNNClassifier(nn.Module):
    """
    这里的bidirectional就是GRU是不是双向的，双向的意思就是既考虑过去的影响，也考虑未来的影响（如一个句子）
    具体而言：正向hf_n=w[hf_{n-1}, x_n]^T,反向hb_0,最后的h_n=[hb_0, hf_n],方括号里的逗号表示concat。
    """

    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1  # 双向2、单向1

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,  # 输入维度、输出维度、层数、bidirectional用来说明是单向还是双向
                          bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.n_directions, output_size)

    def __init__hidden(self, batch_size):  # 工具函数，作用是创建初始的隐藏层h0
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return create_tensor(hidden)  # 加载GPU

    def forward(self, input, seq_lengths):
        # input shape:B * S -> S * B
        input = input.t()
        batch_size = input.size(1)

        hidden = self.__init__hidden(batch_size)  # 隐藏层h0
        pdb.set_trace()
        embedding = self.embedding(input)

        # pack them up
        gru_input = pack_padded_sequence(embedding, seq_lengths)  # 填充了可能有很多的0，所以为了提速，将每个序列以及序列的长度给出

        output, hidden = self.gru(gru_input, hidden)  # 只需要hidden
        if self.n_directions == 2:  # 双向的，则需要拼接起来
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]  # 单向的，则不用处理
        fc_output = self.fc(hidden_cat)  # 最后来个全连接层,确保层想要的维度（类别数）
        return fc_output


def make_tensors(items):
    inputs, seq_lengths, target = [], [], []
    for item in items:
        cnt = []
        for i in range(0, len(item)-1):
            feature = int(item[i]*Base)
            cnt.append(feature)
        inputs.append(tuple(cnt))
        seq_lengths.append(len(item)-1)
        target.append(int(item[len(item)-1]))
    inputs = np.array(inputs)
    seq_lengths = np.array(seq_lengths)
    target = np.array(target)
    return torch.from_numpy(inputs), torch.from_numpy(seq_lengths), torch.from_numpy(target)


# 4：训练和测试模型
def train_model():
    total_loss = 0
    for i in range(0, len(train_set), BATCH_SIZE):
        inputs, seq_lengths, target = make_tensors(train_set[i:min(i+BATCH_SIZE, len(train_set))])
        output = classifier(inputs, seq_lengths)  # 预测输出
        loss = criterion(output, target)  # 求出损失
        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 梯度反传
        optimizer.step()  # 更新参数

        total_loss += loss.item()
        if i % 10 == 0:

            print('[{}] Epoch {}'.format(time_since(start), epoch))
            print('[{}/{}]'.format(i * len(inputs), len(train_set)))
            print('loss={}'.format(total_loss / (i * len(inputs))))

    return total_loss


def test_model():
    correct = 0
    total = len(test_set)
    print("evaluating trained model...")
    with torch.no_grad():
        for i in range(0, len(test_set), BATCH_SIZE):
            inputs, seq_lengths, target = make_tensors(test_set[i:min(i+BATCH_SIZE, len(test_set))])  # 将名字的字符串转换成数字表示
            output = classifier(inputs, seq_lengths)  # 预测输出
            pred = output.max(dim=1, keepdim=True)[1]  # 预测出来是个向量，里面的值相当于概率，取最大的
            correct += pred.eq(target.view_as(pred)).sum().item()  # 预测和实际标签相同则正确率加1

        percent = '%.2f' % (100 * correct / total)
        print('Test set:Accuracy{} / {} {}%'.format(correct, total, percent))

    return correct / total


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


if __name__ == "__main__":
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)  # 定义模型
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)

    # 第三步：定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()  # 分类问题使用交叉熵损失函数
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)  # 使用了随机梯度下降法

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        # Train cycle
        train_model()
        acc = test_model()
        acc_list.append(acc)  # 存入列表，后面画图使用

    # 画图
    epoch = np.arange(1, len(acc_list) + 1, 1)  # 步长为1
    acc_list = np.array(acc_list)



