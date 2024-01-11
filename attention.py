from models import GAT
from utils import load_data, accuracy
import random
import numpy as np
import argparse
import torch
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models import GAT, SpGAT
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=20, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def prepare_attentional_mechanism_input(a, out_feature, Wh):
    # Wh.shape (N, out_feature)
    # self.a.shape (2 * out_feature, 1)
    # Wh1&2.shape (N, 1)
    # e.shape (N, N)
    Wh1 = torch.matmul(Wh, a[:out_feature, :])
    Wh2 = torch.matmul(Wh, a[out_feature:, :])
    # broadcast add
    e = Wh1 + Wh2.T
    leakyrelu = nn.LeakyReLU(args.alpha)
    return leakyrelu(e)



def GraphAttentionLayer(h, adj, W, a, out_f):
    Wh = torch.mm(h, W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
    e = prepare_attentional_mechanism_input(a, out_f, Wh)
    zero_vec = -9e15*torch.ones_like(e)#将没有连接的边置为负无穷
    attention = torch.where(adj > 0, e, zero_vec)
    attention = F.softmax(attention, dim=1)#得到归一化的权重系数
    #attention = F.dropout(attention, self.dropout, training=self.training)
    attention = F.dropout(attention, args.dropout)
    #print(attention[0][:31])
    h_prime = torch.matmul(attention, Wh)
    return F.elu(h_prime), attention

def attention_res():

    adj, features, labels, idx_train, idx_val, idx_test, origin_labels = load_data()

    if args.sparse:
        model = SpGAT(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    nheads=args.nb_heads,
                    alpha=args.alpha)
    else:
        model = GAT(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    nheads=args.nb_heads,
                    alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    features, adj, labels = Variable(features), Variable(adj), Variable(labels)

    feature = F.dropout(features, args.dropout)





    #获取epoch中的参数
    data = torch.load("./model/test/best_epoch.pkl")
    count = 1
    for k, v in data.items():
        if count%2 == 1:
            WW = v
        if count%2 == 0:
            aa = v
        if k == 'out_att.a':
            break
        if count == 2:
            res, att = GraphAttentionLayer(feature, adj, WW, aa, args.hidden)
            count += 1
            continue
        if count%2 == 0 and count != 2:
            x, att = GraphAttentionLayer(feature, adj, WW, aa, args.hidden)
            res = torch.cat([res, x], dim=1)
        count += 1
    res = F.dropout(res, args.dropout)
    x, att = GraphAttentionLayer(res, adj, WW, aa, int(labels.max()) + 1)
    #特征种类，后续可进行更改
    feature_kind = 30
    count = 1
    res = [0 for _ in range(feature_kind+1)]
    #计算attention中含0的数量
    co = [0 for _ in range(feature_kind+1)]
    while feature_kind*count < len(att):
        s = feature_kind*(count-1)
        temp = att[s][s:s+feature_kind+1]
        t = temp.tolist()
        for i in range(len(t)):
            res[i] = res[i]+t[i]
            if t[i] == 0:
                co[i] += 1
        count += 1
    ss = sum(res)
    ss = ss/len(res)
    for i in range(len(res)):
        if res[i] < ss:
            res[i] = 0
    print(res)

    return res






