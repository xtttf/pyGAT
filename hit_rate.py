from __future__ import division
from __future__ import print_function
import pdb
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GAT, SpGAT

case_index = 2

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
path = "./model/case_{}/".format(case_index)
adj, features, labels, idx_train, idx_val, idx_test, origin_labels = load_data(path, "xtf")

# Model and optimizer
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

model.load_state_dict(torch.load('model/case_{}/best_epoch.pkl'.format(case_index)))

output = model(features, adj)
classes = list(sorted(list(set(origin_labels))))
acc_test = accuracy(output, labels)
print(acc_test.data.item())

hit = {}
tot = {}
preds = output.max(1)[1].type_as(labels)
for index in range(0, len(output)):
    pred = classes[preds[index]]
    real = classes[labels[index]]
    tot[real] = tot.get(real, 0) + 1
    if real == pred:
        hit[real] = hit.get(real, 0) + 1

all_hit = 0
res = {}
for k, v in tot.items():
    _hit = hit.get(k, 0)
    all_hit += _hit
    res[k] = 1.0 * _hit / v

print(1.0 * all_hit / len(output))
print(res)
