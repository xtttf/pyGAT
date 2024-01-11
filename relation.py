from __future__ import division
from __future__ import print_function
import pdb
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import csv
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GAT, SpGAT


#case_index = 2

# Training settings
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

# Load data
#path = "./model/case_sumo_{}/".format(case_index)
path = "./model/test/"
adj, features, labels, idx_train, idx_val, idx_test, origin_labels = load_data(path, "test")

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

#model.load_state_dict(torch.load('model/case_highd_{}/best_epoch.pkl'.format(case_index)))
model.load_state_dict(torch.load(path+'best_epoch.pkl'))
output = model(features, adj)

classes = list(sorted(list(set(origin_labels))))
import pdb; pdb.set_trace()
print(classes)
result = {}
preds = output.max(1)[1].type_as(labels)
for index in range(0, len(output)):
    pred = preds[index]
    real = classes[labels[index]]
    if not result.get(real, None):
        result[real] = [0 for _ in range(0, len(classes)+1)]
    result[real][0] += 1
    result[real][pred+1] += 1

for k, v in result.items():
    for i in range(1, len(v)):
        v[i] = 1.0 * v[i] / v[0]
print(result)
classes_out = ['' ,''] + classes
#f = open('model/case_highd_{}/relation.csv'.format(case_index) ,'w',encoding= 'utf-8',newline= '')
f = open(path+'relation.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(classes_out)
for key, value in result.items():
    result_out = [key] + value
    csv_writer.writerow(result_out)
