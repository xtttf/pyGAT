# -*- coding: utf-8 -*-
import numpy as np
list_data = []
eps = 1e-6

with open('./dataset', 'r') as f:
    line = f.readline()
    while line:
        line = line.replace('\n', '').replace('(', '').replace(')', '').replace(' ', '').replace('[','').replace(']','')
        mid = line.split(',')
        for i in range(len(mid)):
            mid[i] = float(mid[i])
        list_data.append(mid)
        line = f.readline()

data = []
for item in list_data:
    cnt = [int(item[0])]
    for index in range(1, len(item)-1, 2):
        cnt.append([item[index], int(item[index+1])])
    data.append(cnt)

tot, correct, wrong, find = 0, 0, 0, 0

for cnt in data:
    n = int(cnt[0])
    tot += n
    cnt = cnt[1:]
    for item in cnt[:2]:
        if item[1] == 0:
            correct += 1
        else:
            wrong += 1
    if n < 3:
        continue
    k = cnt[1][0] - cnt[0][0]
    for index in range(2, n):
        pred = cnt[index-1][0] + k
        if 0.85 * pred < cnt[index][0] < 1.15 * pred:
            if cnt[index][1] == 0:
                correct += 1
            else:
                wrong += 1
            k = cnt[index][0] - cnt[index-1][0]
        else:
            if cnt[index][1] == 1:
                correct += 1
                find += 1
                wrong += 1
print("{}\t{}".format(1.0 * correct / tot, 1.0 * find / wrong))










