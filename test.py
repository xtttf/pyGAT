import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt

def load_data(path="./data/sumo/", dataset="test"):
    l = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    return l

data = load_data()
x = []
y = []
for i in range(len(data)):
    x.append(data[i][0])
    y.append(data[i][1])
plt.plot(x, y)

plt.show()

