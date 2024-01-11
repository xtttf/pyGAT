#LSTM

import os
import pdb

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding, LSTM, GRU

from tensorflow.keras import datasets
from tensorflow.keras.preprocessing import sequence
from keras.utils import np_utils

path = os.getcwd()
filename = os.path.join(path, 'gru', 'gru_data', 'sparse', 'all')

col = []
#此处需要修改参数为特征值
n_feature = 30
for i in range(n_feature):
    col.append(i)
col = tuple(col)
x_train = np.genfromtxt(os.path.join(filename, 'train'), usecols=col)
y_train = np.genfromtxt(os.path.join(filename, 'train'), usecols=(n_feature,))
x_test = np.genfromtxt(os.path.join(filename, 'test'), usecols=col)
y_test = np.genfromtxt(os.path.join(filename, 'test'), usecols=(n_feature,))

x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
#用于分类，将标签转化为one-hot编码
#y_train = np_utils.to_categorical(y_train)
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
#y_test = np_utils.to_categorical(y_test)
#import pdb; pdb.set_trace()
'''
max_features = 10000 # 我们只考虑最常用的10k词汇
maxlen = 500 # 每个评论我们只考虑100个单词
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen) #长了就截断，短了就补0
'''
model = Sequential()
model.add(LSTM(units=50, input_shape=(x_train.shape[1], x_train.shape[2])))
#model.add(GRU(units=50, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.2
)


loss, acc = model.evaluate(x_test, y_test)
print(loss, acc)

