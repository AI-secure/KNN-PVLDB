#!/usr/bin/python

import tensorflow as tf
import numpy as np
from Shapley import ShapNN
from DShap import DShap
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import argparse
import copy
import random

parser = argparse.ArgumentParser(description = None)
parser.add_argument('--num', type=int, required = True)
args = parser.parse_args()

x = args.num

fashion = input_data.read_data_sets("fashion_data/", one_hot=True)  

X_data = []
y_data = []

for _x, _y in zip(fashion.train.images, np.argmax(fashion.train.labels, axis=1)):
    if _y == 0:
        X_data.append(_x)
        y_data.append(0)
    elif _y == 6:
        X_data.append(_x)
        y_data.append(1)
X_data = np.array(X_data)
y_data = np.array(y_data)

X_test_data = X_data[x:x+x//10]
y_test_data = y_data[x:x+x//10]
X_data = X_data[0:x]
y_data = y_data[0:x]
y_data_orig = copy.deepcopy(y_data)

X_benign = []
y_benign = []

X_flip = []
y_flip = []

flip = np.zeros(x)
for i in range(x // 10):
    j = np.random.randint(0, x)
    while flip[j] == 1:
        j = np.random.randint(0, x)
    flip[j] = 1
    y_data[j] = 1 - y_data[j]
    X_flip.append(X_data[j])
    y_flip.append(y_data[j])
for i in range(x):
    if flip[i] == 0:
        X_benign.append(X_data[i])
        y_benign.append(y_data[i])
pickle.dump(flip, open('flip.pkl', 'wb'))

# dshap = DShap(X=X_data,
#               y=y_data_orig,
#               X_test=X_test_data,
#               y_test=y_test_data,
#               num_test=x//10,
#               model_family='NN',
#               nodump=True)
# dshap.model.fit(X_data, y_data_orig)
# print("Original model training accuracy for benign data: %g" % dshap.model.score(X_benign, y_benign))
# dshap = DShap(X=X_data,
#               y=y_data,
#               X_test=X_test_data,
#               y_test=y_test_data,
#               num_test=x//10,
#               model_family='NN',
#               nodump=True)
# dshap.model.fit(X_data, y_data)
# print("Modified model training accuracy for benign data: %g" % dshap.model.score(X_benign, y_benign))
# print("Modified model training accuracy for flipped data: %g" % dshap.model.score(X_flip, y_flip))

dshap = DShap(X=X_data,
              y=y_data,
              X_test=X_test_data,
              y_test=y_test_data,
              num_test=x//10,
              model_family='NN')
dshap.run(save_every=10, err = 0.5)
print(y_data - dshap.y)