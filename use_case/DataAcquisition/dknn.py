import os
import sys
import time
import numpy as np
import sklearn
from utils import *
from Dknn import *
from plot import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, tqdm_notebook

batch_size = 32
data = MNIST(one_hot=False)
device = torch.device('cuda')

#cnn = CNN().to(device)
#optimizer = optim.Adam(cnn.parameters())
#criterion = nn.CrossEntropyLoss()
print('---1. load data---')
x_train = torch.from_numpy(data.x_train).view(-1, 28, 28).unsqueeze(1).unsqueeze(1)
y_train = torch.from_numpy(data.y_train).view(-1,1).long()

x_test = torch.from_numpy(data.x_test).view(-1, 28, 28).unsqueeze(1).unsqueeze(1)
y_test = torch.from_numpy(data.y_test).view(-1,1).long()

#train(cnn, device, x_train, y_train, optimizer, criterion, 1, len(data.x_train) // 5)

#accuracy, avg_loss = evaluate(cnn, device, x_train, y_train, criterion)
#print(f'[Train] Accuracy: {100 * accuracy:5.2f}%, loss: {avg_loss:7.4f}')
#accuracy, avg_loss = evaluate(cnn, device, x_test, y_test, criterion)
#print(f'[Test] Accuracy: {100 * accuracy:5.2f}%, loss: {avg_loss:7.4f}')
print('---2. build cnn model and calculate deep features---')
deep_feats = []
targets = []

cnn = CNN().to(device)
optimizer = optim.Adam(cnn.parameters())
criterion = nn.CrossEntropyLoss()

for i, (X, y) in tqdm_notebook(enumerate(zip(x_train, y_train)), total = len(x_train)):
    X = X.to(device)
    deep_feat, y_pre = cnn(X)
    deep_feats.append(deep_feat.view(deep_feat.size(0), -1).cpu().detach().numpy())
    targets.append(y.numpy())
deep_feats = np.concatenate(deep_feats) # deep features are not normalized
targets = np.concatenate(targets)
print(deep_feats[:2])
print(deep_feats.shape, targets.shape)  

print('---3. calculate knn shapley---')
train_size = 1000
k = 4
knn_values = [[] for _ in range(k)]
sx_train, sy_train = x_train[:train_size], y_train[:train_size]
sx_test, sy_test = x_test[-train_size:], y_test[-train_size:]

for i in range(k):
    print("neighbour number:", i+1)
    knn_values[i] = knn_shapley(i+1, deep_feats[:train_size], deep_feats[train_size:train_size*2], 
                                  targets[:train_size], targets[train_size:train_size*2])
print(len(knn_values[0]))
print(knn_values[0][:10])
print('---4. draw plot---')
plot_knn(knn_values, sx_train, sy_train, sx_test, sy_test, deep_feats)
