import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from Shapley import ShapNN
from DShap_run import DShap
from shap_utils import *
from utils import *
import pickle

path = "./exp_data/DS_uci/"
with open(path+'data.pkl', 'rb') as f:
    data = pickle.load(f)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    x_heldout = data["x_heldout"]
    y_heldout = data["y_heldout"]
from models.uci import *
from utils import *
#data preparation
batch_size = 1024
epochs = 30

x_train = torch.from_numpy(x_train).contiguous().view(-1, 254)
y_train = torch.from_numpy(y_train).view(-1,).long()
print("train_size:", x_train.shape)
x_test = torch.from_numpy(x_test).contiguous().view(-1, 254)
y_test = torch.from_numpy(y_test).view(-1,).long()
print("test_size:", x_test.shape)
x_heldout = torch.from_numpy(x_heldout).contiguous().view(-1, 254)
y_heldout = torch.from_numpy(y_heldout).view(-1,).long()
print("heldout_size:", x_heldout.shape)


device = torch.device('cuda')
uci = UCI().to(device)
optimizer = optim.Adam(uci.parameters())
criterion = nn.CrossEntropyLoss()

# print(y_train.shape)
train(uci, device, x_train, y_train, batch_size, optimizer, criterion, epochs)
accuracy, avg_loss = evaluate(uci, device, x_train, y_train, batch_size, criterion)
print(f'[Train] Accuracy: {100 * accuracy:5.2f}%, loss: {avg_loss:7.4f}')
accuracy, avg_loss = evaluate(uci, device, x_heldout, y_heldout, batch_size, criterion)
print(f'[Test] Accuracy: {100 * accuracy:5.2f}%, loss: {avg_loss:7.4f}')


    
deep_f = []
targets = []
x_deep = torch.cat((x_train, x_test), 0)
y_deep = torch.cat((y_train, y_test), 0)
for X, y in batch(x_deep, y_deep, batch_size):
    X = X.to(device).float()
    fc3, y_pre = uci(X)
    deep_f.append(fc3.view(fc3.size(0), -1).cpu().detach().numpy())
#     targets.append(y.numpy())

deep_f = np.concatenate(deep_f) # deep features are not normalized
# targets = np.concatenate(targets)
print(deep_f.shape)

import math
kmin = 5
kmax = 6
kinterval = 5
fc1_knn_values = [[] for _ in range(math.ceil((kmax-kmin)/kinterval))] # deep features
loo_fc1_knn_values = [[] for _ in range(math.ceil((kmax-kmin)/kinterval))] # deep features

for i, k in enumerate(range(kmin, kmax, kinterval)):
    print("neighbour number:", k)
    fc1_knn_values[i],*_ = old_knn_shapley(k, deep_f[:x_train.shape[0]], deep_f[x_train.shape[0]:], 
                                  y_deep[:x_train.shape[0]], y_deep[x_train.shape[0]:])
    loo_fc1_knn_values[i],*_ = loo_knn_shapley(k, deep_f[:x_train.shape[0]], deep_f[x_train.shape[0]:], 
                                  y_deep[:x_train.shape[0]], y_deep[x_train.shape[0]:])    

import pickle
store_data = './exp_data/DS_uci/'
f = open(store_data+'knn.pkl', 'wb')
data_write = {"knn_values": fc1_knn_values, "loo_fc1_knn_values": loo_fc1_knn_values}
pickle.dump(data_write, f)
f.close() 