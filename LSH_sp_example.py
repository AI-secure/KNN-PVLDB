import time
from math import ceil

import numpy as np
from sklearn.utils import shuffle

from LSH_sp import get_contrast, find_best_r_normalize, g_normalize, f_h, LSH
import matplotlib.pyplot as plt

data = np.load('CIFAR10_resnet50-keras_features.npz')
x_trn = np.vstack((data['features_training'], data['features_testing']))
y_trn = np.hstack((data['labels_training'], data['labels_testing']))

x_trn, y_trn = shuffle(x_trn, y_trn, random_state=0)

x_trn = np.reshape(x_trn, (-1, 2048))
x_tst, y_tst = x_trn[:100], y_trn[:100]
x_val, y_val = x_trn[100:1100], y_trn[100:1100]
x_trn, y_trn = x_trn[1100:], y_trn[1100:]

# we are using 1-nn classifier
K = 1
eps = 0.1

K_star = max(K, ceil(1 / eps))
get_contrast(x_val)
dist_rand = np.load('eps0.1/dist_rand.npy')
contrast = np.load('eps0.1/contrast.npy')
dist_knn = np.load('eps0.1/dist_knn.npy')

dist_rand = np.mean(dist_rand, axis=0)
contrast = np.mean(contrast, axis=0)[K_star - 1]
dist_knn = np.mean(dist_knn, axis=0)[K_star - 1]

search_range = np.arange(1e-3, 10, 1e-3)
r_vec_normalize = find_best_r_normalize(search_range, contrast)
g_vec = g_normalize(contrast, r_vec_normalize)

# plot g(C_K) vs r, we want g(C_k) to be small
# search range, find r that minimize g, shape should be similar to convex
g = g_normalize(contrast, search_range)
plt.figure()
plt.plot(search_range, g)
plt.show()

np.save('eps0.1/selected_param_r_' + str(K_star) + '.npy', r_vec_normalize)
np.save('eps0.1/selected_param_g_' + str(K_star) + '.npy', g_vec)


def equal(a, b):
    return int(a == b)


def fine_tune_val(n_hash_table=10, alpha=0.5, file=False, val_sp_gt=None):
    t = r_vec_normalize
    n_trn = len(x_trn)
    n_hash_bit = int(np.ceil(np.log(n_trn) * alpha / np.log(1 / f_h(1, t))))
    if file is True:
        print(n_hash_bit, file=open('eps0.1/log.txt', 'a'))
    else:
        print(n_hash_bit)

    start = time.time()
    lsh = LSH(n_hash_bit=n_hash_bit, n_hash_table=n_hash_table, x_trn=x_trn, y_trn=y_trn, dist_rand=dist_rand,
              equal=equal, t=t)
    runtime_build_hash = time.time() - start
    if file is True:
        print(runtime_build_hash, file=open('eps0.1/log.txt', 'a'))
    else:
        print(runtime_build_hash)

    start = time.time()
    x_val_knn_approx, nns_vec = lsh.get_approx_KNN(x_val, K_star)
    runtime_query = time.time() - start
    if file is True:
        print(runtime_query, file=open('eps0.1/log.txt', 'a'))
    else:
        print(runtime_query)

    start = time.time()
    sp_approx = lsh.compute_approx_shapley(x_val_knn_approx, y_val, K)
    runtime_approx_value = time.time() - start
    if file is True:
        print('it takes %s to get appox knn value' % runtime_approx_value, file=open('eps0.1/log.txt', 'a'))
    else:
        print('it takes %s to get appox knn value' % runtime_approx_value)

    if val_sp_gt is not None:
        sp_err_inf_val = np.linalg.norm(val_sp_gt - sp_approx, ord=np.inf, axis=1)
        if file is True:
            print('max error %s' % np.percentile(sp_err_inf_val, 90), file=open('eps0.1/log.txt', 'a'))
        else:
            print('max error %s' % np.percentile(sp_err_inf_val, 90))
    return lsh


def fine_tune_test(lsh=None, file=False, sp_gt=None):
    start = time.time()
    x_tst_knn_approx, nns_vec = lsh.get_approx_KNN(x_tst, K_star)
    runtime_query = time.time() - start
    if file is True:
        print(runtime_query, file=open('eps0.1/log.txt', 'a'))
    else:
        print(runtime_query)

    start = time.time()
    sp_approx = lsh.compute_approx_shapley(x_tst_knn_approx, y_tst, K)
    runtime_approx_value = time.time() - start
    if file is True:
        print('it takes %s to get appox knn value' % runtime_approx_value, file=open('eps0.1/log.txt', 'a'))
    else:
        print('it takes %s to get appox knn value' % runtime_approx_value)

    if sp_gt is not None:
        sp_err_inf_val = np.linalg.norm(sp_gt - sp_approx, ord=np.inf, axis=1)
        if file is True:
            print('max error %s' % np.percentile(sp_err_inf_val, 90), file=open('eps0.1/log.txt', 'a'))
        else:
            print('max error %s' % np.percentile(sp_err_inf_val, 90))
    return sp_approx, nns_vec


val_sp_gt = np.load('val_exact_sp_gt.npy')
tst_sp_gt = np.load('tst_exact_sp_gt.npy')
lsh_82_05 = fine_tune_val(82, 0.5, val_sp_gt=val_sp_gt)
sp_approx_82_05, nns_vec_82_05 = fine_tune_test(lsh=lsh_82_05, sp_gt=tst_sp_gt)

np.save('eps0.1/sp_approx_05', sp_approx_82_05)
np.save('eps0.1/lsh_82_05', lsh_82_05)
