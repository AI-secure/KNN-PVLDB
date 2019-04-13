import time
import numpy as np
from scipy.stats import norm
from tqdm import tqdm


def get_contrast(x_trn, save_dir='eps0.1/'):
    num_cores = 8
    mc_num = 5
    eps = 0.1
    n_trn = x_trn.shape[0]
    K = int(1 / eps)

    def compute_distance(i_q, query, x_trn, n_trn, K):
        dist_to_random = np.zeros(n_trn)
        for i_trn in range(n_trn):
            dist_to_random[i_trn] = np.linalg.norm(query - x_trn[i_trn, :], 2)
        dist_to_random_avg = np.mean(dist_to_random)
        dist_to_KNN = np.sort(dist_to_random)[:K]
        if i_q % 100 == 0:
            print(i_q)
        return dist_to_random_avg, dist_to_KNN

    def estimate_contrast(x_trn, query, K):
        # estimate empirical contrast
        n_trn = x_trn.shape[0]
        n_q = query.shape[0]
        from joblib import Parallel, delayed
        result = Parallel(n_jobs=num_cores)(
            delayed(compute_distance)(i_q, query[i_q, :], x_trn, n_trn, K) for i_q in range(n_q))
        dist_to_random_avg = np.array([result[i][0] for i in range(n_q)])
        dist_to_KKN = np.array([result[i][1] for i in range(n_q)])
        assert dist_to_KKN.shape[0] == n_q
        dist_to_KNN_avg_q = np.mean(dist_to_KKN, axis=0)
        dist_to_random_avg_avg = np.mean(dist_to_random_avg)
        contrast = dist_to_random_avg_avg / dist_to_KNN_avg_q
        return dist_to_random_avg_avg, dist_to_KNN_avg_q, contrast

    contrast = []
    dist_rand = []
    dist_knn = []
    for mc_i in range(mc_num):
        start = time.time()
        sample_ind_trn = np.random.choice(np.arange(n_trn), int(n_trn / 5 * 4), replace=False).astype(int)
        sample_ind_query = np.array(
            list(set(np.arange(n_trn).astype(int).tolist()) - set(sample_ind_trn.tolist()))).astype(int)
        dist_rand_, dist_knn_, contrast_ = estimate_contrast(x_trn[sample_ind_trn, :], x_trn[sample_ind_query, :], K)
        dist_rand.append(dist_rand_)
        dist_knn.append(dist_knn_)
        contrast.append(contrast_)

        print('monte carlo iteration%s ' % mc_i)
        elapsed_time = time.time() - start
        print('elapsed time is %s' % elapsed_time)
    dist_knn = np.array(dist_knn)
    contrast = np.array(contrast)
    dist_rand = np.array(dist_rand)
    np.save(save_dir + 'dist_rand', dist_rand)
    np.save(save_dir + 'dist_knn', dist_knn)
    np.save(save_dir + 'contrast', contrast)


def f_h(x, r):
    y = 1 - 2 * norm.cdf(-r / x) - 2 / (np.sqrt(2 * np.pi) * r / x) * (1 - np.exp(-(r ** 2 / (2 * (x ** 2)))))
    return y


def g_unnormalize(dist_rand, dist_knn, r):
    y = np.log(f_h(dist_knn, r)) / np.log(f_h(dist_rand, r))
    return y


def g_normalize(contrast, r):
    y = np.log(f_h(1 / contrast, r)) / np.log(f_h(1, r))
    return y


def find_best_r_normalize(search_range, contrast):
    y = g_normalize(contrast, search_range)
    min_ind = np.argmin(y)
    return search_range[min_ind]


def find_best_r_unnormalize(search_range, dist_rand, dist_knn):
    y = g_unnormalize(dist_rand, dist_knn, search_range)
    min_ind = np.argmin(y)
    return search_range[min_ind]


def lsh_function(t, x, w, b):
    # x is 1-d array
    h = np.floor((np.dot(w, x) + b) / t).astype(int)
    return h


class LSH:
    def __init__(self, n_hash_bit, n_hash_table, x_trn, y_trn, dist_rand, equal, t=0.1):
        self.n_hash_bit = n_hash_bit
        self.n_hash_table = n_hash_table
        self.t = t  # width of projections
        self.dist_rand = dist_rand
        self.x_trn = x_trn
        self.y_trn = y_trn
        self.N, self.dim = x_trn.shape
        self.equal = equal
        # draw w from a normal distribution (2-stable)
        self.w = np.random.normal(0, 1, (n_hash_table, n_hash_bit, self.dim))
        # draw b from U[0,t]
        self.b = np.random.uniform(0, self.t, (n_hash_table, n_hash_bit))
        self.x_trn_hash = [dict() for i in range(n_hash_table)]
        for i in tqdm(range(self.N)):
            hash_code_all = lsh_function(self.t, x_trn[i] / dist_rand, self.w, self.b)
            for l in range(n_hash_table):
                hash_code_trn = '.'.join(map(str, hash_code_all[l, :]))
                if hash_code_trn in self.x_trn_hash[l].keys():
                    self.x_trn_hash[l][hash_code_trn].append(i)
                else:
                    self.x_trn_hash[l][hash_code_trn] = [i]

    def get_approx_KNN(self, x_tst, K):
        N_tst = x_tst.shape[0]
        x_tst_knn = np.ones((N_tst, K)) * (-1)
        nns_len = np.zeros(N_tst)
        for i_tst in tqdm(range(N_tst)):
            nns = []
            for l in range(self.n_hash_table):
                hash_code_int = lsh_function(self.t, x_tst[i_tst] / self.dist_rand, self.w[l, :, :], self.b[l, :])
                hash_code_test = '.'.join(map(str, hash_code_int))
                if hash_code_test in self.x_trn_hash[l].keys():
                    nns += self.x_trn_hash[l][hash_code_test]
            nns = np.unique(nns)
            num_collide_elements = len(nns)
            if len(nns) > 0:
                dist = [np.linalg.norm(self.x_trn[i] / self.dist_rand - x_tst[i_tst] / self.dist_rand, 2) for i in nns]
                dist_min_ind = nns[np.argsort(dist)]
                if num_collide_elements < K:
                    x_tst_knn[i_tst, :num_collide_elements] = dist_min_ind[:num_collide_elements]
                else:
                    x_tst_knn[i_tst, :] = dist_min_ind[:K]
            # pdb.set_trace()
            nns_len[i_tst] = len(nns)
            if i_tst % 100 == 0:
                print('get approximate knn %s' % i_tst)
        return x_tst_knn.astype(int), nns_len

    def compute_approx_shapley(self, x_tst_knn, y_tst, K):
        N_tst, K_star = x_tst_knn.shape
        # flag_sufficient = (x_tst_knn[:,-1]>=0)
        sp_approx = np.zeros((N_tst, self.N))
        for j in tqdm(range(N_tst)):
            non_nan_index = np.where(x_tst_knn[j, :] >= 0)[0]
            if len(non_nan_index) == 0:
                continue
            K_tot = non_nan_index[-1]
            if K_tot == self.N:
                sp_approx[j, x_tst_knn[j, self.N - 1]] = self.equal(self.y_trn[x_tst_knn[j, self.N - 1]],
                                                                    y_tst[j]) / self.N
            for i in np.arange(K_tot - 1, -1, -1):
                sp_approx[j, x_tst_knn[j, i]] = sp_approx[j, x_tst_knn[j, i + 1]] + (
                        self.equal(self.y_trn[x_tst_knn[j, i]], y_tst[j]) - self.equal(
                    self.y_trn[x_tst_knn[j, i + 1]], y_tst[j])) / K * min([K, i + 1]) / (i + 1)

        return sp_approx
