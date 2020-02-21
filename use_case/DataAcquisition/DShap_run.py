import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import tensorflow as tf
import sys
from shap_utils import *
from Shapley import ShapNN
from scipy.stats import spearmanr
import shutil
from sklearn.base import clone
import matplotlib.pyplot as plt
import warnings
import itertools
import _pickle as pkl
from sklearn.metrics import f1_score, roc_auc_score

class DShap(object):
    
    def __init__(self, X, y, X_test, y_test, num_test, sources=None, directory=None, 
                 problem='classification', model_family='logistic', metric='accuracy',
                 seed=None, **kwargs):
        """
        Args:
            X: Data covariates
            y: Data labels
            X_test: Test+Held-out covariates
            y_test: Test+Held-out labels
            sources: An array or dictionary assiging each point to its group.
                If None, evey points gets its individual value.
            num_test: Number of data points used for evaluation metric.
            directory: Directory to save results and figures.
            problem: "Classification" or "Regression"(Not implemented yet.)
            model_family: The model family used for learning algorithm
            metric: Evaluation metric
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting 
                same permutations.
            **kwargs: Arguments of the model
        """
            
        if seed is not None:
            np.random.seed(seed)
#             tf.random.set_random_seed(seed)
        self.problem = problem
        self.model_family = model_family
        self.metric = metric
        self.directory = directory
        self.hidden_units = kwargs.get('hidden_layer_sizes', [])
        if self.model_family is 'logistic':
            self.hidden_units = []
        if self.directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)  
                os.makedirs(os.path.join(directory, 'weights'))
                os.makedirs(os.path.join(directory, 'plots'))
            self._initialize_instance(X, y, X_test, y_test, num_test, sources)
        if len(set(self.y)) > 2:
            assert self.metric != 'f1' and self.metric != 'auc', 'Invalid metric!'
        is_regression = (np.mean(self.y//1==self.y) != 1)
        is_regression = is_regression or isinstance(self.y[0], np.float32)
        self.is_regression = is_regression or isinstance(self.y[0], np.float64)
        self.model = return_model(self.model_family, **kwargs)
        self.random_score = self.init_score(self.metric)
            
    def _initialize_instance(self, X, y, X_test, y_test, num_test, sources=None):
        """Loads or creates data."""
        
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        self.X_heldout, self.y_heldout = X_test[:-num_test], y_test[:-num_test]
        self.X_test, self.y_test = X_test[-num_test:], y_test[-num_test:]
        self.X, self.y, self.sources = X, y, sources
#             pkl.dump({'X': self.X, 'y': self.y, 'X_test': self.X_test,
#                      'y_test': self.y_test, 'X_heldout': self.X_heldout,
#                      'y_heldout':self.y_heldout, 'sources': self.sources}, 
#                      open(data_dir, 'wb'))        
        self.vals_loo = None
            
        previous_results =  os.listdir(self.directory)
#         print(previous_results)
        tmc_numbers = [int(name.split('.')[-2].split('_')[-1])
                      for name in previous_results if 'mem_tmc' in name]
        g_numbers = [int(name.split('.')[-2].split('_')[-1])
                     for name in previous_results if 'mem_g' in name]
#         print(tmc_numbers, g_numbers)
        if(tmc_numbers == []):
            self.tmc_number = str(0) if len(g_numbers)==0 else str(np.max(tmc_numbers) + 1)
            self.g_number = str(0) if len(g_numbers)==0 else str(np.max(g_numbers) + 1)
            tmc_dir = os.path.join(self.directory, 'mem_tmc_{}.pkl'.format(self.tmc_number.zfill(4)))
            g_dir = os.path.join(self.directory, 'mem_g_{}.pkl'.format(self.g_number.zfill(4)))
            self.mem_tmc, self.mem_g = [np.zeros((0, self.X.shape[0])) for _ in range(2)]
            idxs_shape = (0, self.X.shape[0] if self.sources is None else len(self.sources.keys()))
            self.idxs_tmc, self.idxs_g = [np.zeros(idxs_shape).astype(int) for _ in range(2)]
#             pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc}, open(tmc_dir, 'wb'))
            if self.model_family not in ['logistic', 'NN']:
                return
#             pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g}, open(g_dir, 'wb'))
        else:
            # function to continue to calculate the tmc and g shapley
            tmc_dir = os.path.join(self.directory, 'mem_tmc_0000.pkl')
            g_dir = os.path.join(self.directory, 'mem_g_0000.pkl')
            self.tmc_number = str(0) if len(g_numbers)==0 else str(np.max(tmc_numbers))
            self.g_number = str(0) if len(g_numbers)==0 else str(np.max(g_numbers))
            with open(tmc_dir, 'rb') as tmc_f:
                data = pkl.load(tmc_f)
                print(data)
                self.mem_tmc = data['mem_tmc']
                self.idxs_tmc = data['idxs_tmc']
                self.vals_tmc = np.mean(self.mem_tmc, 0)
            with open(g_dir, 'rb') as g_f:
                data = pkl.load(g_f)
                self.mem_g = data['mem_g']
                self.idxs_g = data['idxs_g']
                self.vals_g = np.mean(self.mem_g, 0)
                                
    def init_score(self, metric):
        """ Gives the value of an initial untrained model."""
        if metric == 'accuracy':
            return np.max(np.bincount(self.y_test).astype(float)/len(self.y_test))
        if metric == 'f1':
            return np.mean([f1_score(
                self.y_test, np.random.permutation(self.y_test)) for _ in range(1000)])
        if metric == 'auc':
            return 0.5
        random_scores = []
        for _ in range(100):
            self.model.fit(self.X, np.random.permutation(self.y))
            random_scores.append(self.value(self.model, metric))
        return np.mean(random_scores)
        
    def value(self, model, metric=None, X=None, y=None):
        """Computes the values of the given model.
        Args:
            model: The model to be evaluated.
            metric: Valuation metric. If None the object's default
                metric is used.
            X: Covariates, valuation is performed on a data different from test set.
            y: Labels, if valuation is performed on a data different from test set.
            """
        if metric is None:
            metric = self.metric
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
        if metric == 'accuracy':
#             print(X.shape, y.shape)
            return model.score(X, y)
        if metric == 'f1':
            assert len(set(y)) == 2, 'Data has to be binary for f1 metric.'
            return f1_score(y, model.predict(X))
        if metric == 'auc':
            assert len(set(y)) == 2, 'Data has to be binary for auc metric.'
            return my_auc_score(model, X, y)
        if metric == 'xe':
            return my_xe_score(model, X, y)
        raise ValueError('Invalid metric!')

        #g_run=True, loo_run=True, tmc_run=False, save_run=True
    def run(self, save_every, err, tolerance=0.01, g_run=True, loo_run=True, tmc_run=True, save_run=True):
        """Calculates data sources(points) values.
        
        Args:
            save_every: save marginal contrivbutions every n iterations.
            err: stopping criteria for each of TMC-Shapley or G-Shapley algorithm.
            tolerance: Truncation tolerance. If None, the instance computes its own.
            g_run: If True, computes G-Shapley values.
            loo_run: If True, computes and saves leave-one-out scores.
        """
#         self._knn_shapley()
        if loo_run:
            try:
                len(self.vals_loo)
            except:
                self.vals_loo = self._calculate_loo_vals(sources=self.sources)
                if(save_run == True):
                    self.save_results(overwrite=True)
            print('LOO values calculated!')
        tmc_run, g_run = tmc_run, g_run and self.model_family in ['logistic', 'NN', 'uci']
#         tmc_run, g_run = False, False
#         print(tmc_run, g_run)
        while tmc_run or g_run:         
            if g_run:
                print("g_err:%f"%(error(self.mem_g)))
                print(self.mem_g.shape)
                if error(self.mem_g) < err:
                    g_run = False
                else:
                    self._g_shap(save_every, sources=self.sources)
                    self.vals_g = np.mean(self.mem_g, 0)
#                     print("vals_g:", self.vals_g)
            if tmc_run:
                print("tmc_err:%f"%(error(self.mem_tmc)))
                print(self.mem_tmc.shape)
                if error(self.mem_tmc) < err:
                    tmc_run = False
                else:
                    self._tmc_shap(save_every, tolerance=tolerance, sources=self.sources)
                    self.vals_tmc = np.mean(self.mem_tmc, 0)
#                     print("vals_tmc:", self.vals_tmc)
            if self.directory is not None:
                if save_run == True: 
                    self.save_results()
        self.save_vals(g_run, loo_run, tmc_run)
        
    def save_vals(self, g_run=True, loo_run=True, tmc_run=True):
        if(tmc_run == True):
            tmc_dir = os.path.join(self.directory, 'vals_tmc.pkl')
            pkl.dump({'vals_tmc': self.vals_tmc}, open(tmc_dir, 'wb'))
            print('vals_tmc:', self.vals_tmc)
        if(g_run == True):
            g_dir = os.path.join(self.directory, 'vals_g.pkl')
            pkl.dump({'vals_g': self.vals_g}, open(g_dir, 'wb'))
            print('vals_g:', self.vals_g)
        return
            
    def save_results(self, overwrite=False):
        """Saves results computed so far."""
        if self.directory is None:
            return
        loo_dir = os.path.join(self.directory, 'loo.pkl')
#         knn_dir = os.path.join(self.directory, 'knn.pkl')
        if not os.path.exists(loo_dir) or overwrite:
            pkl.dump({'loo': self.vals_loo}, open(loo_dir, 'wb'))
#         if not os.path.exists(knn_dir) or overwrite:
#             pkl.dump({'knn': self.vals_knn}, open(knn_dir, 'wb'))
            
        tmc_dir = os.path.join(self.directory, 'mem_tmc_{}.pkl'.format(self.tmc_number.zfill(4)))
        g_dir = os.path.join(self.directory, 'mem_g_{}.pkl'.format(self.g_number.zfill(4)))
        print(tmc_dir, g_dir)
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc, 'err_tmc': error(self.mem_tmc)}, open(tmc_dir, 'wb'))
        pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g, 'err_g': error(self.mem_g)}, open(g_dir, 'wb'))
#         print(self.mem_tmc.size, self.idxs_tmc.size)
        
    def _knn_shapley(self):
        print("start knn shapley!")
        K = 5
        trainX, valX = self.X, self.X_test
        trainy, valy = self.y, self.y_test
            
        N = trainX.shape[0]
        M = valX.shape[0]

        value = np.zeros(N)
        for i in range(M):
            X = valX[i]
            y = valy[i]

            s = np.zeros(N)
            diff = (trainX - X).reshape(N, -1) # calculate the distances between valX and every trainX data point
            dist = np.einsum('ij, ij->i', diff, diff) # output the sum distance
            idx = np.argsort(dist) # ascend the distance
            ans = trainy[idx] # get mapping y
            s[idx[N - 1]] = float(ans[N - 1] == y) / N # calculate the last data value

            cur = N - 2
            for j in range(N - 1):
                s[idx[cur]] = s[idx[cur + 1]] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) / K * min(K, cur + 1) / (cur + 1)
                cur -= 1
            for j in range(N): # caculate the sum of deviation of every valx
                value[j] += s[j]
              
        for i in range(N):
            value[i] /= M
        self.vals_knn = value
            
            
    def _tmc_shap(self, iterations, tolerance=None, sources=None):
        """Runs TMC-Shapley algorithm.
        
        Args:
            iterations: Number of iterations to run. # iterations == every_save
            tolerance: Truncation tolerance. (ratio with respect to average performance.)
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        model = self.model
        try:
            self.mean_score            
        except:
            self._tol_mean_score()
        if tolerance is None:
            tolerance = self.tolerance  
        marginals, idxs = [], []
        for iteration in range(iterations):
            if 10*(iteration+1)/iterations % 1 == 0:
                print('{} out of {} TMC_Shapley iterations.'.format(iteration + 1, iterations))
            marginals, idxs = self.one_iteration(tolerance=tolerance, sources=sources)
#             print(marginals, idxs)
            self.mem_tmc = np.concatenate([self.mem_tmc, np.reshape(marginals, (1,-1))])
            self.idxs_tmc = np.concatenate([self.idxs_tmc, np.reshape(idxs, (1,-1))])
        
    def _tol_mean_score(self):
        """Computes the average performance and its error using bagging."""
        scores = []
        self.restart_model()
        for _ in range(1):
            self.model.fit(self.X, self.y)
            for _ in range(100):
                bag_idxs = np.random.choice(len(self.y_test), len(self.y_test))
                scores.append(self.value(self.model, metric=self.metric,
                                         X=self.X_test[bag_idxs], y=self.y_test[bag_idxs]))
        self.tol = np.std(scores)
        self.mean_score = np.mean(scores)
        
    def one_iteration(self, tolerance, sources=None):
        """Runs one iteration of TMC-Shapley algorithm."""
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        idxs, marginal_contribs = np.random.permutation(len(sources.keys())), np.zeros(len(self.X)) # set a permutation sequence
        new_score = self.random_score
        X_batch, y_batch = np.zeros((0,) +  tuple(self.X.shape[1:])), np.zeros(0).astype(int)
        truncation_counter = 0
        for n, idx in enumerate(idxs):
            old_score = new_score
            X_batch = np.concatenate([X_batch, self.X[sources[idx]]])
            y_batch = np.concatenate([y_batch, self.y[sources[idx]]])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.is_regression or len(set(y_batch)) == len(set(self.y_test)): ##FIXIT
                    self.restart_model()
                    self.model.fit(X_batch, y_batch)
                    new_score = self.value(self.model, metric=self.metric)      # calculate the score by utility function 
            marginal_contribs[sources[idx]] = (new_score - old_score) / len(sources[idx]) # calculate the Shapley Value
            if np.abs(new_score - self.mean_score) <= tolerance * self.mean_score:   # decide whether to truncate based on  performance tolerance
                truncation_counter += 1
                if truncation_counter > 5:
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs
    
    def restart_model(self):
        
        try:
            self.model = clone(self.model)
        except:
            self.model.fit(np.zeros((0,) + self.X.shape[1:]), self.y)
        
    def _one_step_lr(self):
        """Computes the best learning rate for G-Shapley algorithm."""
        if self.directory is None:
            address = None
        else:
            address = os.path.join(self.directory, 'weights')
        best_acc = 0.0
        for i in np.arange(1, 5, 0.5):
            model = ShapNN(
                self.problem, batch_size=1, max_epochs=1, 
                learning_rate=10**(-i), weight_decay=0., 
                validation_fraction=0, optimizer='sgd', warm_start=False,
                address=address, hidden_units=self.hidden_units)
            accs = []
            for _ in range(10):
                model.fit(np.zeros((0, self.X.shape[-1])), self.y)
                model.fit(self.X, self.y)
                accs.append(model.score(self.X_test, self.y_test))
            if np.mean(accs) - np.std(accs) > best_acc:
                best_acc  = np.mean(accs) - np.std(accs)
                learning_rate = 10**(-i)
        return learning_rate
    
    def _g_shap(self, iterations, err=None, learning_rate=None, sources=None):
        """Method for running G-Shapley algorithm.
        
        Args:
            iterations: Number of iterations of the algorithm.
            err: Stopping error criteria
            learning_rate: Learning rate used for the algorithm. If None
                calculates the best learning rate.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        address = None
        if self.directory is not None:
            address = os.path.join(self.directory, 'weights')
        if learning_rate is None:
            try:
                learning_rate = self.g_shap_lr
            except AttributeError:
                self.g_shap_lr = self._one_step_lr()
                learning_rate = self.g_shap_lr
        model = ShapNN(self.problem, batch_size=1, max_epochs=1,
                     learning_rate=learning_rate, weight_decay=0.,
                     validation_fraction=0, optimizer='sgd',
                     address=address, hidden_units=self.hidden_units)
        for iteration in range(iterations):
            model.fit(np.zeros((0, self.X.shape[-1])), self.y)
            if 10 * (iteration+1) / iterations % 1 == 0:
                print('{} out of {} G-Shapley iterations'.format(iteration + 1, iterations))
            marginal_contribs = np.zeros(len(sources.keys()))
            model.fit(self.X, self.y, self.X_test, self.y_test, sources=sources,
                      metric=self.metric, max_epochs=1, batch_size=1)
            val_result = model.history['metrics']
            marginal_contribs[1:] += val_result[0][1:]
            marginal_contribs[1:] -= val_result[0][:-1]
            individual_contribs = np.zeros(len(self.X))
            for i, index in enumerate(model.history['idxs'][0]):
                individual_contribs[sources[index]] += marginal_contribs[i]
                individual_contribs[sources[index]] /= len(sources[index])
            self.mem_g = np.concatenate(
                [self.mem_g, np.reshape(individual_contribs, (1,-1))])
            self.idxs_g = np.concatenate(
                [self.idxs_g, np.reshape(model.history['idxs'][0], (1,-1))])
    
    def _calculate_loo_vals(self, sources=None, metric=None):
        """Calculated leave-one-out values for the given metric.
        
        Args:
            metric: If None, it will use the objects default metric.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        
        Returns:
            Leave-one-out scores
        """
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        print('Starting LOO score calculations!')
        if metric is None:
            metric = self.metric 
        self.restart_model()
        self.model.fit(self.X, self.y)
        baseline_value = self.value(self.model, metric=metric)
        vals_loo = np.zeros(len(self.X))
        for i in sources.keys():
            X_batch = np.delete(self.X, sources[i], axis=0)
            y_batch = np.delete(self.y, sources[i], axis=0)
            self.model.fit(X_batch, y_batch)
            removed_value = self.value(self.model, metric=metric)
            vals_loo[sources[i]] = (baseline_value - removed_value)/len(sources[i])
        return vals_loo
    
    def _merge_parallel_results(self, key):
        """Helper method for 'merge_results' method."""
        numbers = [name.split('.')[-2].split('_')[-1]
                   for name in os.listdir(self.directory) if 'mem_{}'.format(key) in name]
        mem  = np.zeros((0, self.X.shape[0]))
        idxs_shape = (0, self.X.shape[0] if self.sources is None else len(self.sources.keys()))
        idxs = np.zeros(idxs_shape)
        vals = np.zeros(len(self.X))
        counter = 0.
        for number in numbers:
            samples_dir = os.path.join(self.directory, 'mem_{}_{}.pkl'.format(key, number))
            print(samples_dir)
            dic = pkl.load(open(samples_dir, 'rb'))
            mem = np.concatenate([mem, dic['mem_{}'.format(key)]])
            idxs = np.concatenate([idxs, dic['idxs_{}'.format(key)]])
            if not len(dic['mem_{}'.format(key)]):
                continue
            counter += len(dic['mem_{}'.format(key)])
            vals *= (counter - len(dic['mem_{}'.format(key)])) / counter
            vals += len(dic['mem_{}'.format(key)]) / counter * np.mean(mem, 0)
            os.remove(samples_dir)
        merged_dir = os.path.join(self.directory, 'mem_{}_0000.pkl'.format(key))
        pkl.dump({'mem_{}'.format(key): mem, 'idxs_{}'.format(key): idxs}, 
                 open(merged_dir, 'wb'))
        return mem, idxs, vals
            
    def merge_results(self):
        """Merge all the results from different runs.
        
        Returns:
            combined marginals, sampled indexes and values calculated 
            using the two algorithms. (If applicable)
        """
        self.marginals_tmc, self.indexes_tmc, self.values_tmc = self._merge_parallel_results('tmc')
        if self.model_family not in ['logistic', 'NN', 'KNN']:
            return
        self.marginals_g, self.indexes_g, self.values_g = self._merge_parallel_results('g')
    
    def performance_plots(self, vals, name=None, num_plot_markers=20, sources=None):
        """Plots the effect of removing valuable points.
        
        Args:
            vals: A list of different valuations of data points each
                 in the format of an array in the same length of the data.
            name: Name of the saved plot if not None.
            num_plot_markers: number of points in each plot.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
                   
        Returns:
            Plots showing the change in performance as points are removed
            from most valuable to least.
        """
        plt.figure(figsize=(8,8))
#         plt.rcParams['figure.figsize'] = 8,8
        plt.rcParams['font.size'] = 20
        plt.xlabel('Fraction of train data removed (%)')
#         plt.xlabel('Fraction of train data removed')

        plt.ylabel('Prediction accuracy (%)', fontsize=20)
        if not isinstance(vals, list) and not isinstance(vals, tuple):
            vals = [vals]
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        vals_sources = [np.array([np.sum(val[sources[i]]) for i in range(len(sources.keys()))])
                  for val in vals]  # values of three algorithms: TMC, G, LOO
        if len(sources.keys()) < num_plot_markers:
            num_plot_markers = len(sources.keys()) - 1
        plot_points = np.arange(0, max(len(sources.keys()) - 10, num_plot_markers),
                           max(len(sources.keys())//num_plot_markers, 1))
        print(plot_points)
#         plot_points = np.flip(plot_points, 0)
#         print(plot_points)
        print(vals_sources[0].size, vals_sources[1].size, vals_sources[2].size)
        
        perfs = [self._portion_performance(
            np.argsort(vals_source)[::-1], plot_points, sources=sources)
                 for vals_source in vals_sources]
        print(perfs)
        rnd = np.mean([self._portion_performance(
            np.random.permutation(np.argsort(vals_sources[0])[::-1]), # vals_sources will sort from high value to low value and output indexes; ATTENTION: [::-1] reverses the sequence
            plot_points, sources=sources) for _ in range(1)], 0)  # repeat 10 times to get the average result; change to 2
        print(rnd)

        plt.plot(plot_points/len(self.X) * 100, perfs[0] * 100, '-', lw=5, ms=10, color='b') #TMC-Shapley
        if len(vals)==4:
            plt.plot(plot_points/len(self.X) * 100, perfs[1] * 100, '--', lw=5, ms=10, color='orange') #G-Shapley
            legends = ['TMC-Shapley ', 'G-Shapley ', 'KNN-Shapley', 'LOO', 'Random']
        elif len(vals)==3:
            legends = ['TMC-Shapley ', 'KNN-Shapley', 'LOO', 'Random']
#         else: # meaningless?
#             legends = ['TMC-Shapley ', 'KNN-Shapley', 'Random']
        plt.plot(plot_points/len(self.X) * 100, perfs[-2] * 100, ':', lw=5, ms=10, color='y') #  KNN-Shapley

        plt.plot(plot_points/len(self.X) * 100, perfs[-1] * 100, '-.', lw=5, ms=10, color='g') # LOO
        plt.plot(plot_points/len(self.X) * 100, rnd * 100, ':', lw=5, ms=10, color='r')    
        
        plt.legend(legends, prop={'size': 10})
        print(self.directory, name)
        plt.savefig('temp.png')
        if self.directory is not None and name is not None:
            plt.savefig(os.path.join(
                self.directory, 'plots', '{}.png'.format(name)),
                        bbox_inches = 'tight')
            plt.close()
            
    def _portion_performance(self, idxs, plot_points, sources=None):
        """Given a set of indexes, starts removing points from the first elemnt
           and evaluates the new model after removing each point."""
#         print(idxs)
        idxs = np.flip(idxs,0)
#         print(idxs)
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        scores = []
        print("sources:", sources)
        init_score = self.random_score
        for i in range(len(plot_points), 0, -1):
            print(i)
            keep_idxs = np.concatenate([sources[idx] for idx in idxs[plot_points[i-1]:]], -1)
            print("keep_idxs:", keep_idxs)
#             print()
            X_batch, y_batch = self.X[keep_idxs], self.y[keep_idxs]
#             print(X_batch.shape, y_batch.shape, keep_idxs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.is_regression or len(set(y_batch)) == len(set(self.y_test)):
                    self.restart_model()
                    self.model.fit(X_batch, y_batch)
                    scores.append(self.value(self.model, metric=self.metric, 
                                             X=self.X_heldout, y=self.y_heldout))
                else:
                    scores.append(init_score)
        return np.array(scores)[::-1]
