#!/usr/bin/env python
# coding: utf-8

# In[296]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pyswarm import pso
from scipy.optimize import minimize, NonlinearConstraint


# In[297]:


chips_data = pd.read_csv("chips.csv", header=None)[1:]
chips_X = chips_data.loc[:, 0:1]
chips_y = np.where(chips_data.loc[:, 2] == 'P', 1, -1).astype(np.int8)


# In[298]:


geyser_data = pd.read_csv("geyser.csv", header=None)[1:]
geyser_X = geyser_data.loc[:, 0:1]
geyser_y = np.where(geyser_data.loc[:, 2] == 'P', 1, -1).astype(np.int8)


# In[309]:


class SVM:
    X_train = None
    y_train = None
    X_test = None
    y_test = None


    def fit(self, X, y, kernel_name='linear', C=1, d=1, xi=1):
        self.X_train = np.array(X, dtype=np.float)
        self.y_train = np.array(y)
        self.N_train = self.X_train.shape[0]
        self.k_train = self.X_train.shape[1]
        self.C = C
        self.w0 = np.random.sample(self.N_train)

        def gaussian(a, b):
            res = np.empty((0, b.shape[0]))
            for row_a in a:
                row = np.array([])
                for row_b in b:
                    row = np.append(row, (row_a - row_b).sum())
                res = np.append(res, np.array([row]), axis=0)
            return res

        kernels = {
            'linear': lambda a, b: a @ b.T,
            'polynomial': lambda a, b: (a @ b.T + 1)**d,
            'gaussian': lambda a, b: np.exp(-xi * gaussian(a, b)**2),
        }
        self.kernel = kernels[kernel_name]
        self.__train__()


    def predict(self, X):
        return np.sign(self.K(X))

    def __possitive_cut__(self, x):
        return (x + np.abs(x))/2

    def K(self, X=None, w0=None):
        X = X if X is not None else self.X_train
        w0 = w0 if w0 is not None else self.w0
        return  (self.lamd * self.y_train * self.kernel(X, self.X_train)).sum(axis=1) - w0


    def __opt_func__(self, lamd):
        a = lamd.reshape((-1,1))
        b = self.y_train.reshape((-1,1))
        return .5 * (a*a.T * b*b.T * self.kernel(self.X_train, self.X_train)).sum() - a.sum()

    def __constraints__(self, lamd):
        return (lamd*self.y_train).sum()

    def __train__(self):
        # ub = np.full(self.N_train, self.C)
        # lb = np.full(self.N_train, 0)
        # self.lamd, fopt = pso(self.__opt_func__, lb=lb, ub=ub, f_ieqcons=self.__constraints__, swarmsize=100, maxiter=1000)
        # print((self.lamd*self.y_train).sum())
        # print(fopt)
        # self.w = np.array([(self.lamd * self.y_train * self.X_train[:, 0]).sum(axis=0), (self.lamd * self.y_train * self.X_train[:, 1]).sum(axis=0)])
        # print(self.K())
        # print(self.K(w0=1))
        # print(self.K(w0=0))
        # self.w0 = (self.w @ self.X_train.T - self.y_train)[np.where(self.lamd>0)[0][0]]
        # print(self.w0)
        bounds = np.full((self.N_train, 2), (0, self.C))
        constraints = NonlinearConstraint(self.__constraints__, 0, 0)
        res = minimize(self.__opt_func__, np.random.sample(self.N_train)*self.C, method='trust-constr', constraints=constraints, bounds=bounds)
        print(res)
        self.lamd = res.x
        print((self.lamd*self.y_train).sum())
        w = np.array([(self.lamd * self.y_train * self.X_train[:, 0]).sum(axis=0), (self.lamd * self.y_train * self.X_train[:, 1]).sum(axis=0)])
        self.w0 = w @ self.X_train.T - self.y_train
        print(self.w0)



    def visualize(self):
        X = self.X_train
        y = self.y_train
        xx, yy = np.meshgrid(np.linspace(X[:,0].min() - 1, X[:,0].max() + 1, 1000), np.linspace(X[:,1].min() - 1, X[:,1].max() + 1, 1000))
        pred = self.predict(np.c_[xx.ravel(), yy.ravel()])
        pred = pred.reshape(xx.shape)

        plt.pcolormesh(xx, yy, pred, cmap=ListedColormap(['#AAAAFF','#FFAFAF']))

        plt.scatter(X[:,0], X[:,1], c=y)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.show()


# In[310]:


svm = SVM()


# In[301]:


# for C in [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
svm.fit(geyser_X, geyser_y)
svm.visualize()


# In[305]:


# for C in [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
# for i in range(2, 6):
svm.fit(chips_X, chips_y, kernel_name='polynomial', d=2)
svm.visualize()


# In[ ]:


# for i in range(1, 6):
svm.fit(chips_X, chips_y, kernel_name='gaussian', xi=1)
svm.visualize()
# print(svm.predict(np.array(geyser_X, dtype=np.float)))
# print(geyser_y)

