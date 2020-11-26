#!/usr/bin/env python
# coding: utf-8

# In[91]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone


# In[92]:


chips_data = pd.read_csv("./data/chips.csv", header=None)[1:]
chips_X = np.array(chips_data.loc[:, 0:1], dtype=np.float)
chips_y = np.where(chips_data.loc[:, 2] == 'P', 1, -1).astype(np.int8)


# In[93]:


geyser_data = pd.read_csv("./data/geyser.csv", header=None)[1:]
geyser_X = np.array(geyser_data.loc[:, 0:1], dtype=np.float)
geyser_y = np.where(geyser_data.loc[:, 2] == 'P', 1, -1).astype(np.int8)


# In[94]:


class CustomClassifier:
    def __init__(self, H, alpha):
        self.alpha = alpha
        self.H = H

    def predict(self, X):
        pred = np.empty((X.shape[0], 0))
        for h in self.H:
            pred = np.append(pred, h.predict(X).reshape((-1, 1)), axis=1)
        return np.sign((self.alpha * pred).sum(axis=1))

    def score(self, X, y):
        return (self.predict(X) == y).sum()/y.size


# In[95]:


def ada_boost(clf, X_train, y_train, T):
    D = np.full(y_train.size, 1/y_train.size)
    H = []
    alpha = np.array([])

    for t in range(T):
        h_t = clone(clf).fit(X_train, y_train, sample_weight=D)
        pred = h_t.predict(X_train)
        err = (D * (pred != y_train)).sum()

        a_t = .5 * np.log((1 - err)/(err + 1/y_train.size))
        H.append(h_t)
        alpha = np.append(alpha, a_t)

        D = D * np.exp(-a_t * y_train * pred)
        D = D/D.sum()

    return CustomClassifier(H, alpha)


# In[119]:


def test(X_train, y_train, X_test, y_test, depths=range(1, 2)):
    for depth in depths:
        accuracies = []
        for T in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
            accuracies.append(ada_boost(DecisionTreeClassifier(max_depth=depth) ,X_train, y_train, T).score(X_test, y_test))
        plt.plot([1, 2, 3, 5, 8, 13, 21, 34, 55], accuracies, label='depth = %d' % depth)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.), ncol=2)
    plt.ylabel('accuracy')
    plt.ylim((0,1.1))
    plt.xlabel('T - шаг бустинга')
    plt.show()


# In[120]:


sep = int(chips_y.size*.8)
test(chips_X[:sep], chips_y[:sep], chips_X[sep:], chips_y[sep:], range(1, 11))


# In[108]:


sep = int(geyser_y.size*.8)
test(geyser_X[:sep], geyser_y[:sep], geyser_X[sep:], geyser_y[sep:], range(1, 11))

