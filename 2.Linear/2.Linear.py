#!/usr/bin/env python
# coding: utf-8

# In[120]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('tkagg')


# In[141]:


class LinearRegression:

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.X_train = X / (np.amax(y))
        self.y_train = y / (np.amax(y))
        self.N_train = X.shape[0]
        self.M = X.shape[1]


    def to_int_tpl_map(self, line):
        return tuple(map(int, line.strip().split()))


    def fit_from_file(self, file_name=None, file=None):
        if file_name:
            file = open(file_name)
        raw_data = tuple(map(self.to_int_tpl_map, file.readlines()))
        self.M = int(raw_data[0][0])
        self.N_train = int(raw_data[1][0])
        train_data = np.array(raw_data[2:self.N_train+2])
        self.y_train = train_data[:, -1] / np.amax(train_data[:, -1])
        self.X_train = train_data[:,:-1] / np.amax(train_data[:, -1])
        self.N_test = int(raw_data[self.N_train+2][0])
        test_data = np.array(raw_data[self.N_train+3:])
        self.y_test = test_data[:,-1]
        self.X_test = test_data[:,:-1]


    def rmse_loss(self, y_pred):
        return np.sqrt((((y_pred - self.y_train)**2).sum())/self.N_train)

    def SVM(self):
        self.w = np.linalg.pinv(self.X_train) @ self.y_train


    def gradient(self, lr=0.01, batch_size=75, n_epochs=100000):

        def loss_derivative(y, y_pred):
            return -2 * (y - y_pred)
        self.w = np.random.sample(self.M)
        for it in range(n_epochs):
            if it % batch_size == 0:
                y_pred = self.X_train @ self.w
                loss_der = loss_derivative(self.y_train, y_pred)
                grad = (np.array([self.X_train[n]*loss_der[n] for n in range(self.N_train)])).sum(axis=0)/self.N_train
                # print('grad:', ' '.join(map(str, grad)))
            self.w = self.w - lr * grad


    def genetic(self, n_epochs=16, loss_limit = 50000):
        n_gens = 2**n_epochs
        # print(n_gens)
        number_len = 22
        e_len = 5
        gens = (np.random.sample((n_gens, self.M * number_len)) < 0.5).astype('int8')
        for epoch in range(n_epochs):
            # print(epoch)
            # print(gens.shape)
            gens_loss = []
            for gen in gens:
                # print([float(int(gen[i+5])*2.)**((2**np.arange(e_len)*gen[1+5:e_len+1+5]).sum() + e_len + 1 - i ) for i in range(e_len+1,number_len)])
                # print(-1*sum([2.**((2**np.arange(e_len)*gen[1+5:e_len+1+5]).sum() + e_len + 1 - i ) for i in range(e_len+1,number_len)]))
                # print([float(int(gen[i+5])*2.)for i in range(e_len+1,number_len)])
                # print((-1**int(gen[0+m])) * sum([(gen[i+m]*2.)**((2**np.arange(e_len)*gen[1+m:e_len+1+m]).sum() + e_len + 1 - i ) for i in range(e_len+1,number_len)]))
                w = np.array(
                    [
                        (-1**int(gen[0+m])) * sum([(2.**gen[i+m])**((2**np.arange(e_len)*gen[1+m:e_len+1+m]).sum() + e_len + 1 - i ) for i in range(e_len+1,number_len)])
                        for m in range(0, self.M * number_len, number_len)
                    ]
                )
                # print(w)
                gens_loss.append(self.rmse_loss(self.X_train @ w))
            gens_loss = np.array(gens_loss)
            # print(gens_loss[gens_loss.argsort()])
            sorted_indexes = gens_loss.argsort()
            sorted_gens = gens[sorted_indexes]
            n_gens = n_gens//2
            gens = sorted_gens[np.where(gens_loss[sorted_indexes][:n_gens] < loss_limit)]
            if gens.shape[0]<n_gens:
                gens = np.concatenate((gens, (np.random.sample((n_gens-gens.shape[0], self.M * number_len)) < 0.5).astype('int8')),axis=0)
            if n_gens > 1:
                gens = np.array([[gens[n+((i%2) * (-1**(n%2)))][i] for i in range(self.M * number_len)] for n in range(n_gens)])
                for n in range(n_gens):
                    rand_i = np.random.randint(0, self.M*number_len)
                    gens[n][rand_i] = -gens[n][rand_i]
            # print(gens.shape)
        top_gen = gens[0]
        self.w = np.array(
            [
                (-1**int(top_gen[0+m])) * np.array([(2.**top_gen[i+m])**((2**np.arange(e_len)*top_gen[1+m:e_len+1+m]).sum() + e_len + 1 - i ) for i in range(e_len+1,number_len)]).sum()
                for m in range(0, self.M * number_len, number_len)
            ]
        )



    def nrmse_score(self, X_test=None, y_test=None):
        if X_test:
            self.X_test = X_test
        if X_test:
            self.y_test = y_test
        y_pred = self.X_test @ self.w
        return np.sqrt(((y_pred - self.y_test)**2).sum()/self.y_test.shape[0])/(np.amax(self.y_test) - np.amin(self.y_test))


    def smape_score(self, X_test=None, y_test=None):
        if X_test:
            self.X_test = X_test
        if X_test:
            self.y_test = y_test
        y_pred = self.X_test  @ self.w
        return 100/self.y_test.shape[0] * (2*np.abs(y_pred - self.y_test)/(np.abs(self.y_test) + np.abs(y_pred))).sum()


# In[142]:


reg = LinearRegression()


# In[72]:


reg.fit_from_file("1.txt")
reg.SVM()
print(reg.smape_score())


# In[87]:


reg.fit_from_file("2.txt")
grad_res = []
for n_epochs in [100, 500, 1000, 10000, 100000]:
    for batch_size in [100, 75, 50, 20, 10, 5 ,1]:
        for lr in 10.**np.arange(-4, -1):
            reg.gradient(lr=lr, batch_size=batch_size, n_epochs=n_epochs)
            smape = reg.smape_score()
            nrmse = reg.nrmse_score()
            grad_res.append([lr, batch_size, n_epochs, smape, nrmse])
            # print('gradient: lr_%f, batch_size_%d, n_epochs_%d',smape)

sorted_res = sorted(grad_res, key=lambda x: x[4])[:10]
print('grad_top10: lr    batch size  n epochs    smape   nrmse\n', '\n'.join([' '.join(map(str, line)) for line in sorted_res]))


# In[88]:


grad_top5 = sorted_res


# In[90]:


print('grad_top5: lr    batch size  n epochs    smape   nrmse\n', '\n'.join([' '.join(map(str, line)) for line in grad_top5]))


# In[98]:


grad_x_plot = list(range(100, 1000, 100))
grad_x_plot.extend(list(range(1000, 10000, 1000)))
grad_x_plot.extend(list(range(10000, 110000, 10000)))
grad_y_plot = []
for n_epochs in grad_x_plot:
    print(n_epochs, end='... ')
    grad_top1 = lr=grad_top5[0]
    reg.gradient(lr=grad_top1[0], batch_size=grad_top1[1], n_epochs=n_epochs)
    grad_y_plot.append(reg.smape_score())


# In[133]:


grad_x_plot_copy = grad_x_plot.copy()
grad_y_plot_copy = grad_y_plot.copy()


# In[132]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(grad_x_plot, grad_y_plot)
plt.title("Gradient")
plt.ylabel('SMAPE', fontsize=16)
plt.xlabel('n epochs', fontsize=16)
plt.show()


# In[74]:


reg.fit_from_file("3.txt")
reg.genetic()
print(reg.nrmse_score())
print(reg.smape_score())


# In[143]:


reg.fit_from_file("3.txt")
genetic_x_plot = list(range(20))
genetic_y_plot = []
for n_epochs in genetic_x_plot:
    print(n_epochs, end='... ')
    reg.genetic(n_epochs=n_epochs)
    genetic_y_plot.append(reg.smape_score())
    print(' '.join(map(str, genetic_y_plot)))


# In[ ]:


genetic_x_plot_copy = genetic_x_plot.copy()
genetic_y_plot_copy = genetic_y_plot.copy()


# In[150]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(genetic_x_plot, genetic_y_plot)
plt.title("Genetic")
plt.ylabel('SMAPE', fontsize=16)
plt.xlabel('n epochs', fontsize=16)
plt.xticks(list(range(20)))
plt.show()

