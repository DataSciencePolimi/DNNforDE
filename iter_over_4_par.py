
# coding: utf-8

# In[1]:


import numpy as np
import utils2 as ut
import diff_eq as de
import pandas as pd
import csv


# In[2]:


def run(n_train, lr, h, k, n_times, n_epoch, n_test, diff_eq, optim_type):
    er = []
    tot_loss = []
    for n in range(n_times):
        seq = ut.build_nn(h, k)
        seq, loss = ut.train_nn(diff_eq, seq, n_epoch, n_train, lr, optim_type=optim_type)
        t_test, x_pred, x_real, error = ut.test_nn(diff_eq, seq, n_test)
        er.append(error.data.numpy())
        tot_loss.append(loss)
    er_mean = np.nanmean(er)
    loss_mean = np.nanmean(tot_loss, axis=0)
    
    return er_mean, loss_mean


# In[3]:


# fixed parameters

n_epoch = 50000
n_test  = 10000

n_times = 10
optim_type = 'Adam'

de1 = de.create_exp(lamb=1)


# In[4]:


# variables

hs  = [2**i for i in range(0, 5)]       # number of hidden neurons per layer
ks  = [i for i in range(1, 4)]          # number of hidden layers
ns  = [2**i for i in range(1,11)]       # number of training points
lrs = [1e-5*2**i for i in range(0, 11)] # learning rate (fixed)

print('hs:',  hs)
print('ks:',  ks)
print('ns:',  ns)
print('lrs:', lrs)

print('total combinations: ', len(hs)*len(ks)*len(ns)*len(lrs)*n_times)


# In[5]:


file_out = 'analysis_10.csv'
f = open(file_out, 'w')
csvwriter = csv.writer(f)
csvwriter.writerow(['h']+['k']+['n_train']+['lr']+['error']+['loss'])
print('h, k, n, lr, error')
for h in hs:
    for k in ks:
        for n_train in ns:
            for lr in lrs:
                error, loss = run(n_train, lr, h, k, 
                                  n_times=n_times, 
                                  n_epoch=n_epoch, 
                                  n_test=n_test, 
                                  diff_eq=de1, 
                                  optim_type=optim_type
                                 )
                f = open(file_out, 'a')
                csvwriter = csv.writer(f)
                
                csvwriter.writerow([h]+[k]+[n_train]+[lr]+[error]+[loss[-1000:]])
                f.close()
                print(h, k, n_train, lr, error)

