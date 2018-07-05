from tqdm import tqdm
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

# select activation function
def add_activation(fc, which, i):
    if which == 'sigmoid':
        fc.add_module("sig"+str(i), torch.nn.Sigmoid())
        
    elif which == 'logsigmoid':
        fc.add_module("logsig"+str(i), torch.nn.LogSigmoid())
        
    elif which == 'relu':
        fc.add_module("logsig"+str(i), torch.nn.ReLu())
        
    else:
        print('wrong activation')
        return 0
    
    
# build a sequential neural network with D_in input, D_out output, k layers of h hinnen neurons
def build_nn(h=1, k=1, D_in=1, D_out=1, activation='sigmoid'):
    fc = torch.nn.Sequential()
    fc.add_module("fc0", torch.nn.Linear(D_in, h))
    add_activation(fc, activation, 0)
    for i in range(k-1):
        fc.add_module("fc"+str(i+1), torch.nn.Linear(h, h))
        add_activation(fc, activation, i+1)
        
    fc.add_module("fc_fin", torch.nn.Linear(h, D_out))
    return fc

# train the nn to solve a differential equation (de), given the number of epoch, the number of training points, learning rate and other parameters (loss, optimizer, ...)
def train_nn(de, nn, n_epoch, n_train, lr=1e-3, optim_type='Adam'):
    Loss = []
    
    # loss function
    criterion = torch.nn.L1Loss(size_average=False)

    # optimizer
    if optim_type=='SGD':
        optimizer = optim.SGD(nn.parameters(), lr=lr, momentum=0.9)
    elif optim_type=='Adam':
        optimizer = optim.Adam(nn.parameters(), lr=lr)
    elif optim_type=='LBFGS':
        optimizer = optim.LBFGS(nn.parameters(), lr=lr)
        def closure():
            f = de.pipeline(t_train, nn)
            loss = criterion(f, Variable(torch.zeros(f.shape)))
            optimizer.zero_grad()
            loss.backward()
            return loss
    else:
        print('optimizer not supported')
        return np.nan, []
    
    # training
    epsilon = 1e-9
    for epoch in range(n_epoch):
        t_train = de.create_points(n_train)
        f = de.pipeline(t_train, nn)
        
        loss = criterion(f, Variable(torch.zeros(f.shape)))
        
        optimizer.zero_grad()
        loss.backward()
        if optim_type=='LBFGS':
            optimizer.step(closure)
        else: 
            optimizer.step()
        
        Loss.append(loss.data.numpy())
        #if epoch > 2 and abs(Loss[-1]-Loss[-2]) < epsilon:
        #    print('equal loss')
        #    break
        
        if np.isnan(loss.data.numpy()):
            break
            
    return nn, Loss


# test the performance: create a testing set, predict the values and calculate the error with respect to the real values
def test_nn(de, nn, n_test):
    t_test = de.create_points(n_test)
    x_pred = de.predict(t_test, nn)
    x_real = de.real(t_test)
    error = ((x_pred-x_real)**2).mean()
    return t_test, x_pred, x_real, error
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
