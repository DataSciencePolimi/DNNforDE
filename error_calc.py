import torch
from torch.autograd import Variable
from torch.autograd import grad
import torch.optim as optim
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

def trial(f, t, bc, a=False):
    l = len(bc)
    if l == 0:
        return f(t)
    elif l == 1:
        return bc[0][1] + (t-bc[0][0])*f(t)
    elif l == 2 and a==False:
        return bc[0][1]*(t-bc[1][0])+bc[1][1]*(t-bc[0][0])+(t-bc[0][0])*(t-bc[1][0])*f(t)
    elif l == 2 and a==True:
        return bc[0][1] + bc[1][1]*(t-bc[0][0])*(t-bc[1][0])**2*f(t)
    else:
        print('error')
        return np.nan

import functions1d as fun

which = 'p0'
# options available: p0, p1, p2, p3, p3b

diff_eq, degree = fun.diff_eq[which], fun.degree[which]
real = fun.real[which]
bc = fun.bc[which]
t_min, t_max = fun.interval[which]

D_in, H, D_out = 1, 10, 1 # number of input, of hidden neurons and output

def f(t, nn):
    x = trial(nn, t, bc)
    
    gx = []
    gx.append(grad([x], [t], grad_outputs=torch.ones(t.shape), create_graph=True)[0])
    for d in range(1, degree):
        gx.append(grad(gx[d-1], [t], grad_outputs=torch.ones(t.shape), create_graph=True)[0])
    z = diff_eq(t, x, gx)
    return z


N_train = np.arange(4, 30, 2)

Error = []
for n_train in N_train:
    print("n_train: ", n_train)
    #t_train = Variable(torch.linspace(t_min, t_max, int(n_train)).view(-1, 1), requires_grad=True)
    t_train = Variable(torch.rand((n_train, 1))*(t_max-t_min)+t_min, requires_grad=True)
    seq = torch.nn.Sequential(
                              torch.nn.Linear(D_in, H),
                              #torch.nn.ReLU(),
                              #torch.nn.Sigmoid(),
                              torch.nn.LogSigmoid(),
                              torch.nn.Linear(H, D_out)
                              )

    criterion = torch.nn.MSELoss(size_average=False)
    #criterion = torch.nn.L1Loss(size_average=False)

    lr = 1e-1
    #optimizer = optim.Adam(seq.parameters(), lr=lr)
    optimizer = optim.LBFGS(seq.parameters(), lr=lr)
    
    n_epoch = 500
    #Loss = []
    for t in tqdm(range(n_epoch)):
        y = f(t_train, seq)
        loss = criterion(y, torch.zeros(y.shape))
        optimizer.zero_grad()
        loss.backward()
        #optimizer.step()
        def closure():
            y = f(t_train, seq)
            loss = criterion(y, torch.zeros(y.shape))
            optimizer.zero_grad()
            loss.backward()
            return loss
        optimizer.step(closure)
        #Loss.append(loss.data.numpy())
        if np.isnan(loss.data.numpy()):
            break
    n_test = 1000
    t_test = Variable(torch.linspace(t_min, t_max, n_test).view(-1, 1), requires_grad=True)
    x_real = real(t_test)
    x_pred = trial(seq, t_test, bc)
    error = abs(x_real-x_pred)
    print('Error: ', error.mean().data.numpy())
    Error.append(error.mean())

plt.plot(N_train, Error)
plt.yscale('log')
plt.savefig('n_train_vs_error_rand.png')
plt.show()

