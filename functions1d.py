import torch
import numpy as np

def real1(x):
    return torch.exp(-x**2/2)/(1+x+x**3)+x**2

def trial1(f, x):
    return 1+x*f(x)

int1=(0, 1)

def real2(x):
    return torch.exp(-x/5)*torch.sin(x)

def trial2(f, x):
    return x*f(x)

int2=(0, 2)

def real3(x):
    return torch.exp(-x/5)*torch.sin(x)

def trial3(f, x):
    return x+x**2*f(x)

int3=(0, 2)

def real3b(x):
    return torch.exp(-x/5)*torch.sin(x)

def trial3b(f, x):
    return x*np.sin(1.)*np.exp(-1./5)+x*(1-x)*f(x)

int3b=(0, 1)

function={
'p1':real1,
'p2':real2,
'p3':real3,
'p3b':real3b
}

trial={
'p1':trial1,
'p2':trial2,
'p3':trial3,
'p3b':trial3b
}

interval={
'p1':int1,
'p2':int2,
'p3':int3,
'p3b':int3b
}

