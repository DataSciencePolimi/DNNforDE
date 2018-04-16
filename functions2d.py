import torch
import numpy as np

def real5(x1, x2):
    return np.exp(-x1)*(x1+x2**3)

def trial5(f, x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x2_3 = x2**3
    e1 = np.exp(-1.)
    ex1 = torch.exp(-x1)
    
    A = (1-x1)*x2_3+x1*(1+x2_3)*e1+(1-x2)*x1*(ex1-e1)+x2*((1+x1)*ex1-(1-x1+2*x1*e1))
    return A + x1*(1-x1)*x2*(1-x2)*f(x).t()

int5=(0, 1, 0, 1)

def real6(x1, x2):
    a=3
    return 1#torch.exp(-(a*x1+x2)/5)*torch.sin((a*x1)**2+x2)

def trial6(f, x):
    return x*f(x)

int6=(0, 1, 0, 1)

def real7(x):
    return torch.exp(-x/5)*torch.sin(x)

def trial7(f, x):
    return x+x**2*f(x)

int7=(0, 1, 0, 1)

def real8(x):
    return torch.exp(-x/5)*torch.sin(x)

def trial8(f, x):
    return x*np.sin(1.)*np.exp(-1./5)+x*(1-x)*f(x)

int8=(0, 1, 0, 1)

function={
'p5':real5,
'p6':real6,
'p7':real7,
'p8':real8
}

trial={
'p5':trial5,
'p6':trial6,
'p7':trial7,
'p8':trial8
}

interval={
'p5':int5,
'p6':int6,
'p7':int7,
'p8':int8
}

