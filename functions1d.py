import torch
import numpy as np


# SIMPLE PROBLEM

def real0(t, lamb=3):
    return torch.exp(-lamb*t)

def trial0(f, t):
    return 1 + t*f(t)

def diff0(t, x, gx, lamb=3):
    return gx[0]+lamb*x

int0=(0, 1)
degree0 = 1


# PROBLEM N. 1

def real1(t):
    t2 = t**2
    return torch.exp(-t2/2)/(1+t+t**3)+t2

def trial1(f, t):
    return 1+t*f(t)

def diff1(t, x, gx):
    t2 = t**2
    t3 = t**3
    return gx[0]+(t+(1+3*t2)/(1+t+t3))*x-t3-2*t-t2*(1+3*t2)/(1+t+t3)

int1=(0, 1)
degree1 = 1


# PROBLEM N. 2

def real2(t):
    return torch.exp(-t/5)*torch.sin(t)

def trial2(f, t):
    return t*f(t)

def diff2(t, x, gx):
    return gx[0]+x/5-torch.exp(-t/5)*torch.cos(t)

int2=(0, 2)
degree2 = 1


# PROBLEM N. 3

def real3(t):
    return torch.exp(-t/5)*torch.sin(t)

def trial3(f, t):
    return t+t**2*f(t)

def diff3(t, x, gx):
    return gx[1]+gx[0]/5+x+torch.exp(-t/5)*torch.cos(t)/5

int3=(0, 2)
degree3 = 2


# PROBLEM N. 3b

def trial3b(f, x):
    return x*np.sin(1.)*np.exp(-1./5)+x*(1-x)*f(x)

int3b=(0, 1)

function={
'p0':real0,
'p1':real1,
'p2':real2,
'p3':real3,
'p3b':real3
}

trial={
'p0':trial0,
'p1':trial1,
'p2':trial2,
'p3':trial3,
'p3b':trial3b
}

diff_eq={
'p0':diff0,
'p1':diff1,
'p2':diff2,
'p3':diff3,
'p3b':diff3
}

interval={
'p0':int0,
'p1':int1,
'p2':int2,
'p3':int3,
'p3b':int3b
}

degree={
'p0':degree0,
'p1':degree1,
'p2':degree2,
'p3':degree3,
'p3b':degree3
}












