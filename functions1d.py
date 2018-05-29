import torch
import numpy as np

l = 3
def real0(x):
    return torch.exp(-l*x)

def trial0(f, x):
    return 1 + x * f(x)

def diff0(t, x, gx):
    return gx[0] + l * x

int0=(0, 1)

degree0 = 1

def real1(x):
    return torch.exp(-x**2/2)/(1+x+x**3)+x**2

def trial1(f, x):
    return 1+x*f(x)

def diff1(t, x, gx):
    t2 = t**2
    t3 = t**3
    return gx[0] + (t + (1+3*t2)/(1+t+t3))*x - t3 - 2*t - t2*(1+3*t2)/(1+t+t3)

int1=(0, 1)

degree1 = 1

def real2(x):
    return torch.exp(-x/5)*torch.sin(x)

def trial2(f, x):
    return x*f(x)

def diff2(t, x, gx):
    return gx[0] + x/5 - torch.exp(-t/5) * torch.cos(t)

int2=(0, 2)

degree2 = 1

def real3(x):
    return torch.exp(-x/5)*torch.sin(x)

def trial3(f, x):
    return x+x**2*f(x)

def diff3(t, x, gx):
    return gx[1] + gx[0]/5 + x + torch.exp(-t/5) * torch.cos(t)/5

int3=(0, 2)

degree3 = 2

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












