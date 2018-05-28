import torch
import numpy as np

def real5(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return torch.exp(-x1)*(x1+x2**3)

def trial5(f, x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x2_3 = x2**3
    e1 = np.exp(-1.)
    ex1 = torch.exp(-x1)
    
    A = (1-x1)*x2_3+x1*(1+x2_3)*e1+(1-x2)*x1*(ex1-e1)+x2*((1+x1)*ex1-(1-x1+2*x1*e1))
    return A + x1*(1-x1)*x2*(1-x2)*f(x).t()

int5=(0, 1, 0, 1)

def real6(x, a=3):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return torch.exp(-(a*x1+x2)/5)*torch.sin((a*x1)**2+x2)

def trial6(f, x, a=3):
    x1 = x[:, 0]
    x2 = x[:, 1]
    a2 = a**2
    x1_2 = x1**2
    eax5 = torch.exp(-a*x1/5)
    ey5 = torch.exp(-x2/5)
    ea5 = np.exp(-a/5)
    e15 = np.exp(-1./5)
    c1 = ey5*torch.sin(x2)
    c2 = ey5*torch.sin(a2+x2)*ea5
    c3 = eax5*torch.sin(a2*x1_2)-x1*np.sin(a2)*ea5
    c4 = (eax5*torch.sin(a2*x1_2+1)-(1-x1)*np.sin(1)-x1*ea5*np.sin(a2+1))*e15
    
    A = (1-x1)*c1+x1*c2+(1-x2)*c3+x2*c4
    return A + x1*(1-x1)*x2*(1-x2)*f(x).t()

int6=(0, 1, 0, 1)

# not complete

def real7(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return x2**2*torch.sin(x1*np.pi)

def trial7(f, x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x_1 = Variable(torch.cat((torch.Tensor(x1), torch.Tensor(np.ones(len(x1)))), 1))
    print(x_1)
    B = 1
    return B + x1*(1-x1)*x2*(f(x) - f(x_1) - 1)

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

