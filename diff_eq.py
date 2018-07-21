import torch
from torch.autograd import Variable
from torch.autograd import grad

import numpy as np

# class differential equations
class DiffEq():
    
    # create a set of training/testing points in the interval selected, equally spaced or randomly selected
    def create_points(self, n, points_type='rand'):
        if points_type == 'rand':
            t = Variable(torch.rand((n, 1))*(self.interval[1]-self.interval[0])+self.interval[0], requires_grad=True)            
        else:
            t = Variable(torch.linspace(self.interval[0], self.interval[1], n).view(-1, 1), requires_grad=True)
        return t
            
    def create_points2(self, n, points_type='rand'):
        if points_type == 'rand':
            t = Variable(torch.rand((n, 1))*(self.interval2[1]-self.interval2[0])+self.interval2[0], requires_grad=True)           
        else:
            t = Variable(torch.linspace(self.interval2[0], self.interval2[1], n).view(-1, 1), requires_grad=True)
        return t
    
    
    # encodes the IC (now implemented just for 1 ic)
    def trial(self, t, x):
        l = len(self.ic)
        if l == 0:
            return x
        elif l == 1:
            return self.ic[0][1] + (t-self.ic[0][0])*x
        elif l == 2:
            return self.ic[0][1]+(t-self.ic[0][0])*(t-self.ic[1][0])*x+(t-self.ic[0][0])*self.ic[1][1]
        
        else:
            print('wrong ic')
            return x
       
    # the solution of the differential equation
    def predict(self, t, nn):
        x = nn(t)
        xhat = self.trial(t, x)
        return xhat
        
    # calculation of the needed derivatives
    def calc_deriv(self, t, x):
        gx = []
        gx.append(grad([x], [t], grad_outputs=torch.ones(t.shape), create_graph=True)[0])
        for d in range(1, self.degree):
            gx.append(grad(gx[d-1], [t], grad_outputs=torch.ones(t.shape), create_graph=True)[0])
        return gx
    
    # calculate the values at the chosen points, calculate derivatives and evaluate the differential equation
    def pipeline(self, t, nn):
        x = self.predict(t, nn)
        gx = self.calc_deriv(t, x)
        f = self.diff_eq(t, x, gx)**2
        if self.energy != None:
            f+=(self.energy(t, x, gx)-self.energy0)**2
        return f
    
    
    def __init__(self, interval, real, ic, diff_eq, degree, interval2=None, energy=None, energy0=None):
        self.interval = interval
        self.interval2 = interval2
        self.real = real
        self.ic = ic
        self.diff_eq = diff_eq
        self.degree = degree
        self.energy = energy
        self.energy0 = energy0
        
        
# to create an instance give: an interval, the IC, the real solution, the differential equation and its degree
def create_exp(lamb=1):
    interval = (0, 2)
    ic = [(0, 1)]
    def real(t):
        return torch.exp(-lamb*t)
    degree = 1
    def diff_eq(t, x, gx):
        return gx[0]+lamb*x

    ex = DiffEq(interval=interval,
                         real=real,
                         ic=ic,
                         diff_eq=diff_eq,
                         degree=degree,
               )
    return ex
    
          
        
def create_ho(a=1, omega=1):
    interval=(1*np.pi, 2*np.pi)
    interval2=(0*np.pi, 1*np.pi)
    ic = [(0, 1), (0, 0)]
    def real(t):
        return torch.cos(omega*t)*a
    degree = 2
    def diff_eq(t, x, gx):
        return gx[1]+x*omega**2
          
    def energy(t, x, gx):
        return 0.5*(gx[0]**2+(omega*x)**2)
    energy0 = 0.5
    
    ho = DiffEq(interval=interval,
                real=real,
                ic=ic,
                diff_eq=diff_eq,
                degree=degree,
                interval2=interval2,
                #energy=energy, energy0=energy0,
               )
    return ho
          
          
          
          
          
          
          
          
          
          
          
          
