# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from src.hmc import *
import torch.distributions as D
import matplotlib.pyplot as plt

def get_logp(name):

    if name == 'bivariate_gaussian':
        logp = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.ones(2),
            covariance_matrix=torch.Tensor([[1.2, 0.7], [0.7, 0.5]])
            ).log_prob
    elif name == 'gaussian_mixture':
        K=8     # Number of components
        means = torch.Tensor([[np.cos(k * np.pi/4), np.sin(k * np.pi/4)] for k in range(K)])
        vars = torch.Tensor([[0.15, 0.15] for k in range(K)])
        mix = D.Categorical(torch.ones(K))
        comp = D.Independent(D.Normal(
            means, vars), 1)
        gmm = D.MixtureSameFamily(mix, comp)
        logp = gmm.log_prob

    elif name == 'dual_moon':
        # dual moon
        def logp(x): 
            x1 = x[0,:,0]
            x2 = x[0,:,1]
            term1 = 3.125 * (torch.sqrt(x1**2+x2**2)-2)**2
            term2 = torch.log(1e-16+torch.exp(-0.5*((x1+2)/0.6)**2) + torch.exp(-0.5*((x1-2)/0.6)**2))
            return -term1 + term2     

    elif name == 'wave':
        # wave
        def logp(x):
            x0 = x[...,:,0]
            x1 = x[...,:,1]
            term1 = torch.exp(-0.5* ((x1 + torch.sin(0.5*np.pi* x0))/0.35)**2)
            term2 = torch.exp(-0.5* ((-x1 - torch.sin(0.5*np.pi* x0) + 3*torch.exp(-0.5/0.36* (x0-1)**2))/0.35)**2)
            logp = torch.log(1e-300+ term1 + term2)   
            return logp
    return logp

def plot_density(name, ax):
    logp = get_logp(name)
    xmin, xmax, ymin, ymax = get_grid_lims(name)
    x1grid = np.linspace(xmin, xmax, 1000)
    x2grid = np.linspace(ymin, ymax, 1000)

    X1, X2 = np.meshgrid(x1grid, x2grid)
    # Pack X and Y into a single 3-dimensional array
    X = np.empty(X1.shape + (2,))
    X[:, :, 0] = X1
    X[:, :, 1] = X2
    p = torch.exp(logp(torch.Tensor(X)))
    cm = plt.cm.Blues
    cm.set_under(color='w')
    ax.contourf(X1, X2, p, cmap = cm, alpha=0.5)
    ax.set_facecolor('white')


def get_grid_lims(distribution):
    if distribution=='gaussian_mixture':
        xmin = -1.3
        xmax = 1.3
        ymin = -1.3
        ymax = 1.3
            
    elif distribution=='wave':
        xmin = -10
        xmax = 10
        ymin = -10
        ymax = 10
    return xmin, xmax, ymin, ymax


def initial_proposal(distribution):

    if distribution=='gaussian_mixture':
        mu0 = torch.zeros([2])
        var0 = torch.ones([2])*0.01

    elif distribution=='wave':
        mu0 = torch.zeros([2])
        #var0 = torch.Tensor([5.5, 1])
        var0 = torch.Tensor([1, 1])
    
    return mu0, var0


def update_proposal(distribution, iter, niters):

    if distribution=='gaussian_mixture':
        delta = 2*np.pi/niters
        mu0 = torch.Tensor([np.cos(iter * delta), np.sin(iter*delta)])
        var0 = torch.ones([2])*0.01


    elif distribution=='wave':
        mu0 = torch.zeros([2])
        var0 = torch.Tensor([5.5, 1])
    
    return mu0, var0