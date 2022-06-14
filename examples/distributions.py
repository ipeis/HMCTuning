# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from src.hmc import *
import torch.distributions as D

def get_logp(name):

    if name == 'bivariate_gaussian':
        logp = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.ones(2),
            covariance_matrix=torch.Tensor([[1.2, 0.7], [0.7, 0.5]])
            ).log_prob
    elif name == 'gaussian_mixture':
        K=8     # Number of components
        means = torch.Tensor([[np.cos(k * np.pi/4), np.sin(k * np.pi/4)] for k in range(K)])
        vars = torch.Tensor([[0.12, 0.12] for k in range(K)])
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
        # wave2
        def logp(x):
            x0 = x[0,:,0]
            x1 = x[0,:,1]
            term1 = torch.exp(-0.5* ((x1 + torch.sin(0.5*np.pi* x0))/0.35)**2)
            term2 = torch.exp(-0.5* ((-x1 - torch.sin(0.5*np.pi* x0) + 3*torch.exp(-0.5/0.36* (x0-1)**2))/0.35)**2)
            return torch.log(1e-300+ term1 + term2)        
    return logp



"""def get_dlogp(name):
    if name == 'gaussian_mixture':
        def dlogp(x):
            P = torch.exp(get_logp('gaussian_mixture')(x))
            num_modes = 7
            angle = torch.arange(0, 2 * np.pi, 2 * np.pi / num_modes)
            mu = torch.stack([torch.cos(angle), torch.sin(angle)], axis=0).reshape(1, -1, 2) * 5.0
            zz = x[None,:].reshape((x.shape[1],1,2))
            grad_x = torch.sum(torch.exp(-0.5*(torch.sum((zz-mu)**2,2)))* (zz-mu)[:,:,0],1)
            grad_y = torch.sum(torch.exp(-0.5*(torch.sum((zz-mu)**2,2)))* (zz-mu)[:,:,1],1)
            return torch.stack((-grad_x/P, -grad_y/P), -1)

    return dlogp"""

def plot_distribution(name):
    return 0


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