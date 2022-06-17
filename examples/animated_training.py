# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from cgitb import reset
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir("..")
from src import *
import argparse
import torch

import torch.distributions as D
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal as mnormal
from distributions import *
from utils import *


# ============= ARGS ============= #

parser = argparse.ArgumentParser(description='Train the hyperparameters of HMC using Variational Inference')

parser.add_argument('--distribution', type=str, default='gaussian_mixture', 
                    help='name of the distribution (from srd/distributions.py)')
parser.add_argument('--T', type=int, default=5, 
                    help='length of the HMC chains (number of states)')
parser.add_argument('--L', type=int, default=5, 
                    help='number of Leapfrog steps within each state update')
parser.add_argument('--chains', type=int, default=100, 
                    help='number of parallel chains (n. of HMC samples)')
parser.add_argument('--chains_sksd', type=int, default=30, 
                    help='number of parallel chains for computing the SKSD')      
parser.add_argument('--steps', type=int, default=100, 
                    help='number of training steps')           
parser.add_argument('--gpu', type=int, default=1,
                    help='use gpu via cuda (1) or cpu (0)')
args = parser.parse_args()


# ============= Activate CUDA ============= #
args.cuda = int(args.gpu>0) and torch.cuda.is_available()
args.cuda = args.cuda == True and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
if str(device) == "cuda":
    print('cuda activated')


if not os.path.isdir('./figs/training/{}'.format(args.distribution)):
    os.makedirs('./figs/training/{}'.format(args.distribution))

def plot_chains(distribution, steps, samples_per_step=100, chains_per_step=10):
    
    f, ax = plt.subplots(figsize=(5, 5))


    plot_density(distribution, ax)


    chain_color='gray'
    sample_color='tab:green'
    ax.axis('off')
 
    xmin, xmax, ymin, ymax = get_grid_lims(distribution)   
    ax.set_xlim(xmin, xmax) 
    ax.set_ylim(ymin, ymax) 
    
    def reset_axis(ax, step, mu0, var0, samples=None,):
        ax.clear()

        plot_density(distribution, ax)

        ax.axis('off')

    
        xmin, xmax, ymin, ymax = get_grid_lims(distribution)   
        xgrid = np.linspace(xmin, xmax, 1000)
        ygrid = np.linspace(ymin, ymax, 1000)

        plot_bi_gaussian(mu0, var0, xgrid, ygrid, ax)

        if len(samples) != 0:
            samples = torch.stack(samples).detach().numpy().reshape(-1, 2)
            x = samples[:, 0]
            y = samples[:, 1]
            alphas = np.linspace(0, 1, len(x))
            ax.scatter(x, y, marker='*', color=sample_color, alpha=alphas)
        
        ax.set_xlim(xmin, xmax) 
        ax.set_ylim(ymin, ymax) 
        

    logp = get_logp(distribution)
    mu0, var0 = initial_proposal(distribution)

    hmc = HMC(dim=2, logp=logp, T=args.T,  L=args.L, chains=samples_per_step, chains_sksd=args.chains_sksd, mu0=mu0, var0=var0, vector_scale=True)
    
    hmc.optimizer = torch.optim.Adam([hmc.log_eps] + [hmc.log_v_r], lr=0.01)
    optimizer_scale = torch.optim.Adam([hmc.log_inflation], lr=0.1)

    loss=[]
    objective=[]
    sksd=[]
    inflation = np.empty((2, 0))
    samples = []
    progress = trange(steps, desc='Loss')
    im=0
    for e in progress:

        # Sample 
        z, chains = hmc.sample(chains=samples_per_step)
        chains = chains[:, 0, :chains_per_step, :].reshape(hmc.T+1, chains_per_step, 2)
        alphas = np.linspace(0.1, 1, chains.shape[0])[::-1]
        for c in range(chains.shape[1]):
            for t in range(1,chains.shape[0]+1):
                reset_axis(ax, e, hmc.mu0.detach().numpy(), torch.exp(hmc.logvar0 + 2*hmc.log_inflation).detach().numpy(), samples)
                ax.plot(chains[:t,c,0].detach().numpy(), chains[:t,c,1].detach().numpy(), marker='o', color=chain_color, alpha=alphas[t-1])
                plt.savefig('figs/training/{}/{:05d}.png'.format(distribution, im), bbox_inches='tight')
                im+=1
            ax.plot(chains[-1, c, 0].detach().numpy(), chains[-1, c, 1].detach().numpy(), marker='*', color=sample_color)
            plt.savefig('figs/training/{}/{:05d}.png'.format(distribution, im), bbox_inches='tight')
            im+=1
            samples.append(chains[-1, c])
            reset_axis(ax, e, hmc.mu0.detach().numpy(), torch.exp(hmc.logvar0 + 2*hmc.log_inflation).detach().numpy(), samples)
            plt.savefig('figs/training/{}/{:05d}.png'.format(distribution, im), bbox_inches='tight')
            im+=1
        #samples = samples + list(z)

        hmc.optimizer.zero_grad()
        z, _ = hmc.sample(hmc.mu0, torch.exp(hmc.logvar0), hmc.chains)
        _loss = -hmc.logp(z)

        _loss[torch.isfinite(_loss)].sum().backward()
        hmc.optimizer.step()

        hmc.optimizer.zero_grad()
        _sksd = hmc.evaluate_sksd(hmc.mu0, torch.exp(hmc.logvar0))
        
        # For some densities the _sksd might be ill
        if not _sksd.isnan():
            optimizer_scale.zero_grad()
            _sksd.backward()
            optimizer_scale.step()

        progress.set_description('HMC (objective=%g)' % -_loss[torch.isfinite(_loss)].mean().detach().numpy())

        loss.append(_loss[torch.isfinite(_loss)].detach().numpy().mean())
        objective.append(-loss[-1])
        sksd.append(_sksd.detach().numpy())
        inflation = np.hstack((inflation, torch.exp(hmc.log_inflation.clone()).detach().numpy()[:, np.newaxis]))

    reset_axis(ax, e, hmc.mu0.detach().numpy(), torch.exp(hmc.logvar0 + 2*hmc.log_inflation).detach().numpy(),samples)
    ax.scatter(torch.stack(samples)[-samples_per_step:, 0].detach().numpy(), torch.stack(samples)[-samples_per_step:, 1].detach().numpy(), marker='*', color=sample_color, alpha=0.5)
    plt.savefig('figs/training/{}/samples.pdf'.format(distribution, im), bbox_inches='tight')

if __name__ == '__main__':

    plot_chains(args.distribution, args.steps, samples_per_step=args.chains, chains_per_step=3)

    make_gif('figs/training/{}/'.format(args.distribution), destination='assets/gifs/training_{}.gif'.format(args.distribution))


