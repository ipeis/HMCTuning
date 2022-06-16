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
from PIL import Image
import glob
from tqdm import tqdm
from distributions import *
from utils import *


# ============= ARGS ============= #

parser = argparse.ArgumentParser(description='Train the hyperparameters of HMC using Variational Inference')

parser.add_argument('--distribution', type=str, default='wave', 
                    help='name of the distribution (from srd/distributions.py)')
parser.add_argument('--T', type=int, default=5, 
                    help='length of the HMC chains (number of states)')
parser.add_argument('--L', type=int, default=5, 
                    help='number of Leapfrog steps within each state update')
parser.add_argument('--cycles', type=int, default=2, 
                    help='number of times the cycle is repeated')
parser.add_argument('--niters', type=int, default=25, 
                    help='number of iters per cycles')           
args = parser.parse_args()


if not os.path.isdir('./figs/cycle/{}'.format(args.distribution)):
    os.makedirs('./figs/cycle/{}'.format(args.distribution))

def cyclic_proposal(distribution, cycles=10, niters=1000):
    
    f, ax = plt.subplots(figsize=(5, 5))
    f.tight_layout()
    chain_color='gray'
    sample_color='tab:green'
    plt.axis('off')
    # Gaussian mixture
    #plt.xlim(-1.3, 1.3) 
    #plt.ylim(-1.3, 1.3) 
    # Dual moon
    #plt.xlim(-2, 2) 
    #plt.ylim(-2, 2) 
    if distribution=='wave':
        plt.xlim(-10, 10) 
        plt.ylim(-10, 10) 
    plt.title('step 0')
    
    def reset_axis(ax, mu0, var0, samples=None,):
        ax.clear()
        plt.axis('off')
        # Gaussian mixture
        #plt.xlim(-1.3, 1.3) 
        #plt.ylim(-1.3, 1.3) 
        # Dual moon
        #plt.xlim(-2, 2) 
        #plt.ylim(-2, 2) 
        plot_density(distribution, ax)
        
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
        
        xgrid = np.linspace(xmin, xmax, 1000)
        ygrid = np.linspace(ymin, ymax, 1000)

        plot_bi_gaussian(mu0, var0, xgrid, ygrid, ax)

        if len(samples) != 0:
            samples = torch.stack(samples).detach().numpy().reshape(-1, 2)
            x = samples[:, 0]
            y = samples[:, 1]

            ax.scatter(x, y, marker='*', color=sample_color)
        
        plt.xlim(xmin, xmax) 
        plt.ylim(ymin, ymax) 
    

    logp = get_logp(distribution)

    niters = 100
    im=0    # Image number
    samples = []

    for cycle in range(args.cycles):
        print('Building cycle {}'.format(cycle))
        for iter in tqdm(range(100)):
            mu0, var0 = update_proposal(distribution,iter, niters)
            hmc = HMC(dim=2, logp=logp, T=args.T,  L=args.L, chains=1, mu0=mu0, var0=var0)

            # Sample 
            z, chains = hmc.sample(mu0=hmc.mu0, var0=torch.exp(hmc.logvar0), chains=10)
            samples.append(z)
            alphas = np.linspace(0.1, 1, chains.shape[0])[::-1]

            for t in range(1,chains.shape[0]+1):
                    reset_axis(ax, mu0, var0, samples)
                    ax.plot(chains[:t,0,0, 0].detach().numpy(), chains[:t,0,0, 1].detach().numpy(), marker='o', color=chain_color, alpha=alphas[t-1])
                    plt.savefig('figs/cycle/{}/{:05d}.png'.format(distribution, im))
                    im+=1

            reset_axis(ax, mu0, var0, samples)

    plt.savefig('figs/cycle/{}/samples.pdf'.format(distribution, im))



if __name__ == '__main__':

    # ============= Initial proposal ============= #
    """Whilst for this example with simple distributions the proposal is fixed, in more complex
    models (VAE) the proposal can be conditioned on data (q(z|x))
    """ 
    #mu0 = torch.zeros(1, 2)
    #var0 = torch.ones(1, 2)*0.1

    cyclic_proposal(args.distribution, args.cycles, args.niters)

    make_gif('figs/cycle/{}/'.format(args.distribution))


