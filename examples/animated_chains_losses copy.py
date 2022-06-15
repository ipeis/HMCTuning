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

# ============= ARGS ============= #

parser = argparse.ArgumentParser(description='Train the hyperparameters of HMC using Variational Inference')

parser.add_argument('--distribution', type=str, default='wave', 
                    help='name of the distribution (from srd/distributions.py)')
parser.add_argument('--T', type=int, default=5, 
                    help='length of the HMC chains (number of states)')
parser.add_argument('--L', type=int, default=5, 
                    help='number of Leapfrog steps within each state update')
parser.add_argument('--chains', type=int, default=1000, 
                    help='number of parallel chains (n. of HMC samples)')
parser.add_argument('--chains_sksd', type=int, default=30, 
                    help='number of parallel chains for computing the SKSD')      
parser.add_argument('--steps', type=int, default=50, 
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


if not os.path.isdir('./figs/{}'.format(args.distribution)):
    os.makedirs('./figs/{}'.format(args.distribution))

def plot_chains(distribution, steps, samples_per_step=100, chains_per_step=10):
    
    f = plt.figure(constrained_layout=True)
    gs = f.add_gridspec(ncols=6, nrows=3)
    ax = []
    ax.append(f.add_subplot(gs[:, :3]))     # Samples
    ax.append(f.add_subplot(gs[0, 3:5]))    # Objective
    ax.append(f.add_subplot(gs[1, 3:5]))    # SKSD
    ax.append(f.add_subplot(gs[2, 3:5]))    # Inflation
    chain_color='gray'
    sample_color='tab:green'
    ax[0].axis('off')
    # Gaussian mixture
    #plt.xlim(-1.3, 1.3) 
    #plt.ylim(-1.3, 1.3) 
    # Dual moon
    #plt.xlim(-2, 2) 
    #plt.ylim(-2, 2) 
    if distribution=='wave':
        ax[0].set_xlim(-10, 10) 
        ax[0].set_ylim(-10, 10) 

    plt.suptitle('step 0')
    
    def reset_axis(ax, step, samples=None):
        ax.clear()
        ax.axis('off')
        # Gaussian mixture
        #plt.xlim(-1.3, 1.3) 
        #plt.ylim(-1.3, 1.3) 
        # Dual moon
        #plt.xlim(-2, 2) 
        #plt.ylim(-2, 2) 
        if distribution=='wave':
            ax.set_xlim(-10, 10) 
            ax.set_ylim(-10, 10) 
        if len(samples) != 0:
            x = torch.stack(samples)[-3*samples_per_step:, 0].detach().numpy()
            y = torch.stack(samples)[-3*samples_per_step:, 1].detach().numpy()
            alphas = np.ones_like(x)
            alphas[:-samples_per_step] = np.linspace(0, 1, len(alphas[:-samples_per_step]))
            #alphas = np.linspace(0.01, 1, len(samples))
            #alphas =  np.floor(alphas / (1.0/(1+int(len(samples) / (samples_per_step))))) 
            #alphas = np.clip(alphas, 0.1, 1)
            ax.scatter(x, y, marker='*', color=sample_color, alpha=alphas)
            plt.title('step {}'.format(step))
        

    logp = get_logp(distribution)
    mu0, var0 = initial_proposal(distribution)

    hmc = HMC(dim=2, logp=logp, T=args.T,  L=args.L, chains=args.chains, chains_sksd=args.chains_sksd, mu0=mu0, var0=var0)

    hmc.optimizer = torch.optim.Adam(hmc.parameters(), lr=0.01)

    loss=[]
    objective=[]
    sksd=[]
    inflation=[]
    samples = []
    progress = trange(steps, desc='Loss')
    im=0
    for e in progress:

        # Sample 
        z, chains = hmc.sample(mu0=hmc.mu0, var0=torch.exp(hmc.logvar0), chains=samples_per_step)
        chains = chains[:, 0, :chains_per_step, :].reshape(hmc.T+1, chains_per_step, 2)
        z = z.reshape(samples_per_step, 2)
        alphas = np.linspace(0.1, 1, chains.shape[0])[::-1]
        for c in range(chains.shape[1]):
            for t in range(1,chains.shape[0]+1):
                reset_axis(ax[0], e, samples)
                ax[0].plot(chains[:t,c,0].detach().numpy(), chains[:t,c,1].detach().numpy(), marker='o', color=chain_color, alpha=alphas[t-1])
                plt.savefig('figs/{}/{:05d}.png'.format(distribution, im))
                im+=1
            ax[0].plot(chains[-1, c, 0].detach().numpy(), chains[-1, c, 1].detach().numpy(), marker='*', color=sample_color)
            plt.savefig('figs/{}/{:05d}.png'.format(distribution, im))
            im+=1
            samples.append(chains[-1, c])
            reset_axis(ax[0], e, samples)
            plt.savefig('figs/{}/{:05d}.png'.format(distribution, im))
            im+=1
        samples = samples + list(z)
        hmc.optimizer.zero_grad()

        hmc.optimizer.zero_grad()
        z, _ = hmc.sample(hmc.mu0, torch.exp(hmc.logvar0), hmc.chains)
        _loss = -hmc.logp(z)

        _loss[torch.isfinite(_loss)].sum().backward()
        hmc.optimizer.step()

        hmc.optimizer.zero_grad()
        _sksd = hmc.evaluate_sksd(hmc.mu0, torch.exp(hmc.logvar0))
        _sksd.backward()
        hmc.optimizer.step()


        progress.set_description('HMC (objective=%g)' % -_loss[torch.isfinite(_loss)].mean().detach().numpy())

        loss.append(_loss[torch.isfinite(_loss)].detach().numpy().mean())
        objective.append(-loss[-1])
        sksd.append(_sksd.detach().numpy())
        inflation.append(torch.exp(hmc.log_inflation.clone()).detach().numpy())

        ax[1].plot(objective, color='tab:blue')
        ax[1].set_title(r'$\log p(x)$')
        ax[2].plot(sksd, color='tab:red')
        ax[2].set_title(r'$\text{SKSD}')
        ax[3].plot(inflation, color='tab:green')

    reset_axis(ax[0], e, [])
    ax[0].scatter(torch.stack(samples)[-samples_per_step:, 0].detach().numpy(), torch.stack(samples)[-samples_per_step:, 1].detach().numpy(), marker='*', color=sample_color, alpha=0.5)
    plt.title('step {}'.format(e))
    plt.savefig('figs/{}/samples.pdf'.format(distribution, im))

def make_gif(folder, delete=True):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{folder}/*.png"))]
    frame_one = frames[0]
    frame_one.save("{}/samples.gif".format(folder), format="GIF", append_images=frames,
               save_all=True, duration=20, loop=0)
    if delete:
        [os.remove(image) for image in sorted(glob.glob(f"{folder}/*.png"))]


if __name__ == '__main__':

    # ============= Initial proposal ============= #
    """Whilst for this example with simple distributions the proposal is fixed, in more complex
    models (VAE) the proposal can be conditioned on data (q(z|x))
    """ 
    #mu0 = torch.zeros(1, 2)
    #var0 = torch.ones(1, 2)*0.1

    plot_chains(args.distribution, args.steps, samples_per_step=1000, chains_per_step=3)

    make_gif('figs/{}/'.format(args.distribution))


