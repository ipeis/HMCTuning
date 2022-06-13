# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir("..")
from src import *
import argparse
import torch

import torch.distributions as D
import matplotlib.pyplot as plt
import numpy as np

# ============= ARGS ============= #

parser = argparse.ArgumentParser(description='Train the hyperparameters of HMC using Variational Inference')

parser.add_argument('--distribution', type=str, default='wave', 
                    help='name of the distribution (from srd/distributions.py)')
parser.add_argument('--T', type=int, default=10, 
                    help='length of the HMC chains (number of states)')
parser.add_argument('--L', type=int, default=5, 
                    help='Number of Leapfrog steps within each state update')
parser.add_argument('--chains', type=int, default=100, 
                    help='Number of parallel chains (n. of HMC samples)')
parser.add_argument('--chains_sksd', type=int, default=30, 
                    help='Number of parallel chains for computing the SKSD')    
parser.add_argument('--steps', type=int, default=1000, 
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


if __name__ == '__main__':

    logp = get_logp(args.distribution)

    #samples = gmm.sample([1000]).numpy()
    #plt.scatter(samples[:, 0], samples[:, 1])

    # ============= Initial proposal ============= #
    """Whilst for this example with simple distributions the proposal is fixed, in more complex
    models (VAE) the proposal can be conditioned on data (q(z|x))
    """ 
    mu0 = torch.zeros(2)
    var0 = torch.ones(2)*0.1
    hmc = HMC(dim=2, logp=logp, T=args.T,  L=args.L, chains=args.chains, chains_sksd=args.chains_sksd)

    # ============= TRAIN ============= #
    print('Training HMC hyperparameters')
    hmc.fit(steps=args.steps)
