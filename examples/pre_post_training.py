# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from src import *
import torch

import torch.distributions as D
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal as mnormal
from examples.distributions import *
from examples.utils import *


def pre_post_training(distribution, steps=100, T=5, L=5, chains=100, chains_sksd=30, vector_scale=True):

    logp = get_logp(distribution)
    mu0_pre, var0_pre = initial_proposal(distribution)

    # Create the HMC object
    hmc = HMC(dim=2, logp=logp, T=T,  L=L, chains=100, chains_sksd=30, mu0=mu0_pre, var0=var0_pre, vector_scale=vector_scale)

    samples_pre, chains = hmc.sample(chains=1000)

    # Numpy samples for plotting with maplotlib
    samples_pre = samples_pre.detach().numpy()

    hmc.fit(steps=steps)

    samples, chains = hmc.sample(chains=1000)

    # Numpy samples for plotting with maplotlib
    samples = samples.detach().numpy()

    f, ax = plt.subplots(1, 2, figsize=(12, 6))
    [plot_density(distribution, axis) for axis in ax]

    ax[0].scatter(samples_pre[0, :, 0], samples_pre[0, :, 1], marker='*', color='tab:green', alpha=0.3)
    ax[1].scatter(samples[0, :, 0], samples[0, :, 1], marker='*', color='tab:green', alpha=0.3)

    mu0 = hmc.mu0.detach().numpy()
    var0 = torch.exp(hmc.logvar0) * torch.exp(2*hmc.log_inflation)
    var0 = var0.detach().numpy()


    xmin, xmax, ymin, ymax = get_grid_lims(distribution)   
    x1grid = np.linspace(xmin, xmax, 1000)
    x2grid = np.linspace(ymin, ymax, 1000)
    plot_bi_gaussian(mu0_pre, var0_pre, x1grid, x2grid, ax[0])
    plot_bi_gaussian(mu0, var0, x1grid, x2grid, ax[1])

    [axis.set_xlim(xmin, xmax) for axis in ax]
    [axis.set_ylim(ymin, ymax) for axis in ax]

    ax[0].set_title('Before training')
    ax[1].set_title('After training')

    
if __name__ == '__main__':

    pre_post_training('wave')


