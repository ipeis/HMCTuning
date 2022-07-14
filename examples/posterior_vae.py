# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os, sys
from random import shuffle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir("..")
from src import *
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt

# ============= ARGS ============= #

parser = argparse.ArgumentParser(description='Train the HMC-VAE model')

parser.add_argument('--gpu', type=int, default=0,
                    help='use gpu via cuda (1) or cpu (0)')
args = parser.parse_args()

# ============= Activate CUDA ============= #
args.cuda = int(args.gpu>0) and torch.cuda.is_available()
args.cuda = args.cuda == True and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
if str(device) == "cuda":
    print('cuda activated')



dataset = datasets.MNIST(train=False, download=True, root='./data', 
                        transform=transforms.Compose([
                            transforms.ToTensor(), 
                            View([28**2]), 
                            BinarizeMNIST()])
                            )

test_loader = torch.utils.data.DataLoader( dataset, batch_size=128, shuffle=False)

if __name__ == '__main__':

    model = load_model('HMCVAE', path='models/HMCVAE/version_0/checkpoints/epoch=106-step=61068.ckpt', device=device) 
    
    #for sample_idx in range(100):
    sample_idx = 89
    batch = iter(test_loader).next()
    x = batch[0][sample_idx].unsqueeze(0).to(device)

    # Define the HMC objective
    model.HMC.logp = model.logp_func(x) 

    # Encoder again for not sharing gradients
    mu_z, logvar_z = model.encoder(x)

    # Unnormalized posterior
    npoints = 1000
    z1lims = mu_z[..., 0][0] - 0.5, mu_z[..., 0][0] + 0.5
    z2lims = mu_z[..., 1][0] - 0.5, mu_z[..., 1][0] + 0.5
    z1 = torch.linspace(float(z1lims[0].detach().cpu().numpy()), float(z1lims[1].detach().cpu().numpy()), npoints)
    z2 = torch.linspace(float(z2lims[0].detach().cpu().numpy()), float(z2lims[1].detach().cpu().numpy()), npoints)
    Z1, Z2 = torch.meshgrid(z1, z2, indexing="ij")
    ZGRID = torch.stack([Z1.ravel(), Z2.ravel()], -1).unsqueeze(0).to(device)
    logp=model.logp(x, ZGRID).reshape(npoints, npoints)
    
    # Normalizing constant
    Z = model.elbo_iwae((x, None))
    posterior = torch.exp(logp-Z)

    # Gaussian proposal
    q = torch.distributions.multivariate_normal.MultivariateNormal(mu_z, torch.diag(torch.exp(logvar_z)[0]))
    logq = q.log_prob(ZGRID).reshape(npoints, npoints)
    gaussian_posterior = torch.exp(logq)

    # Samples
    samples = model.sample_z(mu_z, logvar_z, samples=1000)

    # Plotting
    f, ax = plt.subplots(figsize=(4, 4))
    cont1 = ax.contour(Z1.numpy(), Z2.numpy(), posterior.detach().numpy(), 15, cmap='Greens')
    cont1 = ax.contour(Z1.numpy(), Z2.numpy(), gaussian_posterior.detach().numpy(), 15, cmap='Blues')

    alpha = torch.exp(model.logp(x, samples).squeeze())
    alpha = alpha / alpha.max() * 0.8
    
    plt.scatter(samples[0, :, 0].detach().cpu().numpy(), samples[0, :, 1].detach().cpu().numpy(), marker='*', color='tab:orange', alpha=alpha.detach().numpy())

    # This limits are adapted for sample_idx=89, you might change them
    plt.xlim([1.8, 2.7])
    plt.ylim([-2.1, -1.4])
    plt.axis('off')
    #plt.axis('tight')

    plt.savefig('./assets/pdf/posterior_{}.pdf'.format(sample_idx))
    
