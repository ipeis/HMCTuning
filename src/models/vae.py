# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
from torch import nn
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch
import numpy as np
import pytorch_lightning as pl
import time
from src.models.utils import *

# ============= VAE submodules ============= #

class Likelihood(nn.Module):
    """
    Implements the likelihood functions
    """
    def __init__(self, type='bernoulli', variance=0.1):
        """
        Likelihood initialization

        Args:
            type (str, optional): likelihood type ('gaussian', 'categorical', 'loggaussian' or 'bernoulli'). Defaults to 'gaussian'.
            variance (float, optional): fixed variance for gaussian/loggaussian variables. Defaults to 0.1.
        """
        super(Likelihood, self).__init__()
        self.type=type
        self.variance=variance # only for Gaussian
    
    def forward(self, theta: torch.Tensor, data: torch.Tensor, variance=None) -> torch.Tensor:
        """
        Computes the log probability of a given data under parameters theta

        Args:
            theta (torch.Tensor): tensor with params                (batch_size, latent_samples, dim_data)
            data (torch.Tensor): tensor with data                   (batch_size, dim_data)
            observed (torch.Tensor): tensor with observation mask   (batch_size, dim_data)
            variance (float, optional): Gaussian fixed variance (None for using the predefined). Defaults to None.

        Returns:
            torch.Tensor: tensor with the log probs                 (batch_size, latent_samples, dim_data)
        """
        
        # ============= Gaussian ============= #

        # ============= Bernoulli ============= #
        if self.type=='bernoulli':
            data = data.repeat(theta.shape[-2], 1, 1).permute(1, 0, 2)
            logp = -BCEWithLogitsLoss(reduction='none')(theta, data)

        # You can add more likelihoods here
        return logp


class Prior(nn.Module):
    """
    Implements a Prior distribution
    """
    def __init__(self, type='standard'):
        """
        Prior initialization

        Args:
            type (str, optional): prior type. Defaults to 'standard'.
        """
        super(Prior, self).__init__()
        self.type=type
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the log prob of given latent z under the prior

        Args:
            z (torch.Tensor): latent samples                    (B, latent_samples, latent_dim)

        Returns:
            torch.Tensor: tensor with the log probs             (B, latent_samples)
        """
        if self.type=='standard':
            cnt = z.shape[-1] * np.log(2 * np.pi)
            logp = -0.5 * (cnt + torch.sum((z) ** 2, dim=-1))
        return logp

class Decoder(nn.Module):
    """
    Implements a decoder

    """
    def __init__(self, likelihood = 'bernoulli', network: nn.Module=None, variance=0.1):
        """
        Initialization of the decoder

        Args:
            likelihood (str, optional): likelihood type ('gaussian', 'loggaussian', 'categorical', 'bernoulli'). Defaults to 'gaussian'.
            network (nn.Module, optional): module for computing likelihood parameters. Defaults to None.
            variance (float, optional): Gaussian fixed variance. Defaults to 0.1.
        """
        super(Decoder, self).__init__()

        self.decoder = network
        self.likelihood = Likelihood(likelihood, variance=variance)
        self.distribution = likelihood
        self.variance = variance

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the likelihood parameters or logits for the discrete distributions

        Args:
            z (torch.Tensor): tensor with latent samples            (batch_size, latent_samples, latent_dim)

        Returns:
            torch.Tensor: tensor with likelihood parameters         (batch_size, latent_samples, data_dim)
        """

        theta = self.decoder(z)
        if self.distribution=='bernoulli':
            theta = theta
        # You can implement more data types here
        return theta
    
    def logp(self, x: torch.Tensor, z: torch.Tensor=None, theta: torch.Tensor=None, variance: float=None) -> torch.Tensor:
        """
        Computes the log probability of given x under likelihood parameterized by theta

        Args:
            x (torch.Tensor): data tensor                                                                                   (batch_size, x_dim)
            z (torch.Tensor, optional): latent samples (None if theta is given). Defaults to None.                          (batch_size, latent_samples, latent_dim)
            theta (torch.Tensor, optional): likelihood params (None to compute from given z). Defaults to None.             (batch_size, latent_samples, x_dim)
            variance (float, optional): Gaussian variance (None for using the predefined fixed variance). Defaults to None.            

        Returns:
            torch.Tensor: tensor with log probs
        """
        if theta==None:
            # if theta==None z is needed
            theta = self.forward(z)
        logp = self.likelihood(theta, x, variance)
        return logp

class Encoder(nn.Module):
    """
    Implements an encoder

    """
    def __init__(self, network: nn.Module=None):
        """
        Encoder initialization

        Args:
            network (nn.Module, optional): module for computing approx. posterior parameters. Defaults to None.
        """
        super(Encoder, self).__init__()

        self.encoder = network

    def forward(self, x):
        phi = self.encoder(x)
        mu, logvar = torch.chunk(phi, 2, -1)
        return mu, logvar

    def regularizer(self, mu, logvar):
        kl = -0.5 * torch.sum(1. + logvar - mu ** 2 - torch.exp(logvar), dim=-1, keepdim=True)
        return kl
    
    def logq(self, z, ):
        mu_z, logvar_z = self.forward(x)
        mu_z = mu_z.unsqueeze(-2)
        logvar_z = logvar_z.unsqueeze(-2)
        cnt = mu_z.shape[-1] * np.log(2 * np.pi) + torch.sum(logvar_z, dim=-1)
        logqz_x = -0.5 * (cnt + torch.sum((z - mu_z)**2 * torch.exp(-logvar_z), dim=-1))
        return logqz_x


# ============= Vanilla VAE ============= #

class VAE(pl.LightningModule):
    """
    Implements the structure of a vanilla VAE https://arxiv.org/abs/1312.6114

    """
    def __init__(self, 
        dim_x: int, latent_dim = 10, arch='base', dim_h=256,
        likelihood = 'bernoulli', variance=0.1, 
        batch_size=128, lr=1e-3, samples_MC = 1, data_path='../data/'):
        """
        VAE initialization

        Args:
            dim_x (int): input data dimension
            latent_dim (int, optional): dimension of the latent space. Defaults to 10.
            arch (str, optional): name of the architecture for encoder/decoder from the 'archs' file. Defaults to 'base'.
            dim_h (int, optional): dimension of the hidden vectors. Defaults to 256.
            likelihood (str, optional): input data likelihood type. Defaults to 'gaussian'.
            variance (float, optional): fixed variance for Gaussian likelihoods. Defaults to 0.1.
            batch_size (int, optional): batch size. Defaults to 128.
            lr (float, optional): learning rate for the parameter optimization. Defaults to 1e-3.
            samples_MC (int, optional): number of MC samples for computing the ELBO. Defaults to 1.
        """

        super(VAE, self).__init__()

        encoder, decoder = get_arch(dim_x, latent_dim, arch, dim_h=dim_h)

        self.encoder = Encoder(encoder)
        self.decoder = Decoder(likelihood=likelihood, network=decoder, variance=variance)
        self.prior = Prior(type='standard')
        self.dim_x = dim_x
        self.latent_dim = latent_dim
        self.arch = arch
        self.dim_h = dim_h
        self.likelihood = likelihood
        self.variance = variance
        self.batch_size = batch_size
        self.lr = lr
        self.samples_MC = samples_MC
        self.data_path = data_path
        self.validation=False

        self.save_hyperparameters('dim_x', 'likelihood', 'variance',
        'latent_dim', 'arch', 'batch_size', 'lr', 'dim_h', 'samples_MC', 'data_path')

    
    def forward(self, batch: tuple, samples: int) -> tuple:
        """
        Computes the mean ELBO for a given batch

        Args:
            batch (tuple): contains data
            samples (int): number of MC samples for computing the ELBO

        Returns:
            torch.Tensor: mean loss (negative ELBO)                          
            torch.Tensor: Reconstruction term for x logp(x|z)           
            torch.Tensor: Reconstruction term for y logp(y|z,x)    
            torch.Tensor: KL term     

        """
        # Get data
        x, _ = batch
        mu_z, logvar_z = self.encoder(x)

        z = self.sample_z(mu_z, logvar_z, samples=samples)
        theta = self.decoder(z)

        rec_x = self.decoder.logp(x, z=z, theta=theta).sum(-1)
        kl = self.encoder.regularizer(mu_z, logvar_z)
        
        elbo = rec_x - kl

        return elbo.mean(), rec_x.mean(), kl.mean()
        
    def training_step(self, batch: tuple, batch_idx: int, logging: bool=True) -> torch.Tensor:
        """
        Returns the loss (negative ELBO) for the minimization

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            batch_idx (int): batch index from the training set
            logging (bool): log metrics into Tensorboard (True). Default True

        Returns:
            torch.Tensor: mean loss (negative ELBO)                                  

        """
        elbo, rec_x, kl = self.forward(batch, samples=self.samples_MC)
        loss = -elbo

        self.log('ELBO', -loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if logging:
            self.log('-rec_x', -rec_x, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('kl', kl, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def preprocess_batch(self, batch: tuple) -> tuple:
        """
        Preprocessing operations for the batch

        Args:
            batch (tuple): contains (data)

        Returns:
            tuple: preprocessed batch, contains (data)
        """        
        return batch

    def sample_z(self, mu: torch.Tensor, logvar: torch.Tensor, samples=1) -> torch.Tensor:
        """
        Draw latent samples from a given approx posterior parameterized by mu and logvar

        Args:
            mu (torch.Tensor): tensor with the means                            (batch_size, latent_dim)
            logvar (torch.Tensor): tensor with the log variances                (batch_size, latent_dim)
            samples (int, optional): number of samples. Defaults to 1.          

        Returns:
            torch.Tensor: latent samples
        """
        # Repeat samples times for Monte Carlo
        mu = mu.repeat(samples, 1, 1).transpose(0, 1)
        logvar = logvar.repeat(samples, 1, 1).transpose(0, 1)
        # Reparametrization
        z = reparameterize(mu, torch.exp(logvar))
        return z

 
    def elbo(self, batch: tuple, samples=1000) -> torch.Tensor:
        """
        Computes the mean ELBO of a batch

        Args:
            batch (tuple): contains (data, observed_data, target, observed_target)
            samples (int): number of samples from the latent for MC. Defaults to 1000.

        Returns:
            torch.Tensor: mean elbo
        """
        elbo, _, _ = self.forward(batch)
        return elbo

    def logp(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Returns the log joint logp(x, z) of the model (unnormalized posterior)

        Args:
            x (torch.Tensor): normalized and preprocessed data                                                                              (batch_size, dim_x)
            z (torch.Tensor): latent samples                                                                                                (batch_size, latent_samples, latent_dim)

        Returns:    
            torch.Tensor: log probs                                                                                                         (batch_size, 1)                        
        """

        theta = self.decoder(z)

        logpx_z = self.decoder.logp(x, theta=theta).sum(-1)
        logpz = self.prior(z)

        logp = logpx_z + logpz

        return logp

    def logp_func(self, x: torch.Tensor):
        """
        Returns a function for computing logp(x, z) with fixed x (only depending on z). This function is used as HMC objective.

        Args:
            x (torch.Tensor): normalized and preprocessed data                                                                              (batch_size, dim_x)

        Returns:
            function depending on z ( logp(z, x) for fixed x )
        """
        def logp(z):
            return self.logp(x, z)

        return logp

   
    # ============= Modified PL functions ============= #
    def configure_optimizers(self):
        opt = torch.optim.Adam(list(self.parameters()), lr=self.lr, weight_decay=0.001)  
        return opt

  
