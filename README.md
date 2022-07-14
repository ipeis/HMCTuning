# HMCTuning

This repository contains a Python package for running HMC with Pytorch, including automatic optimization of its hyperparameters. You will be able to i) sample from any distribution, given as a unnormalized target, and ii) automatically tune the HMC hyperparameters to improve the efficiency in exploring the density.

For further details about the algorithm, see Section 3.5 of [our paper](https://arxiv.org/pdf/2202.04599.pdf), where we adapted the HMC tuning for improving the inference in  a Hierarchical VAE for mixed-type partial data. Original idea for optimizing HMC via Variational Inference can be found [here](https://proceedings.mlr.press/v139/campbell21a.html). If you refer to this algorithm, please consider citing both works. If you use this code, please cite:
```
@article{peis2022missing,
  title={Missing Data Imputation and Acquisition with Deep Hierarchical Models and Hamiltonian Monte Carlo},
  author={Peis, Ignacio and Ma, Chao and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel},
  journal={arXiv preprint arXiv:2202.04599},
  year={2022}
}
```

## Instalation 
The installation is straightforward using the following instruction, that creates a conda virtual environment named <code>HMCTuning</code> using the provided file <code>environment.yml</code>:
```
conda env create -f environment.yml
```

## Usage
For an extended usage guide, check [<code>notebooks/usage.ipynb</code>](notebooks/usage.ipynb). For a basic usage, continue reading here. An HMC object can be created as in the following example:
```
from examples.distributions import *
from examples.utils import *

# Load the log probability function of MoG, and the initial proposal
logp = get_logp('gaussian_mixture')
mu0, var0 = initial_proposal('gaussian_mixture')   # [0, 0],  [0.01, 0.01]

# Create the HMC object
hmc = HMC(dim=2, logp=logp, T=5,  L=5, chains=1000, chains_sksd=30, mu0=mu0, var0=var0, vector_scale=True)
```

where:
* <code>dim</code> is an <code>int</code> with the dimension of the target space.
* <code>logp</code> is a <code>Callable</code> (function) that returns the log probability $\log p(\mathbf{x})$ for an input $\mathbf{x}$.
* <code>T</code> is an <code>int</code> with the length of the chains.
* <code>L</code> is an <code>int</code> with the number of Leapfrog steps.
* <code>chains</code> is an <code>int</code> with the number of parallel chains used for each optimization step.
* <code>chains_sksd</code> is an <code>int</code> with the number of parallel chains used independently for computing the SKSD discrepancy within each optimization step.
* <code>mu0</code> is a <code>(bath_size, D)</code> tensor with the means of the Gaussian initial proposal.
* <code>var0</code> is a <code>(bath_size, D)</code> tensor with the variances of the Gaussian initial proposal.


### Sampling
For sampling from the created HMC object, just call:
```
samples, chains = hmc.sample(N)
```
Your final <code>N</code> samples will be stored in <code>samples</code>, and, if needed, you can inspect the full <code>chains</code>.

### Training
To train the HMC hyperparameters, call:
```
hmc.fit(steps=100)
```
This will run the gradient-based optimization algorithm that tunes the hyperparameters using Variational Inference.

## Example 1: 2D densities

In the following gifs you can observe two simple examples on how effective is the training algorithm for wave-shaped (left) and dual-mooon densities. Horizontal scaling is automatically increased during training to inflate the proposal for covering the density. 

<p> 
  <img src="assets/gifs/training_wave.gif" width="400" /> 
  &nbsp; &nbsp;
  <img src="assets/gifs/training_dual_moon.gif" width="400" />
</p>

## Example 2: posterior of a VAE

<p> 
  <img src="assets/pdf/posterior_89.png" width="400" /> 
</p>



### Help
Use the <code>--help</code> option for documentation on the usage of any of the mentioned scripts. 

## Contributors
[Ignacio Peis](https://ipeis.github.io/) <br>

## Contact
For further information: <a href="mailto:ipeis@tsc.uc3m.es">ipeis@tsc.uc3m.es</a>
