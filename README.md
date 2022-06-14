# HMC Tuning

This repository contains a Python package for running HMC with Pytorch, including automatic optimization of its hyperparameters. You will be able to i) sample, by means of HMC, from any target distribution, given as a complex unnormalized target, and ii) automatically tune the HMC hyperparameters to improve the efficiency in exploring the density.

The hyperparameters are:
* Step sizes $\mathbf{\epsilon}$. Matrix with dims $(T,D)$. Different step sizes can be learned to be applied within each state of the chains.
* Momentum variances, $M$. Matrix with dims $(T, D)$.
* An inflation/scale parameter $\mathbf{s}$, that can be a scalar or a vector with dims $D$ so that different inflations can be applied per dimension.
* If not defined, the Gaussian proposal parameters $\mathbf{\mu}$ and $\mathbf{\Sigma}$ can also be tuned.

For further details about the algorithm, see Section 3.5 of [our paper](https://arxiv.org/pdf/2202.04599.pdf), where we adapted the HMC tuning for a Hierarchical VAE. Original idea can be found [here](https://proceedings.mlr.press/v139/campbell21a.html). If you refer to this algorithm, please consider citing both works. If you use this code, please cite:
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
In this section you can find details about the usage of the package. Some useful examples are included in <code>examples</code>. An HMC instance can be created using 
```
hmc = HMC( dim, logp, T,  L, chains, chains_sksd, mu0, var0)
```
where:
* <code>dim</code> is the dimension of the target space.
* <code>logp</code> is a <code>Callable</code> (function) that returns the probability $\log p(\mathbf{x})$ for an input $\mathbf{x}$.
* <code>T</code> is the length if the chains.
* <code>L</code> is the number of Leapfrog steps.
* <code>chains</code> is the number of parallel chains used for each optimization step.
* <code>chains_sksd</code> is the number of parallel chains used independently for computing the SKSD discrepancy within each optimization step.

### Sampling
For sampling from the HMC object, just call:
```
samples, chains = hmc.sample(mu0, var0, chains)
```
where <code>mu0</code> and <code>var0</code> are the initial proposal

### Training
The project is developed in the recent research framework [PyTorch Lightning](https://www.pytorchlightning.ai/). The HH-VAEM model is implemented as a [<code>LightningModule</code>](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html) that is trained by means of a [<code>Trainer</code>](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html). A model can be trained by using:
```
# Example for training HH-VAEM on Boston dataset
python train.py --model HHVAEM --dataset boston --split 0
```
This will automatically download the <code>boston</code> dataset, split in 10 train/test splits and train HH-VAEM on the training split <code>0</code>. Two folders will be created: <code>data/</code> for storing the datasets and <code>logs/</code> for model checkpoints and [TensorBoard](https://www.tensorflow.org/tensorboard) logs. The variable <code>LOGDIR</code> can be modified in <code>src/configs.py</code> to change the directory where these folders will be created (this might be useful for avoiding overloads in network file systems).

The following datasets are available:
- A total of 10 UCI datasets: <code>avocado</code>, <code>boston</code>, <code>energy</code>, <code>wine</code>, <code>diabetes</code>, <code>concrete</code>, <code>naval</code>, <code>yatch</code>, <code>bank</code> or <code>insurance</code>.
- The MNIST datasets: <code>mnist</code> or <code>fashion_mnist</code>.
- More datasets can be easily added to <code>src/datasets.py</code>.
 
For each dataset, the corresponding parameter configuration must be added to <code>src/configs.py</code>.

The following models are also available (implemented in <code>src/models/</code>):
- <code>HHVAEM</code>: the proposed model in the paper.
- <code>VAEM</code>: the VAEM strategy presented in [(Ma et al., 2020)](https://arxiv.org/pdf/2006.11941.pdf) with Gaussian encoder (without including the
Partial VAE).
- <code>HVAEM</code>: A Hierarchical VAEM with two layers of latent variables and a Gaussian encoder.
- <code>HMCVAEM</code>: A VAEM that includes a tuned HMC sampler for the true posterior.
- For MNIST datasets (non heterogeneous data), use <code>HHVAE</code>, <code>VAE</code>, <code>HVAE</code> and <code>HMCVAE</code>.

By default, the test stage will be executed at the end of the training stage. This can be cancelled with <code>--test 0</code> for manually running the test using:
```
# Example for testing HH-VAEM on Boston dataset
python test.py --model HHVAEM --dataset boston --split 0
```
which will load the trained model to be tested on the <code>boston</code> test split number <code>0</code>. Once all the splits are tested, the average results can be obtained using the script in the <code>run/</code> folder:
```
# Example for obtaining the average test results with HH-VAEM on Boston dataset
python test_splits.py --model HHVAEM --dataset boston
```
### Experiments

<p align="center">
  <img width="500" src="imgs/hmc.png">
</p>

The experiments in the paper can be executed using:
```
# Example for running the SAIA experiment with HH-VAEM on Boston dataset
python active_learning.py --model HHVAEM --dataset boston --method mi --split 0

# Example for running the OoD experiment using MNIST and Fashion-MNIST as OoD:
python ood.py --model HHVAEM --dataset mnist --dataset_ood fashion_mnist --split 0
```
Once this is executed on all the splits, you can plot the SAIA error curves or obtain the average OoD metrics using the scripts in the <code>run/</code> folder:
```
# Example for running the SAIA experiment with HH-VAEM on Boston dataset
python active_learning_plots.py --models VAEM HHVAEM --dataset boston

# Example for running the OoD experiment using MNIST and Fashion-MNIST as OoD:
python ood_splits.py --model HHVAEM --dataset mnist --dataset_ood fashion_mnist
```

<br>
<p align="center">
  <img width="900" src="imgs/saia_curves.png">
</p>
<br>

### Help
Use the <code>--help</code> option for documentation on the usage of any of the mentioned scripts. 

## Contributors
[Ignacio Peis](http://www.tsc.uc3m.es/~ipeis/index.html) <br>
[Chao Ma](https://chao-ma.org/) <br>
[José Miguel Hernández-Lobato](https://jmhl.org/) <br>

## Contact
For further information: <a href="mailto:ipeis@tsc.uc3m.es">ipeis@tsc.uc3m.es</a>
