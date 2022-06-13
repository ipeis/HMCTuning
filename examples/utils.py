
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mnormal
from PIL import Image
import glob
import os

def plot_bi_gaussian(mu, Sigma, x1grid, x2grid, ax):
    """Plot a multivariate Gaussian with parameters mu and Sigma in the region given by x1grid and x2grid, using axes ax.
    `mu`: mean, vector of dimension D
    `Sigma`: covariance matrix of dimension DxD.
    `x1grid`: region for first dimension, vector of any dimension N.
    `x2grid`: region for second dimension, vector with same dimension as x1grid.
    `ax`: ax object to plot.
    """

    mu = mu.squeeze()
    if len(Sigma.shape) == 1:
        Sigma = np.diag(Sigma)
    else:
        if Sigma.shape[0] == 1:
            Sigma = np.diag(Sigma.squeeze())

    X1, X2 = np.meshgrid(x1grid, x2grid)
    # Pack X and Y into a single 3-dimensional array
    X = np.empty(X1.shape + (2,))
    X[:, :, 0] = X1
    X[:, :, 1] = X2
    p = mnormal.pdf(X, mu.squeeze(), Sigma)
    ax.contour(X1, X2, p, cmap = plt.cm.Blues, alpha=0.5)




def make_gif(folder, duration=20, delete=True):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{folder}/*.png"))]
    frame_one = frames[0]
    frame_one.save("{}/samples.gif".format(folder), format="GIF", append_images=frames,
               save_all=True, duration=duration, loop=0)
    if delete:
        [os.remove(image) for image in sorted(glob.glob(f"{folder}/*.png"))]
