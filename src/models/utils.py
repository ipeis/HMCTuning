import torch
import torch.nn as nn


# ============= Extra functions ============= #

def reparameterize(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """
    Reparameterized samples from a Gaussian distribution

    Args:
        mu (torch.Tensor): means                    (batch_size, ..., dim)
        var (torch.Tensor): variances               (batch_size, ..., dim)

    Returns:
        torch.Tensor: samples                       (batch_size, ..., dim)
    """
    std = var**0.5
    eps = torch.randn_like(std)
    return mu + eps*std

def deactivate(model: nn.Module):
    """
    Freeze or deactivate gradients of all the parameters in a module

    Args:
        model (nn.Module): module to deactivate
    """
    for param in model.parameters():
        param.requires_grad = False

def activate(model):
    """
    Activate gradients of all the parameters in a module

    Args:
        model (nn.Module): module to activate
    """
    for param in model.parameters():
        param.requires_grad = True

def print_parameters(model: nn.Module):
    """
    Print all the parameters in a module

    Args:
        model (nn.Module): module
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print((name, param.data))

class View(nn.Module):
    """
    Reshape tensor inside Sequential objects. Use as: nn.Sequential(...,  View(shape), ...)
    """
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class BinarizeMNIST(object):
    """
    Transform MNIST images into binary
    """
    def __init__(self, threshold=0.5):
        super(BinarizeMNIST, self).__init__()
        self.threshold = threshold

    def __call__(self, tensor):
        return (tensor>self.threshold).type(tensor.dtype)


def get_arch(dim_x: int, latent_dim: int, arch_name='base', dim_h=256):
    """
    Get NNs for the params of q(z|x), p(x|z) 

    Args:
        dim_x (int): dimension of the input data
        latent_dim (int): dimension of the latent space
        arch_name (str, optional): name of the architecture. Defaults to 'base'.
        dim_h (int, optional): dimension of the hidden units. Defaults to 256.

    Returns:
        torch.nn.Sequential: encoder    q(z|x)
        torch.nn.Sequential: decoder    p(x|z)
    """
    if arch_name=='base':
        encoder = nn.Sequential(nn.Linear(dim_x, dim_h), nn.ReLU(), nn.Linear(dim_h, 2 * latent_dim))
        decoder = nn.Sequential(nn.Linear(latent_dim, dim_h), nn.ReLU(), nn.Linear(dim_h, dim_x))

    # You can add more complex architectures here

    return encoder, decoder


    