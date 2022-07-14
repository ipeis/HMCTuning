import torch
import torch.nn as nn
from src.models import *

def load_model(model: str, path: str, device: str) -> object:
    """
    Load model into device from a given path

    Args:
        model (str): name of the model
        path (str): path to the model checkpoint
        device (str): 'cpu' or 'cuda'

    Returns:
        object: loaded model
    """

    if model=='VAE':
        model = VAE.load_from_checkpoint(path, device=device).eval().to(device)
    elif model=='HMCVAE':
        model = HMCVAE.load_from_checkpoint(path, device=device).eval().to(device)
    return model
