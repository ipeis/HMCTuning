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
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from torchvision import datasets, transforms


# ============= ARGS ============= #

parser = argparse.ArgumentParser(description='Train the HMC-VAE model')

parser.add_argument('--gpu', type=int, default=0,
                    help='use gpu via cuda (1) or cpu (0)')
args = parser.parse_args()

config = {
            'latent_dim': 20,
            'dim_x': 28**2,
            'dim_h': 512,
            'batch_size': 128,
            'epochs': 107,  # for 107*469~=50e3 steps
            'pre_steps': 45e3,
            'T': 10,
        }

dataset = datasets.MNIST(train=True, download=True, root='./data', 
                        transform=transforms.Compose([
                            transforms.ToTensor(), 
                            View([28**2]), 
                            BinarizeMNIST()])
                            )

train_loader = torch.utils.data.DataLoader( dataset, batch_size=config['batch_size'])

epochs = config['epochs']
config.pop('epochs')

if __name__ == '__main__':

    model = model = HMCVAE(**config)

    # ============= TRAIN ============= #
    print('Training HMC-VAE on MNIST')
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=args.gpu,
        default_root_dir='../models/',
        logger=TensorBoardLogger(name='HMCVAE', save_dir='./models/'),
    )
    trainer.fit(model, train_loader)
