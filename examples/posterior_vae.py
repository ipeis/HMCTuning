# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2022 by Ignacio Peis, UC3M.                                    +
#  All rights reserved. This file is part of the HH-VAEM, and is released under +
#  the "MIT License Agreement". Please see the LICENSE file that should have    +
#  been included as part of this package.                                       +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from pytorch_lightning import data_loader
from src import *
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import torchvision

# ============= ARGS ============= #

parser = argparse.ArgumentParser(description='Train the HMC-VAE model')

parser.add_argument('--gpu', type=int, default=1,
                    help='use gpu via cuda (1) or cpu (0)')
args = parser.parse_args()


# ============= Activate CUDA ============= #
args.cuda = int(args.gpu>0) and torch.cuda.is_available()
args.cuda = args.cuda == True and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
if str(device) == "cuda":
    print('cuda activated')

model_name = 'VAE'

config = {
            'latent_dim': 20,
            'dim_x': 28**2,
            'likelihood': 'bernoulli', 
            'batch_size': 100,
            'epochs': 93,  # for 93*540=50e3 steps
            'pre_steps': 45e3,
            'T': 10,
        },

dataset = torchvision.datasets.MNIST(train=True, download=True, root='./data')
train_loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'])

epochs = config['epochs']
config.pop('epochs')

if __name__ == '__main__':

    model = model = VAE(**config)

    # ============= TRAIN ============= #
    print('Training HMC-VAE on MNIST')
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=args.gpu,
        default_root_dir='/models/',
        logger=TensorBoardLogger(name=model_name, save_dir='models/'),
    )
    trainer.fit(model, train_loader)
