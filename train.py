import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from trainer import GTA
from utils import transform_n_normalize

def init():
    # For reproducibility
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='digits')
    parser.add_argument('--output-root', default='results')
    params = parser.parse_args()

    params.model_dir = Path(params.output_root) / 'models'
    params.vis_dir = Path(params.output_root) / 'visualization'

    params.n_epochs = 100
    params.batch_size = 128
    params.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(params.device))

    params.n_classes = 10
    params.adv_weight = 0.1
    params.alpha = 0.3
    params.lr = 1e-4

    params.n_g_ft = 64
    params.n_d_ft = 64
    params.n_z_dim = 512

    return params

def main(params):
    src_train_root = Path(params.data_root) / 'svhn' / 'trainset'
    src_val_root = Path(params.data_root) / 'svhn' / 'testset'
    tgt_train_root = Path(params.data_root) / 'mnist' / 'trainset'

    src_train_dataset = ImageFolder(src_train_root, transform=transform_n_normalize)
    src_val_dataset = ImageFolder(src_val_root, transform=transform_n_normalize)
    tgt_train_dataset = ImageFolder(tgt_train_root, transform=transform_n_normalize)

    src_train_loader = DataLoader(src_train_dataset, batch_size=params.batch_size,
        shuffle=True, num_workers=8, drop_last=True)
    src_val_loader = DataLoader(src_val_dataset, batch_size=params.batch_size,
        shuffle=False, num_workers=8)
    tgt_train_loader = DataLoader(tgt_train_dataset, batch_size=params.batch_size,
        shuffle=True, num_workers=8, drop_last=True)

    gta = GTA(params, src_train_loader, src_val_loader, tgt_train_loader)

    gta.train()

if __name__ == '__main__':
    init()

    params = set_params()

    # Create output directory
    params.model_dir.mkdir(parents=True, exist_ok=True)
    params.vis_dir.mkdir(parents=True, exist_ok=True)

    main(params)
    