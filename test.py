import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from tester import GTA
from utils import transform_n_normalize

def main(params):
    tgt_test_root = Path(params.data_root) / 'mnist' / 'testset'

    tgt_test_dataset = ImageFolder(tgt_test_root, transform=transform_n_normalize)

    params.n_samples = len(tgt_test_dataset)

    tgt_test_loader = DataLoader(tgt_test_dataset, batch_size=params.batch_size,
        shuffle=False, num_workers=8)

    gta = GTA(params, tgt_test_loader)

    gta.test()

if __name__ == '__main__':
    # For reproducibility
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='digits')
    parser.add_argument('--net-f-path', default='results/models/net_F_best.pth')
    parser.add_argument('--net-c-path', default='results/models/net_C_best.pth')
    params = parser.parse_args()

    params.batch_size = 128
    params.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(params.device))

    params.n_classes = 10
    params.n_d_ft = 64

    main(params)
    