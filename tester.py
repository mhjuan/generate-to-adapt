import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import models
import utils

class GTA():
    def __init__(self, params, test_loader):
        self.device = params.device
        self.batch_size = params.batch_size
        self.n_classes = params.n_classes
        self.n_samples = params.n_samples

        self.test_loader = test_loader
        
        # Define networks
        self.net_F = models.NetF(params)
        self.net_C = models.NetC(params)

        # Load trained weights
        self.net_F.load_state_dict(torch.load(params.net_f_path))
        self.net_C.load_state_dict(torch.load(params.net_c_path))

        # Define optimizers
        self.net_F.to(params.device)
        self.net_C.to(params.device)

    def test(self):
        self.net_F.eval()
        self.net_C.eval()

        tot_corrects = 0
    
        for i, data in enumerate(self.test_loader):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.net_C(self.net_F(inputs))

            _, predicts = torch.max(outputs, 1)

            tot_corrects += torch.sum(predicts == labels).item()

        test_acc = tot_corrects / self.n_samples

        print('Test acc: {:.3f}'.format(test_acc))
            