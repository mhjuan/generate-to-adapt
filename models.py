import torch
import torch.nn as nn

class NetF(nn.Module):
    """ Feature extraction network """

    def __init__(self, params):
        super().__init__()

        self.n_d_ft = params.n_d_ft

        self.model = nn.Sequential(
            nn.Conv2d(3, self.n_d_ft, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(self.n_d_ft, self.n_d_ft, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(self.n_d_ft, 2 * self.n_d_ft, 5, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.model(x)

        return y.view(-1, 2 * self.n_d_ft)

class NetG(nn.Module):
    """ Generator network """

    def __init__(self, params):
        super().__init__()
        
        self.n_fout_dim = 2 * params.n_d_ft
        self.n_g_ft = params.n_g_ft
        self.n_z_dim = params.n_z_dim
        self.device = params.device
        self.n_classes = params.n_classes
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(self.n_fout_dim + self.n_z_dim + self.n_classes + 1,
                8 * self.n_g_ft, 2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(8 * self.n_g_ft),
            nn.ReLU(),

            nn.ConvTranspose2d(8 * self.n_g_ft, 4 * self.n_g_ft, 4,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * self.n_g_ft),
            nn.ReLU(),

            nn.ConvTranspose2d(4 * self.n_g_ft, 2 * self.n_g_ft, 4,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * self.n_g_ft),
            nn.ReLU(),

            nn.ConvTranspose2d(2 * self.n_g_ft, self.n_g_ft, 4,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.n_g_ft),
            nn.ReLU(),

            nn.ConvTranspose2d(self.n_g_ft, 3, 4,
                stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):   
        batch_size = x.size(0)

        x = x.view(-1, self.n_fout_dim + self.n_classes + 1, 1, 1)

        noise = torch.randn((batch_size, self.n_z_dim, 1, 1),
            device=self.device, requires_grad=True)

        y = self.model(torch.cat((x, noise), dim=1))

        return y

class NetD(nn.Module):
    """ Discriminator network """

    def __init__(self, params):
        super().__init__()
        
        self.n_d_ft = params.n_d_ft
        self.n_classes = params.n_classes

        self.model = nn.Sequential(
            nn.Conv2d(3, self.n_d_ft, 3, 1, 1),
            nn.BatchNorm2d(self.n_d_ft),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(self.n_d_ft, 2 * self.n_d_ft, 3, stride=1, padding=1),         
            nn.BatchNorm2d(2 * self.n_d_ft),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(2 * self.n_d_ft, 4 * self.n_d_ft, 3, stride=1, padding=1),           
            nn.BatchNorm2d(4 * self.n_d_ft),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(4 * self.n_d_ft, 2 * self.n_d_ft, 3, stride=1, padding=1),           
            nn.BatchNorm2d(2 * self.n_d_ft),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(4, stride=4)           
        )

        self.classifier_cls = nn.Linear(2 * self.n_d_ft, self.n_classes)
        self.classifier_data = nn.Sequential(
        	nn.Linear(2 * self.n_d_ft, 1),
        	nn.Sigmoid()
        )

    def forward(self, x):
        features = self.model(x)

        features = features.view(-1, 2 * self.n_d_ft)

        y_data = self.classifier_data(features)
        y_cls = self.classifier_cls(features)

        return y_data.view(-1), y_cls

class NetC(nn.Module):
    """ Classifier network """

    def __init__(self, params):
        super().__init__()

        self.n_d_ft = params.n_d_ft
        self.n_classes = params.n_classes

        self.model = nn.Sequential(          
            nn.Linear(2 * self.n_d_ft, 2 * self.n_d_ft),
            nn.ReLU(),
            nn.Linear(2 * self.n_d_ft, self.n_classes),                         
        )

    def forward(self, x):       
        y = self.model(x)

        return y