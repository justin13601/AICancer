import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim

torch.manual_seed(1)


class CNN4_LungMalignant(nn.Module):
    def __init__(self):
        super(CNN4_LungMalignant, self).__init__()
        self.name = 'CNN4_LungMalignant'
        self.layer_1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1, padding=1)
        self.layer_2 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=4, stride=2, padding=1)
        self.layer_3 = nn.Conv2d(in_channels=15, out_channels=12, kernel_size=4, stride=2, padding=1)
        self.layer_a = nn.Conv2d(in_channels=12, out_channels=10, kernel_size=4, stride=2, padding=1)
        self.pooling_1 = nn.MaxPool2d(2, 2)
        self.layer_4 = nn.Linear(in_features=1690, out_features=200)
        self.layer_5 = nn.Linear(in_features=200, out_features=1)

    def forward(self, data):
        lay1 = torch.relu(self.layer_1(data))
        lay2 = torch.relu(self.layer_2(lay1))
        lay3 = torch.relu(self.layer_3(lay2))
        lay_a = torch.relu(self.layer_a(lay3))
        pool1 = self.pooling_1(lay_a)
        fully_connected = pool1.view(-1, 1690)
        lay4 = torch.relu(self.layer_4(fully_connected))
        lay5 = self.layer_5(lay4)
        lay5 = lay5.squeeze(1)
        return lay5
