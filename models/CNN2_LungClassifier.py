import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim

torch.manual_seed(1)


class CNN2_LungClassifier(nn.Module):
    def __init__(self):
        super(CNN2_LungClassifier, self).__init__()
        self.name = "CNN2_LungClassifier"
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.conv2 = nn.Conv2d(12, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear_layers = nn.Sequential(
            nn.Linear(10 * 53 * 53, 64),
            nn.Linear(64, 32)
        )
        self.final_fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 53 * 53)
        x = F.relu(self.linear_layers(x))
        x = self.final_fc(x)
        x = x.squeeze(1)
        return x
