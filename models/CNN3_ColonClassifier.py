import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim

torch.manual_seed(1)


class CNN3_ColonClassifier(nn.Module):
    def __init__(self):
        super(CNN3_ColonClassifier, self).__init__()
        self.name = "CNN3_ColonClassifier"
        self.conv1 = nn.Conv2d(3, 9, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(9, 10, 5)

        self.fc1 = nn.Linear(10 * 53 * 53, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, img):
        x = self.pool(F.relu(self.conv1(img)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1)
        return x
