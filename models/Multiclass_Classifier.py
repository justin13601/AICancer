import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim

torch.manual_seed(1)


class Multiclass_Classifier(nn.Module):
    def __init__(self):
        super(Multiclass_Classifier, self).__init__()
        self.name="CNN3_ColonClassifier"
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 9, 3),
            nn.Conv2d(9, 16, 3),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.Conv2d(64, 64, 5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 5),
            nn.Conv2d(128, 128, 3),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(128 * 8 * 8, 1500)
        self.fc2 = nn.Linear(1500, 500)
        self.fc3 = nn.Linear(500, 5)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) 
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.squeeze(1)
        return F.log_softmax(x, dim=1)
