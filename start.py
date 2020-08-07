import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import *
from torchvision import datasets, models, transforms
from torchsummary import summary
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

from utilities import utility_funcs, evaluate_all, predict_funcs, train_net
from models import CNN1_LungColon, CNN2_LungClassifier, CNN3_ColonClassifier, CNN4_LungMalignant

CNN1 = CNN1_LungColon.CNN1_LungColon()
CNN1 = predict_funcs.model_loader(CNN1, 'checkpoint_files\model_CNN1_LungColon_bs256_lr0.001_epoch14')

CNN2 = CNN2_LungClassifier.CNN2_LungClassifier()
CNN2 = predict_funcs.model_loader(CNN2, 'checkpoint_files\model_CNN2_LungClassifier_bs150_lr0.01_epoch9')

CNN3 = CNN3_ColonClassifier.CNN3_ColonClassifier()
CNN3 = predict_funcs.model_loader(CNN3, 'checkpoint_files\model_CNN3_ColonClassifier_bs256_lr0.001_epoch14')

CNN4 = CNN4_LungMalignant.CNN4_LungMalignant()
CNN4 = predict_funcs.model_loader(CNN4, 'checkpoint_files\model_CNN4_LungMalignant_bs64_lr0.0065_epoch12')

img = 'dataset\demo\lungn10.jpeg'

predict_funcs.classify_image(img, CNN1, CNN2, CNN3, CNN4)
