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


def all_models(data,CNN1,CNN2,CNN3,CNN4):
  for images,labels in data:
    for element, lab in zip(images,labels):
      res=""
      element=element.unsqueeze(0)
      out1=torch.sigmoid(CNN1(element.cuda()))
      #print(out1, lab)
      if out1[0]>0.5:
        #res=("Lung")
        out2=torch.sigmoid(CNN2(element.cuda()))
        if out2[0]<0.5:
          res= "Lung: benign"
        else:
          #res = "Lung: malignant"
          out4=torch.sigmoid(CNN4(element.cuda()))
          #print(out4[0],lab)
          if out4[0]<0.5:
            res = "Lung: malignant: aca"
          else:
           res = "Lung: malignant: scc"
      else:
        #print("colon")
        out3=torch.sigmoid(CNN3(element.cuda()))
        if out3[0]<0.5:
          res = "Colon: benign"
        else:
          res = "Colon: malignant"
      print(res)
      
      
def single_image_checker(img_path,CNN1,CNN2,CNN3,CNN4):
      element = image_loader(sample_image_path)
      res=""
      element=element.unsqueeze(0)
      out1=torch.sigmoid(CNN1(element.cuda()))
      #print(out1, lab)
      if out1[0]<1:
        #res=("Lung")
        out2=torch.sigmoid(CNN2(element.cuda()))
        if out2[0]<0.5:
          res= "Lung: benign"
        else:
          #res = "Lung: malignant"
          out4=torch.sigmoid(CNN4(element.cuda()))
          #print(out4[0],lab)
          if out4[0]<0.5:
            res = "Lung: malignant: aca"
          else:
           res = "Lung: malignant: scc"
      else:
        #print("colon")
        out3=torch.sigmoid(CNN3(element.cuda()))
        if out3[0]<0.5:
          res = "Colon: benign"
        else:
          res = "Colon: malignant"
      print(res)
      
      
def image_loader(image_name):
    """
    Load image and prepares it for model.
    
    Args:
        image_name: string representing path of image location
    Returns:
        image.cuda: CUDA tensor
    """
    image = Image.open(image_name)
    image = data_transform2(image).float()
    image = image.unsqueeze(0)
    image = Variable(image)
    return image.cuda()

  
def predict(net, model_path, sample_image_path):
    """
    Predicts output given a trained neural net model and an image.

    Args:
        net -> PyTorch neural network object
        model_path -> return value from get_model_name()
        sample_image_path -> return value from image_loader()

    Returns:
        tensor: CUDA tensor object

    >>> net = CNN2_LungClassifier()
    >>> model_path = get_model_name(net.name, batch_size=150, learning_rate=0.01, epoch=9)
    >>> img_path = '/content/CNN2_LungClassifierData/test/benign/lungn1019.jpeg'
    >>> predict(net, model_path, img_path)

    Console: 
        tensor([0.0005], device='cuda:0', grad_fn=<SigmoidBackward>) -> Prediction: Class 0 (Benign)
    """
    state = torch.load(model_path)
    net.load_state_dict(state)
    net = net.cuda()
    img = image_loader(sample_image_path)
    x = net(img)
    return torch.sigmoid(x)
  
  
def Model_Loader(net, model_path):
    state=torch.load(model_path)
    net.load_state_dict(state)
    net = net.cuda()
    return net
  
  
def connect_accuracy(data,CNN1,CNN2,CNN3,CNN4,print_img=False):
  total=0
  corr=0
  false_n_p=np.array([["pred |label","ca","cn","la","ln","ls"],
                    ["colon_aca", 0, 0, 0, 0, 0],
                    ["colon_n. ", 0, 0, 0, 0, 0],
                    ["lung_aca ", 0, 0, 0, 0, 0],
                    ["lung_n.  ", 0, 0, 0, 0, 0],
                    ["lung_scc ", 0, 0, 0, 0, 0]
                    ])

  for images,labels in data:
    for element, lab in zip(images,labels):
      #print("Lab: "+str(lab))
      lab=int(lab)
      label_string=''
      pred_string=''
      if print_img==True:  
        p=element.numpy()
        print_plot=plt.figure()
        p=np.transpose(p,(1,2,0))
        plt.imshow(p)
      classes = ['colon_aca', 'colon_n', 'lung_aca','lung_n', 'lung_scc']
      element=element.unsqueeze(0)
      out1=torch.sigmoid(CNN1(element.cuda()))
 
      if out1[0]>0.5:
        out2=torch.sigmoid(CNN2(element.cuda()))

        if out2[0]<0.5:
          res= 3
        else:
          out4=torch.sigmoid(CNN4(element.cuda()))
          if out4[0]<0.5:
            res = 2
          else:
            res = 4
      else:
        out3=torch.sigmoid(CNN3(element.cuda()))
        if out3[0]<0.5:
          res = 0
        else:
          res = 1

      label_string=classes[lab]
      pred_string=classes[res]
      if print_img==True:
        plt.title('Label: '+label_string+" | Pred: "+pred_string)
      false_n_p[res+1,lab+1]=int(false_n_p[res+1,lab+1])+1
      total+=1
      if res==lab:
          corr+=1
  print(false_n_p)
  print("correct:",corr,", total:",total,", accuracy:",corr/total)  
  
