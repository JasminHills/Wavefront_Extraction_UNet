
import sys
from collections import OrderedDict
sys.path.insert(0, '../pytorch/models/')
sys.path.insert(0, '../pytorch/')
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from collections import OrderedDict
from U1D import U1
from datasetNw import *
import sys
print('here 1')
from lr_analyzer import *
from criterion import *
import optuna
from optuna.trial import TrialState
from U1D import U1
from train1OptGPU import * 
from train1OptGPU import train 
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
# Load dataset


# Load dataset

data_dir = '/Users/jh2619/Desktop/Machine-learning-for-image-based-wavefront-sensing-master/dataset1/'
data_dir = 'LargeSet/'
data_dir = '18mmm/'
dataset_size=len(os.listdir(data_dir))
dataset = psf_dataset(root_dir = data_dir, 
                      size = dataset_size,
                      transform = transforms.Compose([Normalize(), ToTensor()])) #Noise(),


# Check everything works as expected
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ids=np.arange(0, dataset_size)
model = U1(1, 1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

criterion = RMSELoss()
train(model, 
      dataset, 
      optimizer, 
      criterion,
      split = [0.9, 0.1],
      batch_size=256,
      n_epochs = 10000, 
      device=device, 
      
      model_dir = '18mmMod/',
      visdom = True)


 # Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
    
torch.save(model.state_dict(), 'Model.pth')
