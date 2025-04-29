# Small example of network outputs

# Global import
import sys
import os
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from collections import OrderedDict
from dataset import *
import sys
sys.path.insert(0, '../pytorch/')
from train import *
from lr_analyzer import *
from criterion import *
# Local import
sys.path.insert(0, '../src/pytorch/models/')
sys.path.insert(0, '../pytorch/models/')
print(sys.path)
from Unet import UNet
sys.path.insert(0, '../src/pytorch/')
from dataset import *

import os
import time
import utils
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from dataset import psf_dataset, splitDataLoader, ToTensor, Normalize
from utils_visdom import VisdomWebServer
import aotools
from criterion import *

def train(model, dataset, optimizer, criterion, split=[0.9, 0.1], batch_size=32, 
          n_epochs=1, model_dir='./', random_seed=None, visdom=False):
    
    # Create directory if doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Logging
    log_path = os.path.join(model_dir, 'logs.log')
    utils.set_logger(log_path)
    
    # Visdom support
    if visdom:
        vis = VisdomWebServer()
   
    # Dataset
    dataloaders = {}
    dataloaders['train'], dataloaders['val'] = splitDataLoader(dataset, split=split, 
                                                             batch_size=batch_size, random_seed=random_seed)

    # ---
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=140, gamma=0.5)
    #scheduler = CosineWithRestarts(optimizer, T_max=40, eta_min=1e-7, last_epoch=-1)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7, last_epoch=-1)
    
    # Metrics
    metrics_path = os.path.join(model_dir, 'metrics.json')
    
    metrics = {
        'model': model_dir,
        'optimizer': optimizer.__class__.__name__,
        'criterion': criterion.__class__.__name__,
        'scheduler': scheduler.__class__.__name__,
        'dataset_size': int(len(dataset)),
        'train_size': int(split[0]*len(dataset)),
        'test_size': int(split[1]*len(dataset)),
        'n_epoch': n_epochs,
        'batch_size': batch_size,
        'learning_rate': [],
        'train_loss': [],
        'val_loss': [],
        'zernike_train_loss': [],
        'zernike_val_loss': []
    }
    
    # Zernike basis
#     z_basis = torch.as_tensor(aotools.zernikeArray(100+1, 128, norm='rms'), dtype=torch.float32)
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    # Training
    since = time.time()
    dataset_size = {
        'train':int(split[0]*len(dataset)),
        'val':int(split[1]*len(dataset))
    }
    
    
    best_loss = 0.0
    
    for epoch in range(n_epochs):
        
        logging.info('-'*30)
        epoch_time = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            zernike_loss =0.0
            
            for _, sample in enumerate(dataloaders[phase]):
              
                # GPU support 
                inputs = sample['image'].to(device)
                phase_0 = sample['phase'].to(device)

                if (not inputs[0].isnan().any()):

                   # zero the parameter gradients
                   optimizer.zero_grad()
                   #logging.info(' individual loss: %f %f' % (phase_0, phase_estimation))
                
                   # forward: track history if only in train
                   with torch.set_grad_enabled(phase == 'train'):

                       # Network return phase and zernike coeffs
                       phase_estimation = model(inputs)
                       rgg=torch.squeeze(phase_estimation)
                       loss = criterion(torch.squeeze(phase_estimation), phase_0)

                       #logging.info(phase_0)
                       #logging.info(' individual loss: %f' % ( type(rgg)))
                       # backward
                       if phase == 'train':
                            loss.backward()
                            optimizer.step()   
                       
#                        logging.info(' individual loss: %f %f' % (loss.item(), 
#                                                                  .size(0)))
                       if loss.item()!=np.nan:
                             running_loss += 1 * loss.item() * inputs[0].size(0)
             
                logging.info(' where nan: %f %f' % (running_loss, dataset_size[phase]))       
                logging.info('[%i/%i] %s loss: %f' % (epoch+1, n_epochs, phase, running_loss / dataset_size[phase]))
            
                # Update metrics
                metrics[phase+'_loss'].append(running_loss / dataset_size[phase])
                #metrics['zernike_'+phase+'_loss'].append(zernike_loss / dataset_size[phase])
            if phase=='train':
                metrics['learning_rate'].append(get_lr(optimizer))
                
            # Adaptive learning rate
            if phase == 'val':
                scheduler.step()
                # Save weigths
                if epoch == 0 or running_loss < best_loss:
                    best_loss = running_loss
                    model_path = os.path.join(model_dir, 'model.pth')
                    torch.save(model.state_dict(), model_path)
                # Save metrics
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4) 
                # Visdom update 
                if visdom:
                    vis.update(metrics)
                    
        logging.info('[%i/%i] Time: %f s' % (epoch + 1, n_epochs, time.time()-epoch_time))
#         torch.save(model.state_dict(), 'Model.pth')
    time_elapsed = time.time() - since    
    logging.info('[-----] All epochs completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            
        
def get_lr(optimizer):
    for p in optimizer.param_groups:
        lr = p['lr']
    return lr  

data_dir = 'Data1/'
data_dir = 'LargeSet/'
SetSz = len(os.listdir(data_dir))

# data_dir = '/Users/jh2619/Desktop/Machine-learning-for-image-based-wavefront-sensing-master/src/generation/EgData/'
dataset_size = 50000
dataset = psf_dataset(root_dir = data_dir, 
                      size = dataset_size,
                      transform = transforms.Compose([Noise(), Normalize(), ToTensor()]))
model = UNet(2, 1)
criterion = RMSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.006, momentum=0.9)

# Move Network to GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model.cuda()
# Launch training script. The network weights are automatically saved 
# at the end of an epoch (if the test error is reduced). The metrics are also
# saved at the end of each epoch in JSON format. All outputs are also stored in a 
# log file.
#
# - model = network to train
# - dataset = dataset object
# - optimizer = gradient descent optimizer (Adam, SGD, RMSProp)
# - criterion = loss function
# - split[x, 1-x] = Division train/test. 'x' is the proportion of the test set.
# - batch_size = batch size
# - n_epochs = number of epochs
# - model_dir = where to save the results
# - visdom =  enable real time monitoring

train(model, 
      dataset, 
      optimizer, 
      criterion,
      split = [0.73, 1-0.73],
      batch_size = 10000,
      n_epochs = 20000,
      model_dir = 'NewOne10/',
      visdom = False)

 # Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
    
torch.save(model.state_dict(), 'Model.pth')
