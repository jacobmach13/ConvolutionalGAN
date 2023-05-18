# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:46:53 2021

@author: jmach
"""
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.__version__)
print('Using {} device'.format(device))

from convModels_Trainer import convolutional_generator, convolutional_discriminator, trainer

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#==============================================================================================================================
#                       LOAD MNIST TRAINING DATA
#==============================================================================================================================
transform = transforms.Compose([transforms.ToTensor(), transforms.Scale((32,32)), transforms.Normalize(mean=0.5,std=0.5)])
train_data = datasets.MNIST(root="mnist-data", train=True, download=True, transform = transform)

#===============================================================================================================================
#                       TRAIN WITH ADAM
#===============================================================================================================================
# Input parameters
nz_ = 128           # size of random input to the generator model
epochs_ = 100       # number of epochs
capt = 'MNIST'      # for saving figures/models
ktimes = 1          # number of discriminator trainings per generator training

# Wrap training data in an iterable
batchSize = 60      # batch size
train_loader = DataLoader(train_data, batch_size=batchSize, shuffle = True)

# Determine height and width of the images in pixels
sampleImage, _ = next(iter(train_loader))
imgh, imgw = sampleImage.size(2), sampleImage.size(3)
imgChannels = sampleImage.size(1)     # Number of image channels

# Plot sample images
figure = plt.figure(figsize=(16,16))
cols,rows=4,4
for i in range(cols*rows):
    img = sampleImage[i]
    figure.add_subplot(rows,cols,i+1)
    plt.axis('off')
    plt.imshow(transforms.ToPILImage()(img))
plt.savefig('Figures/fig_conv_test0.pdf')

# Create Generator and Discriminator Models
Generator = convolutional_generator(nz_, imgChannels)
Discriminator = convolutional_discriminator(imgChannels)

# Specify loss criterion
loss_criterion = torch.nn.BCELoss() # binary cross entropy loss

# Specify optimization algorithms for the generator and discriminator
GOpt = torch.optim.Adam(Generator.parameters(), lr = 0.0001, betas=(0.9,0.999))
DOpt = torch.optim.Adam(Discriminator.parameters(), lr = 0.0001, betas=(0.9,0.999))

# Initialize and train
Model = trainer(train_loader, batchSize, Generator, Discriminator, GOpt, DOpt, loss_criterion, epochs_, imgw, imgh, nz_, ktimes, capt)
glosses_adam, dlosses_adam = Model.train()

# plot generator and discriminator losses
plt.figure()
plt.plot(glosses_adam, label='MNIST-ADAM: Generator')
plt.plot(dlosses_adam, label='MNIST-ADAM: Discriminator')
plt.xlabel("Epochs")
plt.ylabel("Binary Cross Entropy Loss")
plt.legend()
plt.savefig('Figures/losscurve_conv_' + capt + '.pdf')
