# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 10:57:32 2021

@author: jmach
"""
import torch
import torchvision.transforms as transforms
from torch import nn

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
is_cuda = torch.cuda.is_available()

#====================================================================================================================
#                           BUILD GENERATOR NETWORK
#====================================================================================================================
class convolutional_generator(nn.Module):
    def __init__(self, nz = 128, img_channels=3):
        super(convolutional_generator, self).__init__()
        
        self.transpose1 = nn.ConvTranspose2d(in_channels = nz, out_channels=256, kernel_size=4, stride=2, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        # Output1 = batchSize x 256 x 4 x 4
        
        self.transpose2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        # Output2 = batchSize x 128 x 8 x 8
        
        self.transpose3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        # Output3 = batchSize x 64 x 16 x 16
        
        self.transpose4 = nn.ConvTranspose2d(in_channels=64, out_channels=img_channels, kernel_size=4, stride=2, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(img_channels)
        self.tanh = nn.Tanh()
        #Output = batchSize x 3 x 32 x 32
                
    def forward(self,x):
        """
        Performs forward propagation
        """
        out = self.transpose1(x)
        out = self.batchnorm1(out)
        out = self.relu1(out)
        
        out = self.transpose2(out)
        out = self.batchnorm2(out)
        out = self.relu2(out)
        
        out = self.transpose3(out)
        out = self.batchnorm3(out)
        out = self.relu3(out)
        
        out = self.transpose4(out)
        out = self.batchnorm4(out)
        out = self.tanh(out)
        
        return out

#==========================================================================================================================
#                       BUILD DISCRIMINATOR NETWORK
#==========================================================================================================================
class convolutional_discriminator(nn.Module):
    def __init__(self,img_channels=3):
        super(convolutional_discriminator, self).__init__()
        
        # convolutional layer 1
        # Input = batchSize x 3 x 32 x 32
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.leaky1 = nn.LeakyReLU(0.2)
        # Output1 = batchSize x 64 x 16 x 16
        
        # convolutional layer 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.leaky2 = nn.LeakyReLU(0.2)
        # Output1 = batchSize x 128 x 8 x 8
        
        # convolutional layer 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.leaky3 = nn.LeakyReLU(0.2)
        # Output1 = batchSize x 256 x 4 x 4
        
        # Fully-connected layer
        self.linear = nn.Linear(256*4*4, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        """
        Performs forward propagation
        """
        # output of first convolutional layer
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.leaky1(out)
        
        # output of second convolutional layer
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.leaky2(out)
        
        # output of third convolutional layer
        out = self.conv3(out)
        out = self.batchnorm3(out)
        out = self.leaky3(out)
        
        # pass through fully connected layer
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        out = self.sigmoid(out)
        
        return out

#==========================================================================================================================
#                       TRAINER CLASS
#==========================================================================================================================
class trainer():
    def __init__(self, data, bsize, gen, disc, gopt, dopt, criterion, epochs, imgx, imgy, nz, k, caption):
        """
        Initializes the trainer for Generative Adversarial Network
        
        Inputs:
        =======
        data:           training data
        bsize:          batch size
        gen:            generator network
        disc:           discriminator network
        gopt:           generator optimizer
        dopt:           discriminator optimizer
        criterion:      loss criterion
        epochs:         number of training epochs
        imgx:           Number of pixels along the width of an image
        imgy:           Number of pixels along the height of an image
        nz:             size of noise input to the generator
        k:              number of discriminator trainings per single generator training
        caption:        string variable for saving plots/models
        """
        self.data = data                    
        self.bsize = bsize  
        self.generator = gen
        self.discriminator = disc                          
        self.g_optimizer = gopt            
        self.d_optimizer = dopt             
        self.criterion = criterion          
        self.epochs = epochs   
        self.imgx = imgx
        self.imgy = imgy            
        self.nz = nz                        
        self.k = k                         
        self.caption = caption     

    def create_noise(self, batch_size, nz):
        """
        Generates random input to the generator model
        """
        return torch.randn(batch_size,nz,1,1).float().to(device)

    def create_real_labels(self):
        """
        Creates real labels
        """
        if is_cuda:
            return torch.ones(self.bsize,1).cuda()-0.01
        return torch.ones(self.bsize,1)-0.01

    def create_fake_labels(self):
        """
        Creates fake labels
        """
        if is_cuda:
            return torch.zeros(self.bsize,1).cuda()+0.01
        return torch.zeros(self.bsize,1)+0.01

    def train(self):
        """
        Trains Generative Adversarial Network
        
        Outputs:
        ========
        g_losses:       generator losses
        d_losses:       discriminator losses
        """
        # Parallelized Generator and Discriminator networks
        generator = nn.DataParallel(self.generator).to(device)
        discriminator = nn.DataParallel(self.discriminator).to(device)
    
        # For storing generator and discriminator losses
        g_losses = []
        d_losses = []
    
        # Frequency of plotting generated images
        freq = int(self.epochs*0.1)
    
        for epoch in range(self.epochs):
            g_loss = 0.0
            d_loss = 0.0
            for count, (feature, label) in enumerate(self.data):
                #=============================================
                #           TRAIN DISCRIMINATOR
                #=============================================
                
                # Obtain real input to the discriminator model
                real_input = feature.float().to(device)
                
                # Obtain fake input to the discriminator model
                noise = self.create_noise(self.bsize,self.nz)
                fake_input = generator(noise)
                
                
                for i in range(self.k):
                    # Zero discriminator gradients
                    self.d_optimizer.zero_grad() 
                    
                    # Pass real data through the discriminator network and compute loss
                    real_output = discriminator(real_input)
                    real_labels = self.create_real_labels()
                    real_loss = self.criterion(real_output, real_labels)
            
                    # Pass fake data through the discriminator network and compute loss
                    fake_output = discriminator(fake_input)
                    fake_labels = self.create_fake_labels()
                    fake_loss = self.criterion(fake_output, fake_labels)
        
                    # Compute discriminator's effective loss due to real and fake inputs
                    discriminator_loss = fake_loss + real_loss 
            
                    # compute gradients and then update discriminator weights only
                    discriminator_loss.backward()       # calculates gradients
                    self.d_optimizer.step()             # updates weights

                #=============================================
                #           TRAIN GENERATOR
                #=============================================
                
                # Zero generator gradients
                self.g_optimizer.zero_grad() 
                
                # Pass random noise through the generator network
                noise = self.create_noise(self.bsize, self.nz)
                generator_output = generator(noise)
            
                # Pass generator output through the discriminator network and compute loss
                discriminator_output = discriminator(generator_output)
                real_labels = self.create_real_labels()
                generator_loss = self.criterion(discriminator_output, real_labels)
            
                # Compute gradients and then update generator weights only
                generator_loss.backward()     
                self.g_optimizer.step()
        
                # Accumulate losses
                d_loss += discriminator_loss.item()
                g_loss += generator_loss.item()
        
            # Save losses for plotting
            d_losses.append(d_loss/count)
            g_losses.append(g_loss/count)
        
            # Sample and show generated images
            if(epoch%freq == 0):
                generated_images = generator(self.create_noise(16,self.nz))
                figure = plt.figure(figsize=(16,16))
                cols,rows=4,4
                for i in range(cols*rows):
                    img = generated_images[i]
                    figure.add_subplot(rows,cols,i+1)
                    plt.axis('off')
                    plt.imshow(transforms.ToPILImage()(img))
                plt.savefig('Figures/'+'fig_conv_' + self.caption + '_' + str(epoch) + '.pdf')
                print(f"Sample generated images for epoch = {epoch} have been saved to 'Figures' subfolder!")
        # Save trained models
        torch.save(generator, 'Models/convGenerator_' + self.caption + '.pt')  
        torch.save(discriminator, 'Models/convDiscriminator_'+ self.caption + '.pt')   
        print(f"Training complete for {self.caption}!")

        # return generator and discriminator losses
        return g_losses, d_losses
