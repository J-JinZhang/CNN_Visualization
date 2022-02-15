"""
Author: Dr. Jin Zhang 
E-mail: j.zhang.vision@gmail.com
Created on 2022.02.10
"""

import os
import numpy as np
from PIL import Image
import pandas as pd

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import save_image

from resnet import resnet34
from inception_net import Froth_Inception
import matplotlib.pyplot as plt


class XRF_ResNet(nn.Module):
    def __init__(self, num_classes=1, aux_logits=False, transform_input=False):
        super(XRF_ResNet, self).__init__()
        self.features = resnet34()
        self.regressor =  nn.Sequential(
            nn.Linear(515, 1024), 
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, 3))

        for m in self.regressor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def forward(self, x0, x1):
        x1 = self.features(x1)
        #print(x1.shape)
        x1 = F.avg_pool2d(x1, kernel_size=10)  #
        plt.bar(range(x1.size(1)), x1.view(-1).data.numpy())
        plt.show()
        x = torch.cat((x0.view(x0.size(0), -1), x1.view(x1.size(0), -1)), dim=1)
        x = self.regressor(x)
        return x
    
    
class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        mean = [0.5561, 0.5706, 0.5491]
        std = [0.1833, 0.1916, 0.2061]
        self.mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        #print(model)
        
        
    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            self.conv_output = grad_out[0, self.selected_filter]  # Gets the conv output of the selected filter
        self.model[self.selected_layer].register_forward_hook(hook_function)  # Hook the selected layer

    def visualise_layer_with_hooks(self, input):
        self.hook_layer()  # Hook the selected layer
        input.requires_grad = True
        # Define optimizer for the image
        optimizer = Adam([input], lr=1e-1, weight_decay=1e-6)
        for i in range(1, 401):   # 重复执行，结果图像逐步增强
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = input.unsqueeze(0)
            #print(f"shape of x: {x.size()}")
            #im_show = x.data * self.std + self.mean
            #plt.imshow(im_show.squeeze(0).permute(2,1,0))
            #
            #plt.show()
            for index, layer in enumerate(self.model):
                x = layer(x)
                #print(f"shape of x: {x.size()}")
                if index == self.selected_layer:
                    break  # (forward hook function triggered)
            im_show = self.conv_output.data
            print(f"size of im_show: {im_show.size()}")
            plt.imshow(im_show.squeeze(0))
            plt.show()
            loss = -torch.mean(self.conv_output)  # We try to minimize the mean of the output of that specific filter
            loss.backward()
            optimizer.step()
            
            if i % 5 == 0:
                im_save = input.data * self.std + self.mean
                save_image(im_save, 'generated/layer_vis_l' + str(self.selected_layer) +
                            '_f' + str(self.selected_filter) + '_iter'+str(i)+'.jpg')
                
                
class TransferIm2TargetIndex():
    def __init__(self, model):
        self.model = model
        self.model.eval()
        mean = [0.5561, 0.5706, 0.5491]
        std = [0.1833, 0.1916, 0.2061]
        self.mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        self.mean_feed = torch.FloatTensor([8.411431353067606, 0.5473817630918659, 23.97601543147942])
        self.std_feed = torch.FloatTensor([0.7960769134313461, 0.05987490652171015, 0.7161782274613697])
        
    def understand_feature_patterns(self, tailing, image, name):
        image.requires_grad = True
        target = torch.FloatTensor([6.0, 0.5, 24.7])
        target = (target - self.mean_feed) / self.std_feed
        optimizer = Adam([image], lr=5e-2, weight_decay=1e-6)
        for i in range(1, 4001):   # 重复执行，结果图像逐步增强
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            out = self.model(tailing.unsqueeze(0), image.unsqueeze(0), name)
            print(f"Out: {out.data * self.std_feed + self.mean_feed}")
            loss = torch.sum((out - target)**2)
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                im_save = image.data * self.std + self.mean
                save_image(im_save, 'generated/regress_vis' + '_iter'+str(i)+'.jpg', normalize=True)

if __name__ == '__main__':
    if not os.path.exists('generated'):
        os.makedirs('generated')
           
    mean_feed = torch.FloatTensor([8.411431353067606, 0.5473817630918659, 23.97601543147942])
    std_feed = torch.FloatTensor([0.7960769134313461, 0.05987490652171015, 0.7161782274613697])
    mean_tailing = torch.FloatTensor([1.3901876578758057, 0.48554048370970193, 25.40719649345569])
    std_tailing = torch.FloatTensor([0.2688268864000507, 0.03469305624144162, 1.0110712690887271])
    """
    1.358	0.505	26.042	20170831002923	8.558	0.542	24.793 (1-20170831002819)
    1.218	0.478	26.029	20170831180439	8.895	0.534	23.778 (2-20170831180139)
    1.206	0.471	26.523	20170830095228	8.675	0.530	24.648 (3-20170830095333)
    1.058	0.489	26.139	20170831153045	8.948	0.566	24.010 (4-20170831153605)
    1.228	0.475	26.587	20170830125447	8.561	0.573	24.917 (5-20170830125655)

    """
            
    transforms = torchvision.transforms.Compose([
                    torchvision.transforms.CenterCrop(300),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.5561, 0.5706, 0.5491],
                                                      std=[0.1833, 0.1916, 0.2061])])
                
    cnn_layer = 11
    filter_pos = 1
    name = '3-20170830095333'
    
    img_pil = Image.open(f'Images/{name}.jpg').convert("RGB")
    img_tensor = transforms(img_pil)
    tailing = (torch.FloatTensor([1.358, 0.505, 26.042]) - mean_tailing) / std_tailing
    
    model = Froth_Inception()
    save_file = './saved_models/XRF_InceptionNet_epoch_300.pth'
    #model = XRF_ResNet()
    #save_file = "./saved_models/XRF_ResNet_epoch_300.pth"
    model.load_state_dict(torch.load(save_file))
    
    layer_vis = CNNLayerVisualization(model.features, cnn_layer, filter_pos)  #Froth_Inception包含两部分输入，这里仅采用其features方法
    layer_vis.visualise_layer_with_hooks(img_tensor)

    #regress_vis = TransferIm2TargetIndex(model)
    #regress_vis.understand_feature_patterns(tailing, img_tensor, name)
    
    