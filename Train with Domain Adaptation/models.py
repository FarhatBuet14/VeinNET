#############################  IMPORT LIBRARIES  ############################
from __future__ import division
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as func
import torch.nn as nn
from torchvision import models
from torch.autograd import Function

import utils
from functions import ReverseLayerF

class resnet50_DANN(torch.nn.Module):
    
    def __init__(self, opt):
        
        super(resnet50_DANN, self).__init__()
        self.opt = opt

        # Feature Extraction Model - ResNET50
        self.feature = models.resnet50(opt.nnClassCount, opt.nnIsTrained)
        modules = list(self.feature.children())[:-1]
        self.feature = torch.nn.Sequential(*modules)

        # ROI Extractiob Model
        self.roi_model = nn.Sequential()
        
        self.roi_model.add_module('c_fc1', nn.Linear(2048, opt.nnClassCount))
        self.roi_model.add_module('c_relu1', nn.ReLU(True))
        
        self.roi_model.apply(utils.weights_init)

        # Domain Classification Model
        
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(2048, 256))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(256))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(256, 1))
        self.domain_classifier.add_module('d_sm', nn.Sigmoid())
        
        self.domain_classifier.apply(utils.weights_init)

        if(opt.gpu):
            self.feature = self.feature.cuda()
            self.roi_model = self.roi_model.cuda()
            self.domain_classifier = self.domain_classifier.cuda()
        
        # Print Trainable and Non-Trainable Parameters
        print('-' * 100)
        print('-' * 100)
        # ------------------------------
        total_params = sum(p.numel() for p in self.feature.parameters())
        print(f'{total_params:,} Feature Model - total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.feature.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} Feature Model - training parameters.')
        # ------------------------------
        total_params = sum(p.numel() for p in self.roi_model.parameters())
        print(f'{total_params:,} ROI Model - total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.roi_model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} ROI Model - training parameters.')
        # ------------------------------
        total_params = sum(p.numel() for p in self.domain_classifier.parameters())
        print(f'{total_params:,} Domain Model - total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.domain_classifier.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} Domain Model - training parameters.')
        

    def forward(self, input, alpha = -1):
        feature = self.feature(input)
        feature = feature.view(-1, 2048)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        roi_output = self.roi_model(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return roi_output, domain_output

if __name__ == "__main__":
    pass
    # main()
