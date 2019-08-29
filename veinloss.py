#############################  IMPORT LIBRARIES  ############################
import numpy as np
import time
import pandas as pd
from os.path import join
from PIL import Image
import cv2
import math
import imgaug.augmenters as iaa
import re
import imutils

import torch
import torch.backends.cudnn as cudnn
from torchsummary import summary
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as func
from torchvision import transforms, datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import Module

import veinloss


######################### Custom Vein Loss #########################
####################################################################

class Vein_loss_class(torch.nn.Module):
    def __init__(self, cropped_fldr, bounding_box_folder, data_folder):
        super(Vein_loss_class, self).__init__()
        self.bounding_box_folder = bounding_box_folder
        self.cropped_fldr = cropped_fldr
        self.data_folder = data_folder
        self.height = 90
        self.width = 70
        self.th = 10
        self.thresh_h = 200
        self.thresh_l = 70

    def get_vein_img(self, save_vein_pic = False,
                    save_bb = True):
        crop = []
        for sample in range(0, self.total_input): 
            top_left = self.output[sample, 0:2]
            top_right = self.output[sample, 2:4]
            
            # Find the angle to rotate the image
            angle  = (180/np.pi) * (np.arctan((top_left[1] - top_right[1])/
                                    (top_left[0] - top_right[0])))
            
            # Rotate the image to cut rectangle from the images
            points_pred = (self.output[sample]).reshape((1, 2, 2))
            points_test = (self.target[sample]).reshape((1, 2, 2))
            image = cv2.imread(self.data_folder + self.img_name[sample])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image.reshape((1, 240, 300))
            image_rotated , keypoints_pred_rotated = iaa.Affine(rotate=-angle)(images=image, 
                                        keypoints=points_pred)
            _ , keypoints_test_rotated = iaa.Affine(rotate=-angle)(images=image, 
                                        keypoints=points_test)
            
            image_rotated = image_rotated.reshape((240, 300))
            keypoints_pred_rotated = keypoints_pred_rotated.reshape((2, 2))
            keypoints_test_rotated = keypoints_test_rotated.reshape((2, 2))
            
            # Rotated Points
            top_left_ = keypoints_pred_rotated[0]    
            top_left_ = tuple(top_left_.reshape(1, -1)[0])
            
            center = np.zeros((2, )).astype(int)
            center[0] = top_left_[0] + int(self.width/2)  - self.th
            center[1] = top_left_[1] + int(self.height/2)
            center = tuple(center.reshape(1, -1)[0])
            
            # Crop the Vein Image
            cropped = cv2.getRectSubPix(image_rotated, (self.width, self.height), 
                                    center)
            crop.append(cropped)
            if(save_vein_pic):
                cv2.imwrite(self.cropped_fldr + self.img_name[sample], cropped)
            
            # Draw Predicted Troughs
            points = keypoints_pred_rotated.reshape((2, 2))    
            for point in points:   
                point = np.array(point).astype(int)
                cv2.circle(image_rotated, (point[0], point[1]), 
                        5, (0, 0, 0), -1)
            
            # Draw Actual Troughs
            points = keypoints_test_rotated.reshape((2, 2))    
            for point in points:   
                point = np.array(point).astype(int)
                cv2.circle(image_rotated, (point[0], point[1]), 
                        5, (255, 0, 0), -1)
            
            # Points for Bounding Boxes
            tl = np.zeros((2, )).astype(int)
            tl[0] = center[0] - int(self.width/2)  
            tl[1] = center[1] - int(self.height/2)
            tl = tuple(tl.reshape(1, -1)[0])
            
            br = np.zeros((2, )).astype(int)
            br[0] = center[0] + int(self.width/2)  
            br[1] = center[1] + int(self.height/2)
            br = tuple(br.reshape(1, -1)[0])
            
            # Draw Bounding Boxes and Save the image
            image_rotated = cv2.rectangle(image_rotated, tl, br , (0,0,0), 2)
            if(save_bb):
                cv2.imwrite(self.bounding_box_folder + self.img_name[sample], 
                            image_rotated)
        crop = np.array(crop)
        return crop
    
    def forward(self,target, output, input, img_name, ids):
        
        self.target = target.cpu().numpy()
        self.output = output.cpu().data.numpy()
        self.input = input.cpu().numpy()
        self.id = ids.cpu().numpy()
        self.id = np.array(self.id, dtype = 'int32')
        self.img_name = img_name
        self.total_input = len(self.id)

        vein_image = self.get_vein_img()
        vein_loss = 0
        # Calculate loss from extracted Vein Image
        for sample in range(0, self.total_input):
            accu = ((vein_image[sample] <= self.thresh_h)  & (vein_image[sample] >= self.thresh_l))
            true = np.count_nonzero(accu)
            false = (accu.shape[0] * accu.shape[1]) - true
            vein_loss += Variable(torch.tensor((false / (false + true))), requires_grad=True)
        vein_loss = vein_loss / self.total_input
        
        return vein_loss * 100
