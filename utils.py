#############################  IMPORT LIBRARIES  ############################
import numpy as np
import pandas as pd
from os.path import join
from PIL import Image
import cv2
import re
import imutils
import os

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.autograd import Variable

######################### Get the IDs  #########################
##################################################################

def get_ID(img_names):
    IDs = []
    for name in img_names:
        temp = name.split("_")[0].split()[0]
        IDs.append(np.array(re.findall(r'\d+', temp)).astype(int)[0])
        
    return IDs

######################### Data Generator #########################
##################################################################

class SeedlingDataset(Dataset):
    def __init__(self, labels, root_dir, cnt_length_thresh = 400, 
                subset=False, trans_pipeline = None, normalize=True):
        self.labels = labels
        self.root_dir = root_dir
        self.trans_pipeline = trans_pipeline
        self.normalize = normalize
        self.cnt_length_thresh = cnt_length_thresh
    
    def get_processed(self, image):
        
        image = np.array(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Find AccumulatedEdged for RGB
        accu = np.zeros(image.shape[:2], dtype="uint8")
        for chan in cv2.split(image):
            chan = cv2.medianBlur(chan, 3)
            chan = cv2.Canny(chan, 50, 150)
            accu = cv2.bitwise_or(accu, chan)
        accu = np.array(accu, dtype = 'float32')

        # Find AccumulatedEdged for HSV
        accu2 = np.zeros(hsv.shape[:2], dtype="uint8")
        for chan in cv2.split(hsv):
            chan = cv2.medianBlur(chan, 3)
            chan = cv2.Canny(chan, 50, 150)
            accu2 = cv2.bitwise_or(accu2, chan)
        accu2 = np.array(accu2, dtype = 'float32')
        
        # Cascade all channels
        pr_img = []
        pr_img.append(gray)
        pr_img.append(accu)
        pr_img.append(accu2)
        pr_img = np.array(pr_img).reshape((240, 300, len(pr_img)))

        return pr_img
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = str(self.labels.iloc[idx, 0])
        fullname = join(self.root_dir, img_name)
        image = Image.open(fullname).convert("RGB")
        image = self.get_processed(image)
        labels = torch.tensor(self.labels.iloc[idx, 1:], 
                            dtype = torch.float32)
        if self.trans_pipeline:
            image = self.trans_pipeline(image)
        return image, labels, img_name

######################### Dataset Builder #########################
##################################################################

def dataset_builder(pathDirData, mode = 'Train'):
    
    if(mode == 'Train'):
        # Import Main Data
        data = np.load(pathDirData + 'train_test_data_without_augmetation.npz') 
        X_names = data['X_train_names'].astype(str)
        y = data['y_train']

        # Import Augmented Data
        data = np.load(pathDirData + "Augmented_Train_data.npz") 
        X_train_aug_names = data['X_train_aug_names'].astype(str)
        y_train_aug = data['y_train_aug'].reshape((-1, 4))

        # Concatenate main data and augmented data
        X_names = np.concatenate((X_names, X_train_aug_names), axis = 0)
        y = np.concatenate((y, y_train_aug), axis = 0)
    
    if(mode == 'Test'):
        # Import Main Data
        data = np.load(pathDirData + 'train_test_data_without_augmetation.npz') 
        X_names = data['X_test_names'].astype(str)
        y = data['y_test']

    ID = get_ID(X_names)
    
    data = []
    for index in range(0, len(X_names)):
        data.append([X_names[index], ID[index], y[index, 0], y[index, 1],
                    y[index, 2], y[index, 3]])

    data_df = pd.DataFrame(data, columns=['file_name', 'id', 'point_1x', 
                                        'point_1y', 'point_2x', 'point_2y']) 

    if(mode == 'Train'):
        train_data = data_df.sample(frac=0.7)
        valid_data = data_df[~data_df['file_name'].isin(train_data['file_name'])]
        return train_data, valid_data
    else:
        return data_df

######################### Load Checkpoint ########################
##################################################################

def load_checkpoint(model, optimizer, losslogger, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print('-' * 100)
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['best_loss']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
        print('-' * 100)
    else:
        print('-' * 100)
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger

if __name__ == "__main__":
    pass
    # main()
