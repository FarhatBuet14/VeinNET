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
################################################################

def get_ID(img_names, dataset = None):
    IDs = []
    for name in img_names:
        if(dataset == "Vera"):
            IDs.append(int(name.split("_")[0]))
        elif(dataset == "Bosphorus"):
            temp = name.split("_")[0].split()[0]
            IDs.append(np.array(re.findall(r'\d+', temp)).astype(int)[0])
        
    return IDs

##################### Train/Validation Split  ####################
##################################################################

def Train_Validation_Split(X_train_names, y_train, ID, split_factor = float(2/3)):
    X_names = []
    X_val_names = []
    y = []
    y_val = []

    # Splitting IDs for Train-Validation Splitting
    ID_a = np.array(list(dict.fromkeys(ID)))
    randomize = np.arange(len(ID_a))
    np.random.shuffle(randomize)
    ID_a = ID_a[randomize]
    slices = int(len(ID_a) * split_factor)
    ID_train = list(ID_a[:slices])
    ID_val = list(ID_a[slices:])

    for sample in range(0, len(X_train_names)):
        if(ID[sample] in ID_train):
            X_names.append(X_train_names[sample])
            y.append(y_train[sample])
        elif(ID[sample] in ID_val):
            X_val_names.append(X_train_names[sample])
            y_val.append(y_train[sample])
        
    return np.array(X_names), np.array(X_val_names), np.array(y), np.array(y_val)

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
        pr_img = np.array(pr_img)
        pr_img = pr_img.reshape((pr_img.shape[1], pr_img.shape[2], pr_img.shape[0]))

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
        # ----------------------------------------------------------- Bosphorus
        # Import Main Data
        data = np.load(pathDirData + 'Train_data_without_augmentation.npz') 
        X_names = data['X_train_names'].astype(str)
        y = data['y_train']

        # Import Augmented Data
        data = np.load(pathDirData + "Augmented_Train_data.npz") 
        X_train_aug_names = data['X_train_aug_names'].astype(str)
        y_train_aug = data['y_train_aug'].reshape((-1, 4))

        # Concatenate main data and augmented data
        X_names = np.concatenate((X_names, X_train_aug_names), axis = 0)
        y = np.concatenate((y, y_train_aug), axis = 0)

        ID = get_ID(X_names, "Bosphorus")
        split_factor = float(2/3)
        X_names, X_val_names, y, y_val = Train_Validation_Split(X_names, y, ID, split_factor)
        ID = get_ID(X_names, "Bosphorus")
        ID_val = get_ID(X_val_names, "Bosphorus")

        # ----------------------------------------------------------- Vera
        data = np.load(pathDirData + 'Train.npz') 
        X_names_v = data['names'].astype(str)
        y_v = np.array(data['manual_points']).reshape((-1, 4))
        
        ID_v = get_ID(X_names_v, "Vera")
        split_factor_v = float(2/3)
        X_names_v, X_val_names_v, y_v, y_val_v = Train_Validation_Split(X_names_v, y_v, ID_v, split_factor_v)
        ID_v = get_ID(X_names_v, "Bosphorus")
        ID_val_v = get_ID(X_val_names_v, "Bosphorus")

        # Concatenate Bosphorus and Vera Data
        X_names = np.concatenate((X_names, X_names_v), axis = 0)
        y = np.concatenate((y, y_v), axis = 0)
        ID = np.concatenate((ID, ID_v), axis = 0)
        
        X_val_names = np.concatenate((X_val_names, X_val_names_v), axis = 0)
        y_val = np.concatenate((y_val, y_val_v), axis = 0)
        ID_val = np.concatenate((ID_val, ID_val_v), axis = 0)

    if(mode == 'Test'):
        # ------------------------------------------------------------- Bosphorus
        data = np.load(pathDirData + 'Test_data.npz') 
        X_names = data['X_test_names'].astype(str)
        y = data['y_test']
        ID = get_ID(X_names, "Bosphorus")

        # ------------------------------------------------------------- Vera
        data = np.load(pathDirData + 'Test.npz') 
        X_names_v = data['names'].astype(str)
        y_v = np.array(data['manual_points']).reshape((-1, 4))
        ID_v = get_ID(X_names_v, "Vera")

        # Concatenate Bosphorus and Vera Data
        X_names = np.concatenate((X_names, X_names_v), axis = 0)
        y = np.concatenate((y, y_v), axis = 0)
        ID = np.concatenate((ID, ID_v), axis = 0)
    
    data = []
    for index in range(0, len(X_names)):
        data.append([X_names[index], ID[index], y[index, 0], y[index, 1],
                    y[index, 2], y[index, 3]])

    data_df = pd.DataFrame(data, columns=['file_name', 'id', 'point_1x', 
                                        'point_1y', 'point_2x', 'point_2y']) 
    

    if(mode == 'Train'):
        data = []
        for index in range(0, len(X_val_names)):
            data.append([X_val_names[index], ID_val[index], y_val[index, 0], y_val[index, 1],
                            y_val[index, 2], y_val[index, 3]])
        
        val_data = pd.DataFrame(data, columns=['file_name', 'id', 'point_1x', 
                                                'point_1y', 'point_2x', 'point_2y']) 
        return data_df, val_data
    
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
