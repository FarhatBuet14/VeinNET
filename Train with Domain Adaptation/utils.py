#############################  IMPORT LIBRARIES  ############################
import numpy as np
import pandas as pd
from os.path import join
from PIL import Image
import cv2
import re
import imutils
import os
import math
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.autograd import Variable

######################### Get the IDs  #########################
################################################################

def get_ID(img_names, dataset = None, id_info_file = None):
    IDs = []
    org = []
    if(dataset == "test_data_2"):
        id_info = pd.read_csv(id_info_file)
        id_col = id_info.iloc[:, 0].values
        name_col = id_info.iloc[:, 7].values
    
    # ---- Find the IDs of the datasets
    for name in img_names:
        if(dataset == "Vera"):
            org.append(0)
            IDs.append(int(name.split("_")[0]))
        elif(dataset == "Bosphorus"):
            org.append(1)
            temp = name.split("_")[0].split()[0]
            IDs.append(np.array(re.findall(r'\d+', temp)).astype(int)[0])
        elif(dataset == "test_data_1"):
            org.append(2)
            temp = name.split("_")[0].split()[0]
            IDs.append(int(temp))
        elif(dataset == "test_data_2"):
            org.append(3)
            if(name in name_col):
                IDs.append(int(id_col[list(name_col).index(name)]))
            else:
                print(name + " is not found in info file..")
                break
        
    return IDs, org

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

def dataset_builder(pathDirData, mode = 'Train', dataset = "Bosphorus"):
    
    if(mode == 'Train'):
        
        if(dataset == "Bosphorus"):
            data = np.load(pathDirData + 'Bos_train_val_data_redefined_80_point_also.npz') 
            # -- Train Data
            X_names = data['names']
            y = data['y']
            ID, org = get_ID(X_names, dataset)
            # -- Val Data
            X_val_names = data['names_val']
            y_val = data['y_val']
            ID_val, org_val = get_ID(X_val_names, dataset)
        
        elif(dataset == "Vera"):
            data = np.load(pathDirData + 'Vera_train_val_data_redefined_80_point_also.npz') 
            # -- Train Data
            X_names = data['names']
            y = data['y']
            ID, org = get_ID(X_names, dataset)
            # -- Val Data
            X_val_names = data['names_val']
            y_val = data['y_val']
            ID_val, org_val = get_ID(X_val_names, dataset)

    elif(mode == 'Test'):
        # ------------------------------------------------------------- Bosphorus
        if(dataset == "Bosphorus"):
            data = np.load(pathDirData + 'Test_Bosphorous.npz') 
            X_names = data['X_test_names'].astype(str)
            y = data['y_test']
            ID, org = get_ID(X_names, dataset)
            print('-' * 100)
            print("Test on Bosphorous Dataset...")
            print('-' * 100)

        # ------------------------------------------------------------- Vera
        if(dataset == "Vera"):
            data = np.load(pathDirData + 'Test_Vera.npz') 
            X_names = data['names'].astype(str)
            y = np.array(data['manual_points']).reshape((-1, 4))
            ID, org = get_ID(X_names, dataset)
            print('-' * 100)
            print("Test on Vera Dataset...")
            print('-' * 100)

        # ------------------------------------------------------------- Test Data - 1
        if(dataset == "test_data_1"):
            data = np.load(pathDirData + 'test_data_1.npz') 
            X_names = data['names'].astype(str)
            y = np.array(data['manual_points']).reshape((-1, 4))
            ID, org = get_ID(X_names, dataset)
            print('-' * 100)
            print("Test on Dataset_1...")
            print('-' * 100)

        # ------------------------------------------------------------- Test Data - 2
        if(dataset == "test_data_2"):
            data = np.load(pathDirData + 'test_data_2.npz') 
            X_names = data['names'].astype(str)
            y = np.array(data['manual_points']).reshape((-1, 4))
            id_info_file = pathDirData + 'Test_data_2_ID_info.csv'
            ID, org = get_ID(X_names, dataset, id_info_file)
            print('-' * 100)
            print("Test on Dataset_2...")
            print('-' * 100)
    
    data = []
    for index in range(0, len(X_names)):
        data.append([X_names[index], ID[index], org[index], y[index, 0], y[index, 1],
                    y[index, 2], y[index, 3]])

    data_df = pd.DataFrame(data, columns=['file_name', 'id', 'origin', 'point_1x', 
                                        'point_1y', 'point_2x', 'point_2y']) 
    

    if(mode == 'Train'):
        data = []
        for index in range(0, len(X_val_names)):
            data.append([X_val_names[index], ID_val[index], org_val[index], y_val[index, 0], 
                            y_val[index, 1], y_val[index, 2], y_val[index, 3]])
        
        val_data = pd.DataFrame(data, columns=['file_name', 'id', 'origin', 'point_1x', 
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

###################### Weights Initialization ####################
##################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        size = m.weight.size()
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    pass
    # main()
