#############################  IMPORT LIBRARIES  ############################
import numpy as np
import pandas as pd
from os.path import join
from PIL import Image
import cv2
import re
import imutils

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
                subset=False, transform = None, normalize=True):
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        self.normalize = normalize
        self.cnt_length_thresh = cnt_length_thresh
    
    def get_processed(self, image):
        # pr_img = []
        pr_img = image
        # Find AccumulatedEdged
        # for chan in cv2.split(image):
        #     chan = cv2.medianBlur(chan, 3)
        #     chan = cv2.Canny(chan, 50, 150)
        #     pr_img.append(chan)
        # pr_img = np.array(pr_img, dtype = 'float32')
        # pr_img = (pr_img).reshape((240, 300, 3))
        # pr_img = np.array((image / np.max(image)), dtype = "uint8")
              
        return pr_img
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = str(self.labels.iloc[idx, 0])
        fullname = join(self.root_dir, img_name)
        image = np.array(Image.open(fullname).convert("L"))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = self.get_processed(image)
        image = Image.fromarray(image, "RGB") # Change the np-array to PIL image
        labels = torch.tensor(self.labels.iloc[idx, 1:], 
                            dtype = torch.float32)
        if self.transform:
            image = self.transform(image)
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


if __name__ == "__main__":
    pass
    # main()
