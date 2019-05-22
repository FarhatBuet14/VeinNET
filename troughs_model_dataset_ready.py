# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:02:45 2019

@author: User
"""

import numpy as np


# Function to shuffle two arrays in Unison

def shuffleData(X,y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X_shuffled = X[randomize]
    y_shuffled = y[randomize]
    return X_shuffled,y_shuffled



train_datafile = "./Troughs_Model/train_data/train.npz"
test_datafile = "./Troughs_Model/test_data/test.npz"
datafile = "./Troughs_Model/data.npz"

train_dataset = np.load(train_datafile)
test_dataset = np.load(test_datafile)



train = train_dataset['data']
X_train = train[:, 0:72000]
y_train = train[:, 72000:]



X_test = test_dataset['data']
X_test_info = test_dataset['info']


X_train = X_train/255
X_test = X_test/255


X_train, y_train = shuffleData(X_train,y_train)


random_seed = 0
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.4, random_state=random_seed)



################# Storing the Dataset to an npz file #######################

np.savez(datafile,
         X_train=X_train, X_test=X_test,
         y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test_info = X_test_info)


























