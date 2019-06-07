# -*- coding: utf-8 -*-
"""
Created on Wed May 22 23:21:51 2019

@author: User
"""

import numpy as np
import cv2
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout
from keras.layers import Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras import regularizers 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support



################# Environment and Variables #############################

from keras import backend as K
K.set_image_data_format('channels_last')

seed = 7
np.random.seed(seed)



datafile = "./Troughs_Model/model_AccuEdges/7/data_with_aug.npz"
#gray_datafile = "./Troughs_Model/model_AccuEdges/2/test_data/test_gray.npz"
weightFile = './Troughs_Model/model_AccuEdges/7/WeightFile_best.hdf5'
prediction_fldr = './Troughs_Model/model_AccuEdges/7/prediction/'


####################### Loading the Data ################################

dataset = np.load(datafile)
#gray_dataset = np.load(gray_datafile)

#X_test = test_dataset['X']
#X_test = X_test.reshape((X_test.shape[0], 240, 300, 1))
#test_names = test_dataset['test_names'].astype(str)
#test_paths = test_dataset['test_paths'].astype(str)
#
#X_test_gray = gray_dataset['X']
#X_test_gray = X_test_gray.reshape((X_test_gray.shape[0], 240, 300, 1))


X_test = dataset['X_test']
X_test_gray = dataset['X_test_gray']
y_test = dataset['y_test']
test_names = dataset['test_names']

X_test = X_test.reshape((X_test.shape[0], 240, 300, 1))
X_test_gray = X_test_gray.reshape((X_test.shape[0], 240, 300, 1))

#X_test = X_test / 255

num_pred_value = 6

################## Defining & Loading Model #############################

#(batch, rows, cols, channels)
input_layer = Input(shape=(240,300, 1), name='Input_Layer')

#------------------------  Left Conv Layers  -----------------------------

conv_11 = Conv2D(32, kernel_size = (3, 3))(input_layer)
b_n_11 = BatchNormalization()(conv_11)
a_11 = Activation("relu")(b_n_11)

conv_12 = Conv2D(32, kernel_size = (3, 3))(a_11)
b_n_12 = BatchNormalization()(conv_12)
a_12 = Activation("relu")(b_n_12)

pool_11 = MaxPooling2D(pool_size=(3, 3))(a_11)
dp_11 = Dropout(0.01)(pool_11)



conv_13 = Conv2D(64, kernel_size = (3, 3))(dp_11)
b_n_13 = BatchNormalization()(conv_13)
a_13 = Activation("relu")(b_n_13)

conv_14 = Conv2D(64, kernel_size = (3, 3))(a_13)
b_n_14 = BatchNormalization()(conv_14)
a_14 = Activation("relu")(b_n_14)

pool_12 = MaxPooling2D(pool_size=(3, 3))(a_13)
dp_12 = Dropout(0.01)(pool_12)



conv_15 = Conv2D(128, kernel_size = (3, 3))(dp_12)
b_n_15 = BatchNormalization()(conv_15)
a_15 = Activation("relu")(b_n_15)

conv_16 = Conv2D(128, kernel_size = (3, 3))(a_15)
b_n_16 = BatchNormalization()(conv_16)
a_16 = Activation("relu")(b_n_16)

pool_13 = MaxPooling2D(pool_size=(3, 3))(a_16)
dp_13 = Dropout(0.01)(pool_13)

flat_1 = Flatten()(dp_13)

#------------------------ Densed Layer  --------------------


dense_1 = Dense(128, kernel_regularizer=regularizers.l2(0.001),
                activation='relu')(flat_1)
dp_14 = Dropout(0.01)(dense_1)
output_layer = Dense(num_pred_value,
                     name='Output_Layer')(dp_14)


#------------------------  Model Brief  -----------------------------

model = Model(inputs = input_layer, outputs = output_layer)
print(model.summary())


model.load_weights(weightFile)


y_pred = model.predict(X_test)


y_pred = y_pred.reshape((y_pred.shape[0], 3, 2))

#X_test_gray = X_test_gray * 255


count = 0
for sample in range(0, len(y_pred)):
    
    image = X_test_gray[sample, :, :, 0].reshape(
            (X_test_gray.shape[1], X_test_gray.shape[2])).astype(np.uint8)
    
    
    if image is not None:
        points = y_pred[sample, :, :]   
        points = points.astype(int)
        dists = np.zeros((3, 1))
        dists[0] = math.hypot(points[0, 0] - points[1, 0],        ## (0-1)
                          points[0, 1] - points[1, 1])   
        dists[1] = math.hypot(points[0, 0] - points[2, 0],        ## (0-2) 
                          points[0, 1] - points[2, 1]) 
        dists[2] = math.hypot(points[2, 0] - points[1, 0],        ## (1-2) 
                          points[2, 1] - points[1, 1]) 
        
        max_dist_arg = np.argmax(dists)
        width = int(dists[max_dist_arg])
        height = int(1.5 * width)
        
        points = points.tolist()
        if(max_dist_arg == 0): del points[2]
        if(max_dist_arg == 1): del points[1]
        if(max_dist_arg == 2): del points[0]
        points = np.array(points)
        
        if(points[0, 0] > points[1, 0]):
            temp = []
            temp.append(points[1])
            temp.append(points[0])
            points = np.array(temp)
        
        top_left = points[0]
        top_right = points[1]
        
        m  = (top_left[1] - top_right[1])/(top_left[0] - top_right[0])
        
        bottom_right = np.zeros((2, 1))
        bottom_right[0] = top_right[0] - np.sqrt( ((m**2) * (height**2)) / (1 + m**2) )
        
        bottom_right[1] = top_right[1] + np.sqrt( (height**2) / (1 + m**2) )
        
        bottom_right = bottom_right.astype(int)
        
        top_left = tuple(top_left.reshape(1, -1)[0])
        bottom_right = tuple(bottom_right.reshape(1, -1)[0])
        top_right = tuple(top_right.reshape(1, -1)[0])
        
#        cv2.rectangle(image, top_left, bottom_right ,(0,0,255),2)
        cv2.line(image,top_left,top_right,(255,0, 0),2)
        cv2.line(image,top_right,bottom_right,(0,0,255),2)

        for point in points:
            point = point.astype(int)
            cv2.circle(image, (point[0], 
                       point[1]), 
                       5, (0, 255, 0), -1)
        filenames = os.listdir(prediction_fldr)
        
        if(test_names[sample] in filenames): print(test_names[sample] + " is overwritten")
        
        cv2.imwrite(prediction_fldr + test_names[sample], image)
        count += 1
        
    else:
        print(test_names[sample] + " is found None")



#points = y_pred[32, :, :]   
#points = points.astype(int)
#dists = np.zeros((3, 1))
#dists[0] = math.hypot(points[0, 0] - points[1, 0],        ## (0-1)
#                  points[0, 1] - points[1, 1])   
#dists[1] = math.hypot(points[0, 0] - points[2, 0],        ## (0-2) 
#                  points[0, 1] - points[2, 1]) 
#dists[2] = math.hypot(points[2, 0] - points[1, 0],        ## (1-2) 
#                  points[2, 1] - points[1, 1]) 
#
#max_dist_arg = np.argmax(dists)
#width = int(dists[max_dist_arg])
#height = int(1.5 * width)
#
#points = points.tolist()
#if(max_dist_arg == 0): del points[2]
#if(max_dist_arg == 1): del points[1]
#if(max_dist_arg == 2): del points[0]
#points = np.array(points)
#
#if(points[0, 0] > points[1, 0]):
#    temp = []
#    temp.append(points[1])
#    temp.append(points[0])
#    points = np.array(temp)
#
#top_left = points[0]
#top_right = points[1]
#m  = (top_left[1] - top_right[1])/(top_left[0] - top_right[0])
#bottom_right = np.zeros((2, 2))
#bottom_right[0, 0] = top_right[0] + np.sqrt(( ((m**2) * (height**2)) / (1 + m**2) ))
#
#bottom_right[0, 1] = top_right[1] + np.sqrt(( (height**2) / (1 + m**2) ))
#
#bottom_right[1, 0] = top_right[0] - np.sqrt(( ((m**2) * (height**2)) / (1 + m**2) ))
#
#bottom_right[1, 1] = top_right[1] - np.sqrt(( (height**2) / (1 + m**2) ))
#
#bottom_right = bottom_right[0]
#
#cv2.rectangle(img, top_left, bottom_right ,(0,0,255),15)
#

























#points = y_pred[32, :, :].astype(int)  
#dists = np.zeros((3, 1))
#dists[0] = math.hypot(points[0, 0] - points[1, 0],        ## (0-1)
#                  points[0, 1] - points[1, 1])   
#dists[1] = math.hypot(points[0, 0] - points[2, 0],        ## (0-2) 
#                  points[0, 1] - points[2, 1]) 
#dists[2] = math.hypot(points[2, 0] - points[1, 0],        ## (1-2) 
#                  points[2, 1] - points[1, 1]) 
#
#max_dist = np.argmax(dists)
#points = points.tolist()
#if(max_dist == 0): del points[2]
#if(max_dist == 1): del points[1]
#if(max_dist == 2): del points[0]
#points = np.array(points)








#y_pred = y_pred.reshape((y_pred.shape[0], 6))
#
#err = abs(y_test - y_pred)
#
#err = np.sum(err, axis = 1)/6
#
#err_avg = sum(err)/len(err)
#
#
#plt.figure(figsize = (20, 10))
#plt.plot(err)
#plt.show()
#
#
#
#pred_datafile = "./Troughs_Model/model_AccuEdges/7/pred_data_7.npz"
#np.savez(pred_datafile,
#         y_pred = y_pred,
#         err = err,
#         err_avg = err_avg,
#         test_names = test_names)
#
#
#






