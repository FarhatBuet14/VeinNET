# -*- coding: utf-8 -*-
"""
Created on Wed May 22 23:21:51 2019

@author: User
"""

import numpy as np
import cv2
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout
from keras.layers import Flatten
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



datafile = "./Troughs_Model/Data_for_AccuEdges/data.npz"
weightFile = './Troughs_Model/1/WeightFile_best.hdf5'
prediction_fldr = './Troughs_Model/1/prediction/'


####################### Loading the Data ################################

dataset = np.load(datafile)

X_test = dataset['X_test']
X_test = X_test.reshape((X_test.shape[0], 300, 240, 1))

X_test_info = dataset['X_test_info']

num_pred_value = 6


################## Defining & Loading Model #############################

#(batch, rows, cols, channels)
input_layer = Input(shape=(300,240,1), name='Input_Layer')

#------------------------  Left Conv Layers  -----------------------------

conv_11 = Conv2D(32, kernel_size = (3, 3), strides = 1, 
               activation = 'relu', padding = 'same')(input_layer)
conv_12 = Conv2D(32, kernel_size = (3, 3), strides = 1, 
               activation = 'relu', padding = 'same')(conv_11)
pool_11 = MaxPooling2D(pool_size=(2, 2))(conv_12)



conv_13 = Conv2D(64, kernel_size = (3, 3), strides = 1, 
               activation = 'relu', padding = 'same')(pool_11)
conv_14 = Conv2D(64, kernel_size = (3, 3), strides = 1, 
               activation = 'relu', padding = 'same')(conv_13)
pool_12 = MaxPooling2D(pool_size=(2, 2))(conv_14)



conv_15 = Conv2D(128, kernel_size = (2, 2), strides = 1, 
               activation = 'relu', padding = 'same')(pool_12)
conv_16 = Conv2D(128, kernel_size = (2, 2), strides = 1, 
               activation = 'relu', padding = 'same')(conv_15)
pool_13 = MaxPooling2D(pool_size=(2, 2))(conv_16)
flat_1 = Flatten()(pool_13)

#------------------------ Densed Layer  --------------------


dense_1 = Dense(128, activation='relu')(flat_1)
output_layer = Dense(num_pred_value,
                     name='Output_Layer')(dense_1)


#------------------------  Model Brief  -----------------------------

model = Model(inputs = input_layer, outputs = output_layer)
print(model.summary())


model.load_weights(weightFile)


y_pred = model.predict(X_test)

y_pred = y_pred.reshape((y_pred.shape[0], 3, 2))


for sample in range(0, len(y_pred)):
    
    image_name = str(X_test_info[sample, 0])
    image_name = image_name.replace("b'", '')
    image_name = image_name.replace("'", '')
    
    image_folder_path = str(X_test_info[sample, 1])
    image_folder_path = image_folder_path.replace("b'", '')
    image_folder_path = image_folder_path.replace("'", '')
    
    image = cv2.imread(image_folder_path)
    
    if image is not None:
        print(str(sample) + " - " + prediction_fldr + image_name)
    
    for point in y_pred[sample, :, :]:   
        point = point.astype(int)
        cv2.circle(image, (point[0], 
                   point[1]), 
                   5, (0, 255, 0), -1)
    
    cv2.imwrite(prediction_fldr + image_name, image)















































