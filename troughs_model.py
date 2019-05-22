# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:24:19 2019

@author: User
"""


##################### Library Imports ################################

import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler


#################### Environment & Variables ############################
from keras import backend as K
K.set_image_data_format('channels_last')

datafile = "./Troughs_Model/data.npz"

weightFile = './WeightFile_best.hdf5'


####################### Loading the Data ################################

dataset = np.load(datafile)

X_train = dataset['X_train']
y_train = dataset['y_train']

X_val = dataset['X_val']
y_val = dataset['y_val']


X_train = X_train.reshape((X_train.shape[0], 300, 240, 1))
X_val = X_val.reshape((X_val.shape[0], 300, 240, 1))


#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#y_train_main = y_train
#y_train = sc_X.fit_transform(y_train)


num_pred_value = y_train.shape[1]

####################### Defining the model (Functional API) ##############################

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras import regularizers 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support



epochs = 100
batch_size = 8
verbose = 1


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


#------------------------  Right Conv Layer  -----------------------------

#conv_21 = Conv2D(64, kernel_size = (3, 3), strides = 1, 
#               activation = 'relu', padding = 'same')(input_layer)
#conv_22 = Conv2D(64, kernel_size = (3, 3), strides = 1, 
#               activation = 'relu', padding = 'same')(conv_21)
#pool_21 = MaxPooling2D(pool_size=(2, 2))(conv_22)
#
#
#
#conv_23 = Conv2D(128, kernel_size = (3, 3), strides = 1, 
#               activation = 'relu', padding = 'same')(pool_21)
#conv_24 = Conv2D(128, kernel_size = (3, 3), strides = 1, 
#               activation = 'relu', padding = 'same')(conv_23)
#pool_22 = MaxPooling2D(pool_size=(2, 2))(conv_24)
#
#
#
#conv_25 = Conv2D(128, kernel_size = (3, 3), strides = 1, 
#               activation = 'relu', padding = 'same')(pool_22)
#conv_26 = Conv2D(128, kernel_size = (3, 3), strides = 1, 
#               activation = 'relu', padding = 'same')(conv_25)
#pool_23 = MaxPooling2D(pool_size=(2, 2))(conv_26)
#flat_2 = Flatten()(pool_23)

#------------------------  Cascaded & Densed Layer  --------------------

#merge = concatenate([flat_1, flat_2])

dense_1 = Dense(128, activation='relu')(flat_1)
output_layer = Dense(num_pred_value,
                     name='Output_Layer')(dense_1)


#------------------------  Model Brief  -----------------------------

model = Model(inputs = input_layer, outputs = output_layer)
print(model.summary())
plot_model(model, to_file='shared_input_layer.png')


#################### Compiling the Model ################################
optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='mae', optimizer=optimizer, 
              metrics=['mse'])

#################### Defining the Checkpoints ###########################

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.25, 
                                            min_lr=0.00001)

wigth  = ModelCheckpoint(weightFile, monitor = 'val_loss' )
callbacks = [wigth, learning_rate_reduction]


############################ Data Augmentation ############################

training_samples = X_train.shape[0]
validation_samples = X_val.shape[0]


history = model.fit(X_train, y_train,
                    validation_data = [X_val , y_val],
                    epochs = epochs, verbose = verbose,
                    callbacks= callbacks)

#                    steps_per_epoch = int(training_samples/batch_size),
#                    validation_steps = int(validation_samples/batch_size)
model.save('Saved_Model.h5')












































