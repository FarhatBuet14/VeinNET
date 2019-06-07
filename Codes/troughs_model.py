# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:24:19 2019

@author: User
"""


##################### Library Imports ################################

import numpy as np

#################### Environment & Variables ############################
from keras import backend as K
K.set_image_data_format('channels_last')

datafile = "./Troughs_Model/model_AccuEdges/7/data_with_aug.npz"

weightFile = './Troughs_Model/model_AccuEdges/7/WeightFile_best.hdf5'

wight_bef = './Troughs_Model/model_AccuEdges/7/best.hdf5'

####################### Loading the Data ################################

dataset = np.load(datafile) 

X = dataset['X']
y = dataset['y']

#X = X / 255



random_seed = 0
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, 
                                                  random_state=random_seed)


X_train = X_train.reshape((X_train.shape[0], 240, 300, 1))
X_val = X_val.reshape((X_val.shape[0], 240, 300, 1))



num_pred_value = y_train.shape[1]

####################### Defining the model (Functional API) ##############################

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras import regularizers 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support



epochs = 1000
batch_size = 8
verbose = 1


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

#model.load_weights(wight_bef)

history = model.fit(X_train, y_train,
                    validation_data = [X_val , y_val],
                    epochs = epochs, verbose = verbose,
                    callbacks= callbacks)

#                    steps_per_epoch = int(training_samples/batch_size),
#                    validation_steps = int(validation_samples/batch_size)
#model.save('Saved_Model.h5')










































