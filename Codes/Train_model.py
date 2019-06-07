# -*- coding: utf-8 -*-
"""
Created on Sun May  5 02:12:14 2019

@author: User
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv("dataset.csv")
data = np.array(data)
data = data[:, 1:]

X = data[:216000, :]
Y = data[216000:216004, :]

Total_Samples = X.shape[1]
n_features_model = X.shape[0]
train_samples = int(Total_Samples * 0.9)
n_outputs = Y.shape[0];


X_train = X[:, :train_samples]
X_test = X[:, train_samples:]
y_train = Y[:, :train_samples]
y_test = Y[:, train_samples:]


#####################  Feature Scaling  ############################

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

X_train = sc.fit_transform(X_train)
y_train = sc.fit_transform(y_train)


########################  Reshaping  #####################

X_1 = np.zeros((Total_Samples, Window_Size, n_features_model_1))
X_1[:, :, 0] = ACCx_scaled
X_1[:, :, 1] = ACCy_scaled
X_1[:, :, 2] = ACCz_scaled

X_2 = np.zeros((Total_Samples, Window_Size, n_features_model_2))
X_2[:, :, 0] = Noise_PPG_scaled

y = np.zeros((Total_Samples, 1))
y[:, 0:1] = BPMs_scaled

y_2 = np.zeros((Total_Samples, Window_Size, 1))
y_2[:, :, 0] = Clean_PPG_scaled




training_samples = 1766
validation_samples = 436


X_val_1 = X_1[0:validation_samples, :]
X_train_1 = X_1[validation_samples:training_samples, :, :]       ## ACC_Sig
X_test_1 = X_1[training_samples:Total_Samples, :, :]


X_val_2 = X_2[0:validation_samples, :]
X_train_2 = X_2[validation_samples:training_samples, :, :]       ##PPG_Sig
X_test_2 = X_2[training_samples:Total_Samples, :, :]


y_val = y[0:validation_samples, :]
y_train = y[validation_samples:training_samples, :]             ##BPm
y_test = y[training_samples:Total_Samples, :]



y_val_2 = y_2[0:validation_samples, :]
y_train_2 = y_2[validation_samples:training_samples, :]         ##Noise
y_test_2 = y_2[training_samples:Total_Samples, :]



###########################  Build the model  #################################

import keras
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten
from keras.models import Model
from keras.layers import Dropout, Dense
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras import regularizers 
from keras.layers import Reshape


seed = 7
np.random.seed(seed)


####################################################################
#########################   Model-1   ##############################

#----------- ACC-x denoising -------------

ACCx_sig = Input(shape=(1000, 1, ),
                   name='ACCx_input')

#----------- 1st Conv Layer -------------
conv_1 = Conv1D(filters=8, kernel_size=3, activation='relu')(ACCx_sig)
conv_1 = Conv1D(filters=8, kernel_size=3, activation='relu')(conv_1)
out_1 = Conv1D(filters=8, kernel_size=3, activation='relu')(conv_1)


#----------- 2nd Conv Layer -------------
conv_2 = Conv1D(filters=32, kernel_size=3, activation='relu')(out_1)
conv_2 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv_2)
out_2 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv_2)


last_1 = Flatten()(out_2)
Noise_output_1 = Dense(n_outputs_1, 
                        name='Noise_X_output')(last_1)
Noise_output_Reshaped_1 = Reshape((1000,1))(Noise_output_1)
PPg_sig = Input(shape=(1000, 1, ),
                   name='PPG_input')
PPg_sig_2 = keras.layers.Subtract()([PPg_sig, Noise_output_Reshaped_1])



#----------- ACC-y denoising -------------

ACCy_sig = Input(shape=(1000, 1, ),
                   name='ACCy_input')

#----------- 1st Conv Layer -------------
conv_3 = Conv1D(filters=8, kernel_size=3, activation='relu')(ACCy_sig)
conv_3 = Conv1D(filters=8, kernel_size=3, activation='relu')(conv_3)
out_3 = Conv1D(filters=8, kernel_size=3, activation='relu')(conv_3)


#----------- 2nd Conv Layer -------------
conv_4 = Conv1D(filters=32, kernel_size=3, activation='relu')(out_3)
conv_4 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv_4)
out_4 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv_4)


last_2 = Flatten()(out_4)
Noise_output_2 = Dense(n_outputs_1, 
                        name='Noise_Y_output')(last_2)
Noise_output_Reshaped_2 = Reshape((1000,1))(Noise_output_2)

PPg_sig_3 = keras.layers.Subtract()([PPg_sig_2, Noise_output_Reshaped_2])



#----------- ACC-z denoising -------------

ACCz_sig = Input(shape=(1000, 1, ),
                   name='ACCz_input')

#----------- 1st Conv Layer -------------
conv_5 = Conv1D(filters=8, kernel_size=3, activation='relu')(ACCz_sig)
conv_5 = Conv1D(filters=8, kernel_size=3, activation='relu')(conv_5)
out_5 = Conv1D(filters=8, kernel_size=3, activation='relu')(conv_5)


#----------- 2nd Conv Layer -------------
conv_6 = Conv1D(filters=32, kernel_size=3, activation='relu')(out_5)
conv_6 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv_6)
out_6 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv_6)


last_2 = Flatten()(out_6)
Noise_output_2 = Dense(n_outputs_1, 
                        name='Noise_Z_output')(last_2)
Noise_output_Reshaped_2 = Reshape((1000,1))(Noise_output_2)

Clean_ppg = keras.layers.Subtract()([PPg_sig_3, Noise_output_Reshaped_2])
HR_value = Dense(n_outputs_2, 
                 name='HR_output')(last_)


#-------------- Model Fitting ---------------

model = Model(inputs = [ACCx_sig, PPg_sig, ACCy_sig, ACCz_sig], 
              outputs = [HR_value, Clean_ppg])



model.summary()

optimizer = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='mae', optimizer=optimizer, metrics=['mse'], 
              loss_weights=[1, 1])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_HR_output_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.25, 
                                            min_lr=0.00006)

callbacks_list = [learning_rate_reduction]


epochs = 100
batch_size = 32
verbose = 1
training_samples = 1766
validation_samples = 436



history = model.fit([X_train_1[:, :, 0:1], 
                    X_train_2, X_train_1[:, :, 1:2], X_train_1[:, :, 2:3]], 
                    [y_train, y_train_2[:, :, 0]],
                    validation_data = [[X_val_1, X_val_2] , 
                                       [y_val, y_val_2[:, :, 0]]],
                    epochs = epochs, verbose = verbose,
                    steps_per_epoch = int(training_samples/batch_size),
                    validation_steps = int(validation_samples/batch_size),
                    callbacks= callbacks_list)

model.save('Functional_api_total.h5')


########## Prediction ##########

prediction = model.predict([X_train_1, X_train_2])
predicted_HR = np.array(prediction[0])
predicted_Noise = np.array(prediction[1])


predicted_HR_test = sc.inverse_transform(predicted_HR_test);

########## Error Analysis ##########

mae = (predicted_HR - y_train).absolute()
mae = mae.sum()

val_mae = (predicted_HR_val - y_val).absolute()
val_mae = val_mae.sum()

test_mae = (predicted_HR_test - y_test).absolute()
test_mae = test_mae.sum()





# Plot mean_squared_error of HR
plt.figure(figsize=(24, 12))
plt.plot(model.history.history['HR_output_mean_squared_error'], 'r', label='mse of training HR')
plt.plot(history.history['val_HR_output_mean_squared_error'], 'b', label='mse of validation HR')
plt.title('mean_squared_error of HR')
plt.ylabel('mean_squared_error of HR')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()






