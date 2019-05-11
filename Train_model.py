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
#X_test = sc.fit_transform(X_test)
y_train = sc.fit_transform(y_train)
#y_test = sc.fit_transform(y_test)


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


#(1-1766) - train Data     -- 12 Subject
#(1766-3200) - test Data   -- 11 Subject

#train_1 - 148
#train_2 - 148
#train_3 - 140
#total Validation - 436
#Validation Data   -- 3 Subject





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


####################################################################
#########################   Model-2   ##############################

#----------- 1st Conv Layer -------------

conv_11 = Conv1D(filters=16, kernel_size=3, activation='relu')(Clean_ppg)
conv_11 = Conv1D(filters=16, kernel_size=3, activation='relu')(conv_11)
out_11 = Conv1D(filters=16, kernel_size=3, activation='relu')(conv_11)


##----------- 2nd Conv Layer -------------

conv_12 = Conv1D(filters=32, kernel_size=3, activation='relu')(out_11)
conv_12 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv_12)
out_12 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv_12)


#----------- Model_2 Output -------------
last_ = Flatten()(out_12)
#last_ = Dense(1024, kernel_regularizer=regularizers.l2(0.001), 
#              activation='relu')(last_)
#last_ = Dropout(0.5)(last_)
last_ = Dense(512, kernel_regularizer=regularizers.l2(0.001), 
              activation='relu')(last_)
last_ = Dropout(0.5)(last_)
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

#validation_split = 0.1
#validation_splitting = int (training_samples * 0.1) + 1
#train_splitting = training_samples - validation_splitting


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

#from keras.models import load_model
#
#model = load_model('Functional_api_total.h5')
#
#import numpy as np
#import matplotlib.pyplot as plt

########## Prediction ##########

prediction = model.predict([X_train_1, X_train_2])
predicted_HR = np.array(prediction[0])
predicted_Noise = np.array(prediction[1])

prediction_val = model.predict([X_val_1, X_val_2])
predicted_HR_val = np.array(prediction_val[0])
predicted_Noise_val = np.array(prediction_val[1])

prediction_test = model.predict([X_test_1, X_test_2])
predicted_HR_test = np.array(prediction_test[0])
predicted_Noise_test = np.array(prediction_test[1])




predicted_HR = sc.inverse_transform(predicted_HR);
predicted_HR_val = sc.inverse_transform(predicted_HR_val);
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

# Plot mean_squared_error of Noise
plt.figure(figsize=(24, 12))
plt.plot(history.history['Noise_output_mean_squared_error'], 'r', label='mse of training Noise')
plt.plot(history.history['val_Noise_output_mean_squared_error'], 'b', label='mse of validation Noise')
plt.title('mean_squared_error of Noise')
plt.ylabel('mean_squared_error of Noise')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()


# Plot Losses (MAE) of HR
plt.figure(figsize=(24, 12))
plt.plot(history.history['HR_output_loss'], 'r--', label='Loss of training HR')
plt.plot(history.history['val_HR_output_loss'], 'b--', label='Loss of validation HR')
plt.title(' Losses (MAE) of HR')
plt.ylabel('Loss of HR')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()



# Plot Losses (MAE) of Noise
plt.figure(figsize=(24, 12))
plt.plot(history.history['Noise_output_loss'], 'r--', label='Loss of training Noise')
plt.plot(history.history['val_Noise_output_loss'], 'b--', label='Loss of validation Noise')
plt.title(' Losses (MAE) of Noise')
plt.ylabel('Loss of Noise')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()


# Overall Test Set Evaluation
evl = model.evaluate([X_test_1, X_test_2], [y_test, y_test_2[:, :, 0]], batch_size = batch_size, verbose = verbose)


# Suject wise Test Set Evaluation
test_evaluation_list = [[],[],[],[],[],[],[],[],[],[],[]]
# 1st_Sub
test_evaluation_list[0] = model.evaluate([X_test_1[0:142, :, :], X_test_2[0:142, :, :]], [y_test[0:142, :], y_test_2[0:142, :, 0]], batch_size = batch_size, verbose = verbose)
# 2nd_Sub
test_evaluation_list[1] = model.evaluate([X_test_1[142:279, :, :], X_test_2[142:279, :, :]], [y_test[142:279, :], y_test_2[142:279, :, 0]], batch_size = batch_size, verbose = verbose)
# 3rd_Sub
test_evaluation_list[2] = model.evaluate([X_test_1[279:423, :, :], X_test_2[279:423, :, :]], [y_test[279:423, :], y_test_2[279:423, :, 0]], batch_size = batch_size, verbose = verbose)
# 4th_Sub
test_evaluation_list[3] = model.evaluate([X_test_1[423:575, :, :], X_test_2[423:575, :, :]], [y_test[423:575, :], y_test_2[423:575, :, 0]], batch_size = batch_size, verbose = verbose)
# 5th_Sub
test_evaluation_list[4] = model.evaluate([X_test_1[575:675, :, :], X_test_2[575:675, :, :]], [y_test[575:675, :], y_test_2[575:675, :, 0]], batch_size = batch_size, verbose = verbose)
# 6th_Sub
test_evaluation_list[5] = model.evaluate([X_test_1[675:832, :, :], X_test_2[675:832, :, :]], [y_test[675:832, :], y_test_2[675:832, :, 0]], batch_size = batch_size, verbose = verbose)
# 7th_Sub
test_evaluation_list[6] = model.evaluate([X_test_1[832:964, :, :], X_test_2[832:964, :, :]], [y_test[832:964, :], y_test_2[832:964, :, 0]], batch_size = batch_size, verbose = verbose)
# 8th_Sub
test_evaluation_list[7] = model.evaluate([X_test_1[964:1106, :, :], X_test_2[964:1106, :, :]], [y_test[964:1106, :], y_test_2[964:1106, :, 0]], batch_size = batch_size, verbose = verbose)
# 9th_Sub
test_evaluation_list[8] = model.evaluate([X_test_1[1106:1227, :, :], X_test_2[1106:1227, :, :]], [y_test[1106:1227, :], y_test_2[1106:1227, :, 0]], batch_size = batch_size, verbose = verbose)
# 10th_Sub
test_evaluation_list[9] = model.evaluate([X_test_1[1227:1327, :, :], X_test_2[1227:1327, :, :]], [y_test[1227:1327, :], y_test_2[1227:1327, :, 0]], batch_size = batch_size, verbose = verbose)
# 11th_Sub
test_evaluation_list[10] = model.evaluate([X_test_1[1327:1434, :, :], X_test_2[1327:1434, :, :]], [y_test[1327:1434, :], y_test_2[1327:1434, :, 0]], batch_size = batch_size, verbose = verbose)
#
#
## Prediction
#prediction = model.predict([X_test_1, X_test_2])
#predicted_HR = np.array(prediction[0])
#predicted_Noise = np.array(prediction[1])
#
#
#predicted_HR_list = predicted_HR.tolist()
#y_test_list = y_test.tolist()
#
#
#plt.figure(figsize=(24, 12))
#plt.scatter(predicted_HR.tolist(), 'r--', label='predicted_HR')
#plt.scatter(y_test.tolist(), 'b--', label='test HR')
#plt.title(' Prediction Plot')
#plt.ylabel('HR')
#plt.xlabel('Window')
#plt.ylim(0)
#plt.legend()
#plt.show()














































































