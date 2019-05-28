import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
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



datafile = "./Troughs_Model/model_AccuEdges/5/data.npz"
#gray_datafile = "./Troughs_Model/model_AccuEdges/2/test_data/test_gray.npz"
weightFile = './Troughs_Model/model_AccuEdges/5/WeightFile_best.hdf5'
prediction_fldr = './Troughs_Model/model_AccuEdges/5/prediction/'


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

X_test = X_test / 255

num_pred_value = 6


################## Defining & Loading Model #############################

#(batch, rows, cols, channels)
input_layer = Input(shape=(240,300,1), name='Input_Layer')

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

#X_test_gray = X_test_gray * 255


count = 0
for sample in range(0, len(y_pred)):
    
    image = X_test_gray[sample, :, :, 0].reshape(
            (X_test_gray.shape[1], X_test_gray.shape[2])).astype(np.uint8)
    
    
    if image is not None:
        for point in y_pred[sample, :, :]:   
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





y_pred = y_pred.reshape((y_pred.shape[0], 6))

err = abs(y_test - y_pred)

err = np.sum(err, axis = 1)/6

err_avg = sum(err)/len(err)


plt.figure(figsize = (20, 10))
plt.plot(err)
plt.show()



pred_datafile = "./Troughs_Model/model_AccuEdges/5/pred_data_4.npz"
np.savez(pred_datafile,
         y_pred = y_pred,
         err = err,
         err_avg = err_avg,
         test_names = test_names)









