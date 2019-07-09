############################  Import Libraries  ###############################
###############################################################################

import numpy as np

##############################  Import Data  ##################################
###############################################################################

# Input Data Folders
train_Output_data = "./Model_Output/"
data_folder = "./Data/"
Aug_train_data_folder = './Data/Augmented_Train_data/'

# Output Data Folders
weightFile = train_Output_data + 'WeightFile_best.hdf5'
saved_model_File = train_Output_data + 'Saved_Model.h5'

# Import Main Data
data = np.load(data_folder + 'train_test_data_without_augmetation.npz') 
X_train_names = data['X_train_names'].astype(str)
X_train_bmp = data['X_train_bmp']
X_train_gray = data['X_train_gray']
X_train = data['X_train']
y_train = data['y_train']

# Import Augmented Data
data = np.load(data_folder + "Augmented_Train_data.npz") 
X_train_aug_names = data['X_train_aug_names'].astype(str)
X_train_aug_bmp = data['X_train_aug_bmp']
X_train_aug_gray = data['X_train_aug_gray']
X_train_aug_accu = data['X_train_aug_accu']
y_train_aug = data['y_train_aug'].reshape((X_train_aug_accu.shape[0], 4))

# Concatenate main data and augmented data
X_names = np.concatenate((X_train_names, X_train_aug_names), axis = 0)
X_bmp = np.concatenate((X_train_bmp, X_train_aug_bmp), axis = 0)
X_gray = np.concatenate((X_train_gray, X_train_aug_gray), axis = 0)
X_accu = np.concatenate((X_train, X_train_aug_accu), axis = 0)
y = np.concatenate((y_train, y_train_aug), axis = 0)

# Normalizing Input
X_accu = X_accu / 255

# Train Test Splitting
random_seed = 0
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_accu, y, test_size = 0.3, 
                                                  random_state=random_seed)

# Reshape the train and validation set for making enterable to the model
X_train = X_train.reshape((X_train.shape[0], 240, 300, 1))
X_val = X_val.reshape((X_val.shape[0], 240, 300, 1))

num_pred_value = y_train.shape[1]

########################## Import libraries for Model  ########################
###############################################################################

import numpy as np
import keras
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D
from keras.layers import BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.utils import plot_model
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import keras.backend as K
K.set_image_data_format('channels_last')

#################### Identity Block library for ResNet Model  #################
###############################################################################

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. We'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step:
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

####################### Conv Block library for ResNet Model  ##################
###############################################################################

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(F3, (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

############################### ResNet Model library###########################
###############################################################################
    
def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape, name = 'input')

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL 
    X = AveragePooling2D(pool_size=(2, 2), name= "avg_pool", padding='same')(X)

    # output layer
    X = Flatten()(X)
    X = Dropout(0.3)(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

#######################  Create Model and Compilation #########################
###############################################################################

model = ResNet50(input_shape = (240,300, 1), classes = num_pred_value)
print(model.summary())
plot_model(model, to_file='Model Layer Summary.png')

# Compiling the Model
optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='mae', optimizer=optimizer, metrics=['mse'])

# Defining the Checkpoints
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.25, 
                                            min_lr=0.00001)
weights  = ModelCheckpoint(weightFile, monitor = 'val_loss' )
callbacks = [weights, learning_rate_reduction]

epochs = 2
batch_size = 8
verbose = 1
history = model.fit(X_train, y_train,
                    validation_data = [X_val , y_val],
                    epochs = epochs, batch_size = batch_size,
                    verbose = verbose, callbacks= callbacks)

model.save(saved_model_File)
