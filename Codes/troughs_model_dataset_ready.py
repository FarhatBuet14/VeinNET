# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:02:45 2019

@author: User
"""

import numpy as np


# Function to shuffle two arrays in Unison

def shuffleData(X,y,names):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X_shuffled = X[randomize]
    y_shuffled = y[randomize]
    names_shuffled = names[randomize]
    return X_shuffled,y_shuffled, names_shuffled



#train_datafile = "./Troughs_Model/model_AccuEdges/1/train_data/train.npz"
train_datafile = "./Troughs_Model/model_AccuEdges/6/train_data/train_1.npz"
train_datafile_1 = "./Troughs_Model/model_AccuEdges/6/train_data/train_2.npz"
neg_datafile = './Troughs_Model/model_AccuEdges/6/neg_img_data.npz'    
#test_datafile = "./Troughs_Model/model_AccuEdges/test_data/test.npz"
datafile = "./Troughs_Model/model_AccuEdges/6/data.npz"


train_dataset = np.load(train_datafile)
#test_dataset = np.load(test_datafile)

X_train = train_dataset['X']
y_train = train_dataset['Y']
train_names = train_dataset['train_names'].astype(str)
train_paths = train_dataset['train_paths'].astype(str)

#---------------------------------------------------------------------
train_dataset_1 = np.load(train_datafile_1)

X_train_1 = train_dataset_1['X']
y_train_1 = train_dataset_1['Y']
train_names_1 = train_dataset_1['train_names'].astype(str)
train_paths_1 = train_dataset_1['train_paths'].astype(str)


#----------------------------------------------------------------------

data = np.load(neg_datafile)

X = data['X']
X_gray = data['X_gray']
Y = data['Y']
train_names_bla = data['train_names'].astype(str)


X_test = X
X_test_gray = X_gray
y_test = Y
test_names = train_names_bla


#----------------------------------------------------------------------


X = np.concatenate((X_train, X_train_1), axis = 0)
y = np.concatenate((y_train, y_train_1), axis = 0)
names = np.concatenate((train_names, train_names_1), 
                       axis = 0)
#paths = np.concatenate((train_paths,train_paths_1), axis = 0)


#X_test = test_dataset['X']
#test_names = test_dataset['test_names'].astype(str)
#test_paths = test_dataset['test_paths'].astype(str)

#X_train = X_train/255
#X_test = X_test/255


#X_train, y_train, train_names, train_paths = shuffleData(X_train,y_train, 
#                                                         train_names, train_paths)

X_train, y_train, train_names= shuffleData(X,y,names)


#random_seed = 0
#from sklearn.model_selection import train_test_split
##X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.4, 
##                                                  random_state=random_seed)
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.4, 
#                                                  random_state=random_seed)



################# Storing the Dataset to an npz file #######################

#import cv2
#img = X[1000]
#cv2.imshow("ba", img)
#
#while True:    
#    key = cv2.waitKey(1)
#    if key == 27:
#        cv2.destroyAllWindows()
#        break


#np.savez(datafile,
#         
#         X_train=X_train, X_test=X_test,
#         train_names = train_names,
#         train_paths = train_paths,
#         
#         y_train=y_train,
#         test_names = test_names,
#         test_paths = test_paths,
#         
#         X_val=X_val, y_val=y_val)

#np.savez(datafile,
#         X_train=X_train,
#         y_train=y_train,
#         train_names = train_names,
#         X_val=X_val, y_val=y_val,
#         X_test = X_test,
#         X_test_gray = X_test_gray,
#         y_test = y_test,
#         test_names = test_names)


np.savez(datafile,
         X=X_train,
         y=y_train,
         names = train_names,
         X_test = X_test,
         X_test_gray = X_test_gray,
         y_test = y_test,
         test_names = test_names)










