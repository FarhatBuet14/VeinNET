# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:13:32 2019

@author: User
"""

import numpy as np
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage
import pandas as pd


def draw_troughs(img, points):
    
    points = points.reshape((3, 2))
    
    for point in points:   
        point = np.array(point).astype(int)
        cv2.circle(img, (point[0], 
                   point[1]), 
                   5, (255, 255, 0), -1)

    return img

#################### Environment & Variables ############################
from keras import backend as K
K.set_image_data_format('channels_last')

datafile = "./Troughs_Model/model_AccuEdges/6/train_data/data.npz"


####################### Loading the Data ################################

dataset = np.load(datafile) 

X = dataset['X']
y = dataset['y']

X_test = dataset['X_test']
X_test_gray = dataset['X_test_gray']
y_test = dataset['y_test']
test_names = dataset['test_names']



images = X.reshape((X.shape[0], X.shape[1],
                           X.shape[2], 1))
keypoints = y.reshape((y.shape[0], 3, 2))


points = []
for point in keypoints:
    points.append(point)






images_aug_1 , keypoints_aug_1 = iaa.Affine(rotate=30)(images=images[:75], 
                                      keypoints=points[:75])

images_aug_2 , keypoints_aug_2 = iaa.Affine(rotate=50)(images=images[75:150], 
                                      keypoints=points[75:150])

images_aug_3 , keypoints_aug_3 = iaa.Affine(rotate=60)(images=images[150:225], 
                                      keypoints=points[150:225])

images_aug_4 , keypoints_aug_4 = iaa.Affine(rotate=80)(images=images[225:300], 
                                      keypoints=points[225:300])

images_aug_5 , keypoints_aug_5 = iaa.Affine(rotate=-30)(images=images[300:375], 
                                      keypoints=points[300:375])

images_aug_6 , keypoints_aug_6 = iaa.Affine(rotate=-50)(images=images[375:450], 
                                      keypoints=points[375:450])

images_aug_7 , keypoints_aug_7 = iaa.Affine(rotate=-60)(images=images[450:525], 
                                      keypoints=points[450:525])

images_aug_8 , keypoints_aug_8 = iaa.Affine(rotate=-80)(images=images[525:600], 
                                      keypoints=points[525:600])


images_aug_ = np.concatenate((images_aug_1, images_aug_2, images_aug_3, images_aug_4,
                             images_aug_5, images_aug_6, images_aug_7, images_aug_8), axis = 0)

keypoints_aug_ = np.concatenate((keypoints_aug_1, keypoints_aug_2, 
                                keypoints_aug_3, keypoints_aug_4,
                                keypoints_aug_5, keypoints_aug_6, 
                                keypoints_aug_7, keypoints_aug_8), axis = 0)



images_lr , keypoints_lr = iaa.Fliplr(1.0)(images=images[600:1212], 
                                       keypoints=points[600:1212])

images_aug_1 , keypoints_aug_1 = iaa.Affine(rotate=30)(images=images_lr[:75], 
                                      keypoints=keypoints_lr[:75])

images_aug_2 , keypoints_aug_2 = iaa.Affine(rotate=50)(images=images_lr[75:150], 
                                      keypoints=keypoints_lr[75:150])

images_aug_3 , keypoints_aug_3 = iaa.Affine(rotate=60)(images=images_lr[150:225], 
                                      keypoints=keypoints_lr[150:225])

images_aug_4 , keypoints_aug_4 = iaa.Affine(rotate=80)(images=images_lr[225:300], 
                                      keypoints=keypoints_lr[225:300])

images_aug_5 , keypoints_aug_5 = iaa.Affine(rotate=-30)(images=images_lr[300:375], 
                                      keypoints=keypoints_lr[300:375])

images_aug_6 , keypoints_aug_6 = iaa.Affine(rotate=-50)(images=images_lr[375:450], 
                                      keypoints=keypoints_lr[375:450])

images_aug_7 , keypoints_aug_7 = iaa.Affine(rotate=-60)(images=images_lr[450:525], 
                                      keypoints=keypoints_lr[450:525])

images_aug_8 , keypoints_aug_8 = iaa.Affine(rotate=-80)(images=images_lr[525:612], 
                                      keypoints=keypoints_lr[525:612])


images_aug = np.concatenate((images_aug_1, images_aug_2, images_aug_3, images_aug_4,
                             images_aug_5, images_aug_6, images_aug_7, images_aug_8), axis = 0)

keypoints_aug = np.concatenate((keypoints_aug_1, keypoints_aug_2, 
                                keypoints_aug_3, keypoints_aug_4,
                                keypoints_aug_5, keypoints_aug_6, 
                                keypoints_aug_7, keypoints_aug_8), axis = 0)


images_aug = np.concatenate((images_aug_, images_aug), axis = 0)

keypoints_aug = np.concatenate((keypoints_aug_, keypoints_aug), axis = 0)


images_aug = images_aug.reshape((images_aug.shape[0], images_aug.shape[1], images_aug.shape[2]))

keypoints_aug = keypoints_aug.reshape((keypoints_aug.shape[0], 6))



images = np.concatenate((X, images_aug), axis = 0)

points = np.concatenate((y, keypoints_aug), axis = 0)




#img = draw_troughs(images[2332], points[2332])
#
#
#cv2.imshow("flip_lr", img)
#
#while True:
#    key = cv2.waitKey(1)
#    if(key == 27):
#        cv2.destroyAllWindows()
#        break



def shuffleData(X,y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X_shuffled = X[randomize]
    y_shuffled = y[randomize]
    return X_shuffled,y_shuffled


images, points = shuffleData(images,points)





new_datafile =  "./Troughs_Model/model_AccuEdges/6/data_with_aug.npz"

np.savez(new_datafile,
         X=images,
         y=points,
         X_test = X_test,
         X_test_gray = X_test_gray,
         y_test = y_test,
         test_names = test_names)


















































