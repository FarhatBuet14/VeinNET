# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:17:26 2019

@author: User
"""

import numpy as np
import cv2
import os

def get_accumEdged(image):
    accumEdged = np.zeros(image.shape[:2], dtype="uint8")

    # loop over the blue, green, and red channels, respectively
    for chan in cv2.split(image):
        chan = cv2.medianBlur(chan, 3)
        edged = cv2.Canny(chan, 50, 150)
        accumEdged = cv2.bitwise_or(accumEdged, edged)
        
    return accumEdged



#rejected_folder = "./Rejected/left_norm_300/"
rejected_folder = "./Troughs_Model/model_AccuEdges/1/Rest_Neg_images/"
#datafile = "./Troughs_Model/test_data/test_gray.npz"

filenames = os.listdir(rejected_folder)


accumEdged_images = []
trough_points = []
image_names = []
image_paths = []

for image_filename in filenames:
    
    image = cv2.imread(rejected_folder + image_filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    accumEdged = get_accumEdged(gray)
#    accumEdged = accumEdged/255
#    gray = gray/255
    accumEdged_images.append(gray)
    image_names.append(image_filename)
    image_paths.append(rejected_folder + image_filename)


np.savez(datafile,
         X = accumEdged_images,
         test_names = image_names,
         test_paths = image_paths)


#####################  Read datasets and merge  ###################

#datafile = "./Troughs_Model/test_data/left_act.npz"
#dataset = np.load(datafile)
#
#X1 = dataset['X']
#n1 = dataset['test_names'].astype(str)
#p1 = dataset['test_paths'].astype(str)
#
#
#datafile = "./Troughs_Model/test_data/left_bag.npz"
#dataset = np.load(datafile)
#
#X2 = dataset['X']
#n2 = dataset['test_names'].astype(str)
#p2 = dataset['test_paths'].astype(str)
#
#
#datafile = "./Troughs_Model/test_data/left_ice.npz"
#dataset = np.load(datafile)
#
#X3 = dataset['X']
#n3 = dataset['test_names'].astype(str)
#p3 = dataset['test_paths'].astype(str)
#
#
#datafile = "./Troughs_Model/test_data/left_norm.npz"
#dataset = np.load(datafile)
#
#X4 = dataset['X']
#n4 = dataset['test_names'].astype(str)
#p4 = dataset['test_paths'].astype(str)
#
#
#datafile = "./Troughs_Model/test_data/right_norm.npz"
#dataset = np.load(datafile)
#
#X5 = dataset['X']
#n5 = dataset['test_names'].astype(str)
#p5 = dataset['test_paths'].astype(str)
#
#
#X = np.concatenate((X1, X2, X3, X4, X5), axis = 0)
#N = np.concatenate((n1, n2, n3, n4, n5), axis = 0)
#P = np.concatenate((p1, p2, p3, p4, p5), axis = 0)
#
#
#######################  Save Data  #####################
#
#save_datafile = "./Troughs_Model/test_data/test.npz"
#
#np.savez(save_datafile,
#         X = X,
#         test_names = N,
#         test_paths = P)
#
#img = X[32]
#cv2.imshow("ba", img)
#
#while True:    
#    key = cv2.waitKey(1)
#    if key == 27:
#        cv2.destroyAllWindows()
#        break






























