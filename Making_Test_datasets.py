# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:17:26 2019

@author: User
"""

import numpy as np
#import cv2
#import os
#
#def get_accumEdged(image):
#    accumEdged = np.zeros(image.shape[:2], dtype="uint8")
#
#    # loop over the blue, green, and red channels, respectively
#    for chan in cv2.split(image):
#        chan = cv2.medianBlur(chan, 3)
#        edged = cv2.Canny(chan, 50, 150)
#        accumEdged = cv2.bitwise_or(accumEdged, edged)
#        
#    return accumEdged
#
#
#
#rejected_folder = "./Rejected/right_norm_300/"
#datafile = "./Troughs_Model/test_data/right_norm.npz"
#
#
#filenames = os.listdir(rejected_folder)
#
#
#accumEdged_images = []
#trough_points = []
#image_names = []
#image_paths = []
#
#for image_filename in filenames:
#    
#    image = cv2.imread(rejected_folder + filenames[0])
#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    
#    accumEdged = get_accumEdged(gray)
#    
#    
#    accumEdged = gray.reshape((1, (240*300)))
#    accumEdged_images.append(accumEdged)
#    image_names.append(filenames[0])
#    image_paths.append(rejected_folder + filenames[0])
#
#
#test_data = np.zeros((len(image_names), (72000)))
#
#test_info = np.empty((len(image_names), 2)).astype(
#        np.dtype("a52"))
#
#
#for index in range(0, len(image_names)):
#    test_info[index, 0] = image_names[index]
#    test_info[index, 1] = image_paths[index]
#    test_data[index, 0:72000] = accumEdged_images[index][:]
#
#
#
#np.savez(datafile,
#         test_data = test_data,
#         test_info = test_info)
#



#####################  Read datasets and merge  ###################

datafile = "./Troughs_Model/test_data/left_act.npz"
dataset = np.load(datafile)

data_1 = dataset['test_data']
info_1 = dataset['test_info']


datafile = "./Troughs_Model/test_data/left_bag.npz"
dataset = np.load(datafile)

data_2 = dataset['test_data']
info_2 = dataset['test_info']


datafile = "./Troughs_Model/test_data/left_ice.npz"
dataset = np.load(datafile)

data_3 = dataset['test_data']
info_3 = dataset['test_info']


datafile = "./Troughs_Model/test_data/left_norm.npz"
dataset = np.load(datafile)

data_4 = dataset['test_data']
info_4 = dataset['test_info']


datafile = "./Troughs_Model/test_data/right_norm.npz"
dataset = np.load(datafile)

data_5 = dataset['test_data']
info_5 = dataset['test_info']


######################  Save Data  #####################

length = len(data_1) + len(data_2) + len(data_3) + len(data_4) + len(data_5)

data = np.zeros((length, data_1.shape[1]))

data[0 : len(data_1)] = data_1

data[len(data_1) : 
    (len(data_1)+len(data_2))] = data_2

data[(len(data_1)+len(data_2)) : 
    (len(data_1)+len(data_2)+len(data_3))] = data_3

data[(len(data_1)+len(data_2)+len(data_3)) : 
    (len(data_1)+len(data_2)+len(data_3)+len(data_4))] = data_4

data[(len(data_1)+len(data_2)+len(data_3)+len(data_4)) : length] = data_5



######################  Save Info  #####################



length = len(info_1) + len(info_2) + len(info_3) + len(info_4) + len(info_5)

info = np.zeros((length, info_1.shape[1])).astype(np.dtype("a52"))

info[0 : len(info_1)] = info_1

info[len(info_1) : 
    (len(info_1)+len(info_2))] = info_2

info[(len(info_1)+len(info_2)) : 
    (len(info_1)+len(info_2)+len(info_3))] = info_3

info[(len(info_1)+len(info_2)+len(info_3)) : 
    (len(info_1)+len(info_2)+len(info_3)+len(info_4))] =info_4

info[(len(info_1)+len(info_2)+len(info_3)+len(info_4)) : length] = info_5


save_datafile = "./Troughs_Model/test_data/test.npz"

np.savez(save_datafile,
         data = data,
         info = info)































