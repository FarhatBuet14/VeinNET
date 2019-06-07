# -*- coding: utf-8 -*-
"""
Created on Wed May 22 08:27:14 2019

@author: User
"""

############################# Libraries ##########################
#############################################################
#############################################################

import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
from operator import itemgetter
import math
import os
import pandas as pd 
import shutil



def get_accumEdged(image):
    accumEdged = np.zeros(image.shape[:2], dtype="uint8")

    # loop over the blue, green, and red channels, respectively
    for chan in cv2.split(image):
        chan = cv2.medianBlur(chan, 3)
        edged = cv2.Canny(chan, 50, 150)
        accumEdged = cv2.bitwise_or(accumEdged, edged)
        
    return accumEdged


def find_contour_needed(accumEdged, length_threshold = 100):
    
    contour_image = accumEdged
    
    cnts = cv2.findContours(contour_image, cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    cnts_nedded = []
    Length = []
    for c in cnts:
        if( cv2.arcLength(c, False) > length_threshold ):
            Length.append(cv2.arcLength(c, False))
            cnts_nedded.append(c)

    return cnts_nedded, Length




def cnt_concat(cnts_nedded):

    all_cnts = np.zeros((1, 1, 2))
    
    for cnt in cnts_nedded:
        all_cnts = np.append(all_cnts, cnt, axis = 0)
    
    all_cnts = all_cnts[1:, :, :]
    all_cnts = np.reshape(all_cnts, (all_cnts.shape[0], 2) )
    
    return all_cnts




def get_trough_points(image, all_cnts):
    
    cnt_x = all_cnts[:, 0]
    cnt_y = all_cnts[:, 1]

#    plt.figure(1, figsize=(12, 6))
#    plt.subplot(311)
#    plt.plot(cnt_x)
#    plt.ylabel('Controur_points')
#    
#    plt.figure(1, figsize=(12, 6))
#    plt.subplot(312)
#    plt.plot(cnt_y)
#    plt.ylabel('Controur_points')
#    
    grad = np.gradient(cnt_y)
#    
#    plt.figure(1, figsize=(12, 6))
#    plt.subplot(313)
#    plt.plot(grad)
#    plt.ylabel('Gradients')
#    plt.show()
#    
    troughs = []   
    for i in range(2, len(grad)-3):
        if((grad[i] < 0)  &  (grad[i-1] > 0) & (i>0)):
            
            troughs.append([cnt_x[i], cnt_y[i]])
    
    troughs = np.array(troughs)


    troughs = np.array(sorted(troughs, key=itemgetter(0)))
    
    
    ############  make one point #############
    not_okay = True  
    while(not_okay): 
        
        dists = np.zeros(( (len(troughs) - 1), 1))
        for i in range(0, len(troughs)-1):
            dists[i, 0] = math.hypot(troughs[i, 0] - troughs[i+1, 0], 
                              troughs[i, 1] - troughs[i+1, 1])   
        
        not_okay = True in (dists < 10)
        
        if(not_okay):
            
            troughs_final = []
            merge = False
            
            for i in range(0, len(troughs)-1):
                if((dists[i] < 10) & (merge == False)):
                    #print("merge")
                    troughs_final.append( ((troughs[i, :] + troughs[i+1, :]) / 2).astype(int) )
                    merge = True
                else:
                    if(merge == False):           
                        #print("not_merge")
                        troughs_final.append(troughs[i, :])
                    else:
                        merge = False
            
            troughs = np.array(troughs_final)
    
    copy = image.copy()
    for point in troughs:   
        point = point.astype(int)
        cv2.circle(copy, (point[0], 
                   point[1]), 
                   5, (0, 255, 0), -1) 
    
    ############  remove other points #############
    
    pointed = [[0,0]]
    for i in range(0, len(troughs)): 
        for j in range(0, len(troughs)):
            
            if( i != j ):
                
                dist = math.hypot(troughs[i, 0] - troughs[j, 0], 
                                        troughs[i, 1] - troughs[j, 1])
                
                if((dist > 20) & (dist < 40)):
                    pointed.append(troughs[i, :])
                    break
    
    pointed = np.array(pointed)
    troughs = pointed[1:, :]

    for point in troughs:   
        point = point.astype(int)
        cv2.circle(image, (point[0], 
                   point[1]), 
                   5, (0, 255, 0), -1)
    
    return image, copy, troughs


 

############## Main Code ############

#folder_path = "./HandVeinDatabase/left - 1200/act - 300/"
#datafile = "./Troughs_Model/train_data/left_act.npz"
    
folder_path = "./Troughs_Model/model_AccuEdges/1/Pos_images/"
datafile = "./Troughs_Model/model_AccuEdges/2/train_data/train_2.npz"
pred_datafile = "./Troughs_Model/model_AccuEdges/2/train_data/pred_data.npz"

pred_data = np.load(pred_datafile)

y_pred = pred_data['y_pred']
test_names = pred_data['test_names'].astype(str)













filenames = os.listdir(folder_path)
#filenames = filenames[:-1]


accumEdged_images = []
trough_points = []
image_names = []
image_paths = []
rejected_image_names = []
pred = []
found = False
for image_filename in filenames:
    image = cv2.imread(folder_path + image_filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    accumEdged = get_accumEdged(gray)
    
#    length_threshold = 20
#    [cnts_nedded, length_cnt] = find_contour_needed(accumEdged, length_threshold)
#    all_cnts = cnt_concat(cnts_nedded)
#    cnt_image = cv2.drawContours(gray.copy(), 
#                                 np.array(all_cnts).reshape((-1,1,2)).astype(np.int32),
#                                 -1, (0,255,0), 3)
#    
#    trough_image , bef_image, troughs = np.array(get_trough_points(gray.copy(), all_cnts))
    
    
    
    
#    if(len(troughs) == 3):
#        troughs = troughs.reshape((1, 6))
    accumEdged_images.append(accumEdged)
#        trough_points.append(troughs)
    image_names.append(image_filename)
    image_paths.append(folder_path + image_filename)
    for count in range(0, len(test_names)):   
        if((image_filename == test_names[count]) & (found == False)):
            pred.append(y_pred[count])
            found = True
    
    found = False
        
#    else:
#        rejected_image_names.append(image_filename)



#np.savez(datafile,
#         X = accumEdged_images,
#         Y = trough_points,
#         train_names = image_names,
#         train_paths = image_paths)

#np.savez(datafile,
#         X = accumEdged_images,
#         Y = pred,
#         train_names = image_names,
#         train_paths = image_paths)


#####################  Read datasets and merge  ###################

#datafile = "./Troughs_Model/train_data/left_act.npz"
#dataset = np.load(datafile)
#
#X1 = dataset['X']
#Y1 = dataset['Y']
#n1 = dataset['train_names'].astype(str)
#p1 = dataset['train_paths'].astype(str)
#
#
#datafile = "./Troughs_Model/train_data/left_bag.npz"
#dataset = np.load(datafile)
#
#X2 = dataset['X']
#Y2 = dataset['Y']
#n2 = dataset['train_names'].astype(str)
#p2 = dataset['train_paths'].astype(str)
#
#
#datafile = "./Troughs_Model/train_data/left_ice.npz"
#dataset = np.load(datafile)
#
#X3 = dataset['X']
#Y3 = dataset['Y']
#n3 = dataset['train_names'].astype(str)
#p3 = dataset['train_paths'].astype(str)
#
#
#datafile = "./Troughs_Model/train_data/left_norm.npz"
#dataset = np.load(datafile)
#
#X4 = dataset['X']
#Y4 = dataset['Y']
#n4 = dataset['train_names'].astype(str)
#p4 = dataset['train_paths'].astype(str)
#
#
#datafile = "./Troughs_Model/train_data/right_norm.npz"
#dataset = np.load(datafile)
#
#X5 = dataset['X']
#Y5 = dataset['Y']
#n5 = dataset['train_names'].astype(str)
#p5 = dataset['train_paths'].astype(str)
#
#
#X = np.concatenate((X1, X2, X3, X4, X5), axis = 0)
#Y = np.concatenate((Y1, Y2, Y3, Y4, Y5), axis = 0)
#Y = Y.reshape((Y.shape[0], 6))
#N = np.concatenate((n1, n2, n3, n4, n5), axis = 0)
#P = np.concatenate((p1, p2, p3, p4, p5), axis = 0)
#
#
########################  Save Data  #####################
#
#save_datafile = "./Troughs_Model/train_data/train.npz"
#
#np.savez(save_datafile,
#         X = X,
#         Y = Y,
#         train_names = N,
#         train_paths = P)


#img = accumEdged_images[150]
#cv2.imshow("ba", img)
#
#while True:    
#    key = cv2.waitKey(1)
#    if key == 27:
#        cv2.destroyAllWindows()
#        break






































