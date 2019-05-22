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

    grad = np.gradient(cnt_y)

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



folder_path = "./HandVeinDatabase/left - 1200/act - 300/"
datafile = "./Troughs_Model/train_data/left_act.npz"


filenames = os.listdir(folder_path)
filenames = filenames[:-1]


accumEdged_images = []
trough_points = []
image_names = []
image_paths = []

for image_filename in filenames:
    image = cv2.imread(folder_path + image_filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    accumEdged = get_accumEdged(gray)
    
    length_threshold = 20
    [cnts_nedded, length_cnt] = find_contour_needed(accumEdged, length_threshold)
    all_cnts = cnt_concat(cnts_nedded)
    cnt_image = cv2.drawContours(gray.copy(), 
                                 np.array(all_cnts).reshape((-1,1,2)).astype(np.int32),
                                 -1, (0,255,0), 3)
    
    trough_image , bef_image, troughs = np.array(get_trough_points(gray.copy(), all_cnts))

    if(len(troughs) == 3):
        accumEdged = accumEdged.reshape((1, (240*300)))
        troughs = troughs.reshape((1, 6))
        accumEdged_images.append(accumEdged)
        trough_points.append(troughs)
        image_names.append(image_filename)
        image_paths.append(folder_path + image_filename)
        

train_data = np.zeros((len(image_names), (6 + 72000))).astype(float)
train_info = np.zeros((len(image_names), 2)).astype(str)

for index in range(0, len(image_names)):
    train_info[index, 0] = image_names[index]
    train_info[index, 1] = image_paths[index]
    train_data[index, 0:72000] = accumEdged_images[index][:]
    train_data[index, 72000:] = trough_points[index][:]



np.savez(datafile,
         train_data = train_data,
         train_info = train_info)


#####################  Read datasets and merge  ###################
#
#datafile = "./Troughs_Model/train_data/left_act.npz"
#dataset = np.load(datafile)
#
#data_1 = dataset['train_data']
#info_1 = dataset['train_info']
#
#
#datafile = "./Troughs_Model/train_data/left_bag.npz"
#dataset = np.load(datafile)
#
#data_2 = dataset['train_data']
#info_2 = dataset['train_info']
#
#
#datafile = "./Troughs_Model/train_data/left_ice.npz"
#dataset = np.load(datafile)
#
#data_3 = dataset['train_data']
#info_3 = dataset['train_info']
#
#
#datafile = "./Troughs_Model/train_data/left_norm.npz"
#dataset = np.load(datafile)
#
#data_4 = dataset['train_data']
#info_4 = dataset['train_info']
#
#
#datafile = "./Troughs_Model/train_data/right_norm.npz"
#dataset = np.load(datafile)
#
#data_5 = dataset['train_data']
#info_5 = dataset['train_info']
#
#
#######################  Save Data  #####################
#
#length = len(data_1) + len(data_2) + len(data_3) + len(data_4) + len(data_5)
#
#data = np.zeros((length, data_1.shape[1]))
#
#data[0 : len(data_1)] = data_1
#
#data[len(data_1) : 
#    (len(data_1)+len(data_2))] = data_2
#
#data[(len(data_1)+len(data_2)) : 
#    (len(data_1)+len(data_2)+len(data_3))] = data_3
#
#data[(len(data_1)+len(data_2)+len(data_3)) : 
#    (len(data_1)+len(data_2)+len(data_3)+len(data_4))] = data_4
#
#data[(len(data_1)+len(data_2)+len(data_3)+len(data_4)) : length] = data_5
#
#
#
#######################  Save Info  #####################
#
#
#
#length = len(info_1) + len(info_2) + len(info_3) + len(info_4) + len(info_5)
#
#info = np.zeros((length, info_1.shape[1])).astype(str)
#
#info[0 : len(info_1)] = info_1
#
#info[len(info_1) : 
#    (len(info_1)+len(info_2))] = info_2
#
#info[(len(info_1)+len(info_2)) : 
#    (len(info_1)+len(info_2)+len(info_3))] = info_3
#
#info[(len(info_1)+len(info_2)+len(info_3)) : 
#    (len(info_1)+len(info_2)+len(info_3)+len(info_4))] =info_4
#
#info[(len(info_1)+len(info_2)+len(info_3)+len(info_4)) : length] = info_5
#
#
#save_datafile = "./Troughs_Model/train_data/train.npz"
#
#np.savez(save_datafile,
#         data = data,
#         info = info)
#
#
#
#










































