# -*- coding: utf-8 -*-
"""
Created on Tue May 14 04:30:28 2019

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
#    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

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
    
    troughs = []   
    for i in range(2, len(grad)-3):
        if((grad[i] < 0)  &  (grad[i-1] > 0) & (i>0)):
            
            troughs.append([cnt_x[i], cnt_y[i]])
    
    troughs = np.array(troughs)


#    troughs = np.array(sorted(troughs, key=itemgetter(0)))
    
    
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
    
    
    pointed = [[0,0]]
    for i in range(0, len(troughs)-1): 
        for j in range(i+1, len(troughs)):
            
            if( (int(troughs[i, 0]) in (np.array(pointed)[:, 0])) | 
                (int(troughs[j, 0]) in (np.array(pointed)[:, 0])) ):
                    None
            else:
                
                dist = math.hypot(troughs[i, 0] - troughs[j, 0], 
                                        troughs[i, 1] - troughs[j, 1])
#                
#                
#                    if( (troughs[i, 0] > 50)  &  (troughs[i, 1] < 150) &
#                       (troughs[i, 0] < 200) & (troughs[i, 1] > 75)):
                if((dist > 20) & (dist < 40)):       
                    pointed.append(troughs[i, :])
                    pointed.append(troughs[j, :])
    
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

image_name = "p14_left_act_3.bmp"
image = cv2.imread(folder_path + image_name, )
orig = image.copy()

gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
accumEdged = get_accumEdged(gray)
length_threshold = 20
[cnts_nedded, length_cnt] = find_contour_needed(accumEdged, length_threshold)
all_cnts = cnt_concat(cnts_nedded)
trough_image , bef_image, troughs = np.array(get_trough_points(gray.copy(), all_cnts))

cnt_image = cv2.drawContours(gray.copy(), 
                             np.array(all_cnts).reshape((-1,1,2)).astype(np.int32),
                             -1, (0,255,0), 3)


all_img = cv2.hconcat((accumEdged, cnt_image))
all_img = cv2.hconcat((all_img, bef_image))
all_img = cv2.hconcat((all_img, trough_image))


############################ distance Calculatio  ###############################
#troughs = np.array(troughs)
#dists = []
#for i in range(0, len(troughs)-1): 
#    dists.append(math.hypot(troughs[i, 0] - troughs[i+1, 0], troughs[i, 1] - troughs[i+1, 1]))
#
#
#print(dists)




cv2.imshow(image_name, all_img)     

while True:    
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
        break














