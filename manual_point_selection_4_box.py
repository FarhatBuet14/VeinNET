# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:30:15 2019

@author: User
"""

import cv2
import numpy as np
import os

def get_accumEdged(image):
    accumEdged = np.zeros(image.shape[:2], dtype="uint8")

    # loop over the blue, green, and red channels, respectively
    for chan in cv2.split(image):
        chan = cv2.medianBlur(chan, 3)
        edged = cv2.Canny(chan, 50, 150)
        accumEdged = cv2.bitwise_or(accumEdged, edged)
        
    return accumEdged



rest_neg_folder = './Troughs_Model/model_AccuEdges/1/Rest_Neg_images/'
filenames = os.listdir(rest_neg_folder)

datafile = './Troughs_Model/model_AccuEdges/1/neg_img_data.npz'    

points = []
point = []
names = []
images = []
count = 0
for file in filenames:

    def mouse_drawing(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(str(x) + " , " + str(y))
            point.append(x)
            point.append(y)
    
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_drawing)
        
    img = cv2.imread(rest_neg_folder + file)    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    accumEdged = get_accumEdged(gray)

    cv2.imshow("Frame", img)
       
    while True:    
        key = cv2.waitKey(1)
        if key == 27:
            images.append(accumEdged)
            points.append(point)
            names.append(file)
            point = []
            count += 1
            print("done - " + str(count))
            cv2.destroyAllWindows()
            break

points = np.array(points)

np.savez(datafile,
         X = images,
         Y = points,
         train_names = names)


#data = np.load(datafile)
#
#X = data['X']
#Y = data['Y']
#train_names = data['train_names'].astype(str)
#
#
#img = X[200]
#cv2.imshow("Frame", img)
#       
#while True:    
#    key = cv2.waitKey(1)
#    if key == 27:
#        cv2.destroyAllWindows()
#        break











#import numpy as np
#import os
#import cv2
#
#datafile = './Troughs_Model/model_AccuEdges/1/train_data/neg_img_data.npz'    
#rest_neg_folder = './Troughs_Model/model_AccuEdges/1/Rest_Neg_images/'
#filenames = os.listdir(rest_neg_folder)
#
#
#data = np.load(datafile)
#
#X = data['X']
#Y = data['Y']
#train_names = data['train_names'].astype(str)
#
#gray_img = []
#for name in train_names:
#    img = cv2.imread(rest_neg_folder + name)
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    gray_img.append(gray)
#
#
#np.savez(datafile,
#         X = X,
#         X_gray = gray_img,
#         Y = Y,
#         train_names = train_names)
















