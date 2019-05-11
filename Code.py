# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 19:33:05 2019

@author: User
"""

import numpy as np
import imutils
import cv2
#import matplotlib.pyplot as plt



total_samples = 5


k = 1
for i in range(1, int(total_samples/2)+1):
    for j in range(1, 4):
        # load the image, clone it, and setup the mouse callback function
        image = cv2.imread("p" + str(i) + "_left_act_" + str(j) + ".bmp")
        accumEdged = np.zeros(image.shape[:2], dtype="uint8")
        
        # loop over the blue, green, and red channels, respectively
        for chan in cv2.split(image):
            chan = cv2.medianBlur(chan, 3)
            edged = cv2.Canny(chan, 150, 200)
            accumEdged = cv2.bitwise_or(accumEdged, edged)
                
        # show the accumulated edge map
        cv2.imshow("Edge Map", accumEdged)
        
        while True:    
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break
        
        
        contour_image = accumEdged

        cnts = cv2.findContours(contour_image, cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        
        cnts_nedded = []
        for c in cnts:
            if(cv2.contourArea(c) > 100):
                cnts_nedded.append(c)
        
        
        points_value = []
        for x in range(0, accumEdged.shape[1]):
            points_value.append(accumEdged[(accumEdged.shape[0]-10), x])
        
        for i in range(0, len(points_value)):
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#        indexes = [index for index in range(len(points_value)) 
#                            if points_value[index] == 255]
        
        if(len(indexes) > 0):
            point = [(accumEdged.shape[0]-10), int(sum(indexes)/len(indexes))]
            
            
            #points_value.count(255)
            print("p" + str(i) + "_left_act_" + str(j) + " === " 
                  + str(len(indexes)) + "   ---  (done) ")

            
            orig = image.copy()
            cv2.circle(orig, (point[1], point[0]), 5, (0, 255, 0), -1)
            
            cv2.imshow("Edge Map", orig)
        
            while True:    
                key = cv2.waitKey(1)
                if key == 27:
                    cv2.destroyAllWindows()
                    break

            

            orig = image.copy()
            for i in range(0, len(indexes)):
                cv2.circle(orig, (indexes[i], (accumEdged.shape[0]-10)), 5, (0, 255, 0), -1)

            cv2.imshow("Edge Map", orig)
            
            while True:    
                key = cv2.waitKey(1)
                if key == 27:
                    cv2.destroyAllWindows()
                    break
        
        else:
            print("p" + str(i) + "_left_act_" + str(j) + " ===  " + str(len(indexes)))


        












##i = 1
## loop over the contours
#for c in cnts:
#    orig = image.copy()
#	# draw each contour on the output image with a 3px thick purple
#	# outline, then display the output contours one at a time
#    cv2.drawContours(orig, [c], -1, (240, 0, 159), 3)
#    #cv2.imwrite("Countours" + str(i) + ".jpg",  orig)
#    #i = i + 1
#    
#    cv2.imshow("Contours", orig)
#    while True:
#        key = cv2.waitKey(1)
#        if key == 27:
#            cv2.destroyAllWindows()
#            break
#
#
#
#
#
#cv2.arcLength(cnts_nedded[0])
#
#
#
#
#cv2.imshow("Contours", orig)
#while True:
#    key = cv2.waitKey(1)
#    if key == 27:
#        cv2.destroyAllWindows()
#        break
#    
#
#
#hull = []
#hull.append(cv2.convexHull(cnts[0]))
#orig = image.copy()
#cv2.drawContours(orig, hull[0], 0, (255,0,0), 1, 8)
#
#
#
#M = cv2.moments(hull[0])
#cX = int(M["m10"] / M["m00"])
#cY = int(M["m01"] / M["m00"])
#orig = image.copy()
#cv2.circle(orig, (cX,cY), 7, (0,255,0), -1)
#
#
#    
#
#a = cnts[0]
#x = np.array(a[:, 0:1, 0].T)
#y = np.array(a[:, 0:1, 1].T)
#point = a.shape[0]
#for i in range(0, point-1):
#    p = a[i, 0:1, 0:2]
#    x = p[0, 0]
#    y = p[0, 1]
#
#
#grad = np.gradient(y, x)
#
#f = np.arr([1, 2, 3, 4])













































