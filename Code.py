# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 19:33:05 2019

@author: User
"""

import numpy as np
import imutils
import cv2



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

        if(len(indexes) > 0):
            point = [(accumEdged.shape[0]-10), int(sum(indexes)/len(indexes))]

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


  


