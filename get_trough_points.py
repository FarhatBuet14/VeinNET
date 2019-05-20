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
        edged = cv2.Canny(chan, 150, 200)
        accumEdged = cv2.bitwise_or(accumEdged, edged)
            
    cv2.imshow("Edge Map", accumEdged)
    
    while True:    
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        
    return accumEdged




def find_contour_needed(accumEdged, area_threshold = 100):
    
    contour_image = accumEdged
    
    cnts = cv2.findContours(contour_image, cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    cnts_nedded = []
    area = []
    for c in cnts:
        if(cv2.contourArea(c) > area_threshold):
            area.append(cv2.contourArea(c))
            cnts_nedded.append(c)

    return cnts_nedded, area



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
       
    #plt.figure(1, figsize=(12, 6))
    #plt.subplot(211)
    #plt.plot(cnt_y)
    #plt.ylabel('Controur_points')
    
    grad = np.gradient(cnt_y)
    
    #plt.figure(1, figsize=(12, 6))
    #plt.subplot(212)
    #plt.plot(grad)
    #plt.ylabel('Gradients')
    #plt.show()
    
    troughs = []   
    for i in range(0, len(grad)):
        if((grad[i] < 0)  &  (grad[i-1] > 0) & (i>0)):
            troughs.append([cnt_x[i], cnt_y[i]])
    
    troughs = np.array(troughs)


    troughs = np.array(sorted(troughs, key=itemgetter(0)))
    
    
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
    
    
    
    for point in troughs:       
        cv2.circle(image, (int(point[0]), 
                           int(point[1])), 
                           5, (0, 255, 0), -1)    
    
    cv2.imshow("Trough point", image)
        
    while True:    
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break


    return troughs




############## Main Code ############

folder_path = "./HandVeinDatabase/left - 1200/act - 300/"

image_name = "p1_left_act_3.bmp"
image = cv2.imread(folder_path + image_name, )


accumEdged = get_accumEdged(image)
area_threshold = 40
[cnts_nedded, area_cnt] = find_contour_needed(accumEdged, area_threshold)
all_cnts = cnt_concat(cnts_nedded)
troughs = np.array(get_trough_points(image.copy(), all_cnts))









