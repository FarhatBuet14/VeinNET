import numpy as np
import imutils
import cv2
from operator import itemgetter
import math
import os



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

    # Delete the short contours less than length_threshold
    cnts_nedded = []
    Length = []
    for c in cnts:
        if( cv2.arcLength(c, False) > length_threshold ):
            Length.append(cv2.arcLength(c, False))
            cnts_nedded.append(c)

    return cnts_nedded, Length




def cnt_concat(cnts_nedded):   # Concatenate all contours

    all_cnts = np.zeros((1, 1, 2))
    
    for cnt in cnts_nedded:
        all_cnts = np.append(all_cnts, cnt, axis = 0)
    
    all_cnts = all_cnts[1:, :, :]
    all_cnts = np.reshape(all_cnts, (all_cnts.shape[0], 2) )
    
    return all_cnts




def get_trough_points(image, all_cnts): # Algorithm apply for finding valley points
    
    cnt_x = all_cnts[:, 0]
    cnt_y = all_cnts[:, 1]
  
    grad = np.gradient(cnt_y)
    troughs = []   
    for i in range(2, len(grad)-3):
        if((grad[i] < 0)  &  (grad[i-1] > 0) & (i>0)):
            
            troughs.append([cnt_x[i], cnt_y[i]])
    
    troughs = np.array(troughs)


    troughs = np.array(sorted(troughs, key=itemgetter(0)))
    
    
    # make the closer points to one point 
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
                    troughs_final.append( ((troughs[i, :] + troughs[i+1, :]) / 2).astype(int) )
                    merge = True
                else:
                    if(merge == False):           
                        troughs_final.append(troughs[i, :])
                    else:
                        merge = False
            
            troughs = np.array(troughs_final)
    
    # remove other points except troughs
    
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

    # Draw Trough points on Image
    for point in troughs:   
        point = point.astype(int)
        cv2.circle(image, (point[0], 
                   point[1]), 
                   5, (0, 255, 0), -1)
    
    return image, troughs

# Get those exact two points from the troughs
def get_two_points(image, image_name, points, height = 90, width = 50, th = 20):
    
    if(len(points) < 3): # Error if the troughs can not be calculated through algorithm
        err = image_name
    
    else:
        
        points = points.astype(int)
        dists = np.zeros((3, 1))
        dists[0] = math.hypot(points[0, 0] - points[1, 0],        ## (0-1)
                          points[0, 1] - points[1, 1])   
        dists[1] = math.hypot(points[0, 0] - points[2, 0],        ## (0-2) 
                          points[0, 1] - points[2, 1]) 
        dists[2] = math.hypot(points[2, 0] - points[1, 0],        ## (1-2) 
                          points[2, 1] - points[1, 1]) 
        
        max_dist_arg = np.argmax(dists)
        
        points = points.tolist()
        if(max_dist_arg == 0): del points[2]
        if(max_dist_arg == 1): del points[1]
        if(max_dist_arg == 2): del points[0]
        points = np.array(points)
        
        for point in points:   
            point = point.astype(int)
            cv2.circle(image, (point[0], 
                       point[1]), 
                       5, (0, 255, 0), -1)
        err = None
    
    
    return image, points, err



############## Main Code ############

folder_path = "./Data/All/"
save_folder_path = "./Extracted/Troughs/"

filenames = os.listdir(folder_path)
error_files = []
count = 0
for file in filenames:
    file_type = file.split(".")[1]
    if(file_type == "bmp"): 
        count += 1
        image_name = file
        image = cv2.imread(folder_path + image_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        accumEdged = get_accumEdged(gray)
        
        length_threshold = 20
        [cnts_nedded, length_cnt] = find_contour_needed(accumEdged, length_threshold)
        all_cnts = cnt_concat(cnts_nedded)
        cnt_image = cv2.drawContours(gray.copy(), 
                                     np.array(all_cnts).reshape((-1,1,2)).astype(np.int32),
                                     -1, (0,255,0), 3)
        
        trough_image , troughs = np.array(get_trough_points(gray.copy(), all_cnts))
        
        final_trough_image, points, err = get_two_points(gray.copy(), image_name, troughs, 
                                                         height = 90, width = 50, th = 20)
        if(err != None):
            error_files.append(err)
            
        else:
            all_img = cv2.hconcat((accumEdged, cnt_image))
            all_img = cv2.hconcat((all_img, trough_image))
            all_img = cv2.hconcat((all_img, final_trough_image))
            
            cv2.imwrite(save_folder_path + image_name, all_img)
        

#cv2.imshow('bla', image)     
#
#while True:    
#    key = cv2.waitKey(1)
#    if key == 27:
#        cv2.destroyAllWindows()
#        break

#error_files = np.array(error_files)
#
#np.savez(save_folder_path + 'Error.npz', error_files = error_files)


error_data = np.load(save_folder_path + 'Error.npz') 

err_files = error_data['error_files'].astype(str)

image = cv2.imread(folder_path + err_files[0])










































