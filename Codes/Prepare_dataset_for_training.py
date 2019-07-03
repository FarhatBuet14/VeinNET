############################  Import Libraries  ###############################
###############################################################################

import numpy as np
import imutils
import cv2
from operator import itemgetter
import math
import os
import shutil

#########################  Find Accumulated Image  ############################
###############################################################################

def get_accumEdged(image):
    accumEdged = np.zeros(image.shape[:2], dtype="uint8")

    for chan in cv2.split(image):
        chan = cv2.medianBlur(chan, 3)
        edged = cv2.Canny(chan, 50, 150)
        accumEdged = cv2.bitwise_or(accumEdged, edged)
        
    return accumEdged

####################  Find Conrtoue from Accumulated Image  ###################
###############################################################################
 
def find_contour_needed(accumEdged, cnt_length_thresh = 100):
    
    contour_image = accumEdged
    
    cnts = cv2.findContours(contour_image, cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # Delete the short contours less than length_threshold
    cnts_nedded = []
    Length = []
    for c in cnts:
        if( cv2.arcLength(c, False) > cnt_length_thresh ):
            Length.append(cv2.arcLength(c, False))
            cnts_nedded.append(c)

    return cnts_nedded, Length


#########################  Concatenate all contours  ##########################
###############################################################################

def cnt_concat(cnts_nedded):

    all_cnts = np.zeros((1, 1, 2))
    
    for cnt in cnts_nedded:
        all_cnts = np.append(all_cnts, cnt, axis = 0)
    
    all_cnts = all_cnts[1:, :, :]
    all_cnts = np.reshape(all_cnts, (all_cnts.shape[0], 2) )
    
    return all_cnts


###############  Apply Algorithm for finding valley points  ###################
###############################################################################

def get_trough_points(image, all_cnts): # 
    
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
                    troughs_final.append(((troughs[i, :] + troughs[i+1, :]) / 2).astype(int))
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


###################  Get exact two points from troughs  #######################
###############################################################################

def get_two_points(image, image_name, points, height = 90, width = 50, th = 20):
    
    if(len(points) != 3): # Error if the troughs can not be calculated through algorithm
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


#############################  Draw Troughs  ##################################
###############################################################################

def draw_troughs(img, points):
    
    points = points.reshape((2, 2))
    
    for point in points:   
        point = np.array(point).astype(int)
        cv2.circle(img, (point[0], 
                   point[1]), 
                   5, (0, 0, 0), -1)

    return img


###########################  Find Valley Points  ##############################
###############################################################################

def data_2_points(names, data_folder, cnt_length_thresh = 20):
    
    error_files = [] # can not be calculated through algorithm
    algo_extracted_files = [] #  calculated through algorithm
    final_points = []
    
    for image_name in names:
        
        image = cv2.imread(data_folder + image_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        accumEdged = get_accumEdged(gray)
        
        [cnts_nedded, length_cnt] = find_contour_needed(accumEdged, cnt_length_thresh)
        all_cnts = cnt_concat(cnts_nedded)       
        _, troughs = np.array(get_trough_points(gray.copy(), all_cnts))
        
        _, points, err = get_two_points(gray.copy(), image_name, troughs, 
                                                         height = 90, width = 50, th = 20)
        
        if(err != None): #if the troughs can not be calculated through algorithm
            error_files.append(err)
            
        else: 
            algo_extracted_files.append(image_name)
            final_points.append(points)

                
    error_files = np.array(error_files)
    algo_extracted_files = np.array(algo_extracted_files)
    final_points = np.array(final_points)

    return algo_extracted_files, error_files, final_points


###############################  Main Code  ###################################
###############################################################################

data_folder = "./Data/Dataset/"
extraction_folder = "./Extracted/"
train_data_folder = "./Data/Train_Data/"
test_data_folder = "./Data/Test_Data/"

data_algo = np.load(extraction_folder + 'algo_result.npz')
data_manual = np.load(extraction_folder + 'manual_selsction_data.npz')

manual_select_names = data_manual['names'].astype(str)
algo_select_names = data_algo['pos_vein'].astype(str)

manual_select_points = data_manual['manual_points']
_, _, algo_select_points = data_2_points(algo_select_names, data_folder, cnt_length_thresh = 20)

#######################  Distribute Test Train Data  ##########################
###############################################################################

# Prepare Train Dataset
X_train_bmp = []
X_train = []

for file in algo_select_names:
    shutil.copy(data_folder + file, train_data_folder + file)
    img = cv2.imread(train_data_folder + file)
    accu = get_accumEdged(img)
    X_train_bmp.append(img)
    X_train.append(accu)

X_train_bmp = np.array(X_train_bmp)
X_train = np.array(X_train)
y_train = algo_select_points.reshape((len(algo_select_points), 4))

# Prepare Test Dataset
X_test_bmp = []
X_test = []

for file in manual_select_names:
    shutil.copy(data_folder + file, test_data_folder + file)
    img = cv2.imread(test_data_folder + file)
    accu = get_accumEdged(img)
    X_test_bmp.append(img)
    X_test.append(accu)

X_test_bmp = np.array(X_test_bmp)
X_test = np.array(X_test)
y_test = manual_select_points.reshape((len(manual_select_points), 4))


# Save all the data to a .npz file
np.savez(data_folder + 'train_test_data.npz', 
         X_train_bmp = X_train_bmp,
         X_train = X_train,
         y_train = y_train,
         X_test_bmp = X_test_bmp,
         X_test = X_test,
         y_test = y_test)
