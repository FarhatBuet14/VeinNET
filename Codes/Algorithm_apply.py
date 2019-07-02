############################  Import Libraries  ###############################
###############################################################################

import numpy as np
import imutils
import cv2
from operator import itemgetter
import math
import os
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
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

def data_2_points(data_folder, troughs_folder, cnt_length_thresh = 20):
    filenames = os.listdir(data_folder)
    
    error_files = [] # can not be calculated through algorithm
    algo_extracted_files = [] #  calculated through algorithm
    final_points = []
    
    
    for file in filenames:
        file_type = file.split(".")[1] # Avoid Placeholder files
        if(file_type == "bmp"): 
            image_name = file
            image = cv2.imread(data_folder + image_name)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            accumEdged = get_accumEdged(gray)
            
            [cnts_nedded, length_cnt] = find_contour_needed(accumEdged, cnt_length_thresh)
            all_cnts = cnt_concat(cnts_nedded)
            cnt_image = cv2.drawContours(gray.copy(), 
                                         np.array(all_cnts).reshape((-1,1,2)).astype(np.int32),
                                         -1, (0,255,0), 3)
            
            trough_image , troughs = np.array(get_trough_points(gray.copy(), all_cnts))
            
            final_trough_image, points, err = get_two_points(gray.copy(), image_name, troughs, 
                                                             height = 90, width = 50, th = 20)
            
            if(err != None): #if the troughs can not be calculated through algorithm
                error_files.append(err)
                
            else:
                
                algo_extracted_files.append(image_name)
                final_points.append(points)
                
                # Concatenate Accumilated + Cnt + Trough Images 
                all_img = cv2.hconcat((accumEdged, cnt_image))
                all_img = cv2.hconcat((all_img, trough_image))
                all_img = cv2.hconcat((all_img, final_trough_image))
                
                cv2.imwrite(troughs_folder + image_name, all_img)
                
    error_files = np.array(error_files)
    algo_extracted_files = np.array(algo_extracted_files)
    final_points = np.array(final_points)

    return algo_extracted_files, error_files, final_points


#####################  Calculate loss from two points  ########################
###############################################################################

def cal_loss_from_points(point_extracted_images, points,
                       data_folder, vein_folder, bounding_box_folder,
                       height = 90, width = 70, th = 10,
                       thresh_h = 200, thresh_l = 70):
    vein_loss = []
    count = 0
    for file in point_extracted_images:
        image = cv2.imread(data_folder + file)
        point = points[count]
        top_left = point[0]
        top_right = point[1]
        
        # Find the angle to rotate the image
        angle  = (180/np.pi) * (np.arctan((top_left[1] - top_right[1])/
                                (top_left[0] - top_right[0])))
        
        point = point.reshape((1, 2, 2))
        image = image.reshape((1, 240, 300, 3))
        image_rotated , keypoints_rotated = iaa.Affine(rotate=-angle)(images=image, 
                                      keypoints=point)
        
        image_rotated = image_rotated.reshape((240, 300, 3))
        keypoints_rotated = keypoints_rotated.reshape((2, 2))
        
        # Rotated Points
        top_left_ = keypoints_rotated[0]    
        top_left_ = tuple(top_left_.reshape(1, -1)[0])
        
        center = np.zeros((2, )).astype(int)
        center[0] = top_left_[0] + int(width/2)  - th
        center[1] = top_left_[1] + int(height/2)
        center = tuple(center.reshape(1, -1)[0])
        
        # Crop the Vein Image
        crop = cv2.getRectSubPix(image_rotated, (width, height), center)
        cv2.imwrite(vein_folder + file, crop)
        
        tl = np.zeros((2, )).astype(int)
        tl[0] = center[0] - int(width/2)  
        tl[1] = center[1] - int(height/2)
        tl = tuple(tl.reshape(1, -1)[0])
        
        br = np.zeros((2, )).astype(int)
        br[0] = center[0] + int(width/2)  
        br[1] = center[1] + int(height/2)
        br = tuple(br.reshape(1, -1)[0])
        
        # Draw Bounding Boxes
        image_rotated = draw_troughs(image_rotated, keypoints_rotated)
        image_rotated = cv2.rectangle(image_rotated, tl, br , (0,0,0), 2)
        cv2.imwrite(bounding_box_folder + file, image_rotated)
        
        # Calculate loss from extracted Vein Image
        vein_image = crop
        gray = cv2.cvtColor(vein_image, cv2.COLOR_BGR2GRAY)
        accu = ((gray <= thresh_h)  & (gray >= thresh_l))
        true = np.count_nonzero(accu)
        false = (accu.shape[0] * accu.shape[1]) - true
        vein_loss.append(false / (false + true))
        
        count += 1
    
    vein_loss = np.array(vein_loss)
    return vein_loss



#####################  Distribute Pos & Neg Vein Images #######################
###############################################################################

def distr_pos_neg_vein(vein_folder, vein_names, vein_loss,
                       pos_vein_folder, neg_vein_folder,
                       loss_thresh = 0.01):
    pos_vein = vein_names[vein_loss <= loss_thresh]
    neg_vein = vein_names[vein_loss > loss_thresh]
    
    # Move the pos and neg images
    for image in pos_vein: shutil.copy(vein_folder + image, pos_vein_folder)
    for image in neg_vein: shutil.copy(vein_folder + image, neg_vein_folder)

    return pos_vein, neg_vein




###############################################################################
###############################################################################
###############################################################################
################################ Main Code ####################################

data_folder = "./Data/All/"
extraction_folder = "./Extracted/"
troughs_folder = "./Extracted/Troughs/"
vein_folder = "./Extracted/Vein_Images/"
bounding_box_folder = "./Extracted/Bounding_box/"
pos_vein_folder = "./Extracted/Vein_Images/pos_vein_img/"
neg_vein_folder = "./Extracted/Vein_Images/neg_vein_img"



algo_extracted_files, error_files, final_points = data_2_points(data_folder = data_folder,
                                                                troughs_folder = troughs_folder,
                                                                cnt_length_thresh = 20)


vein_loss = cal_loss_from_points(point_extracted_images = algo_extracted_files, 
                                 points = final_points,
                                 data_folder = data_folder, 
                                 vein_folder = vein_folder, 
                                 bounding_box_folder = bounding_box_folder,
                                 height = 90, width = 70, th = 10,
                                 thresh_h = 200, thresh_l = 70)

pos_vein, neg_vein = distr_pos_neg_vein(vein_folder = vein_folder, 
                               vein_names = algo_extracted_files,
                               vein_loss = vein_loss,
                               pos_vein_folder = pos_vein_folder,
                               neg_vein_folder = neg_vein_folder,
                               loss_thresh = 0.01)

# Save all the data to a .npz file
np.savez(extraction_folder + 'result.npz', 
         algo_extracted_files = algo_extracted_files,
         error_files = error_files,
         points = final_points,
         vein_loss = vein_loss,
         pos_vein = pos_vein,
         neg_vein = neg_vein)
















