############################  Import Libraries  ###############################
###############################################################################

import numpy as np
import os
import cv2
import math
import imgaug.augmenters as iaa

##############################  Import Data  ##################################
###############################################################################

from keras import backend as K
K.set_image_data_format('channels_last')

# Input Data Folders
dataset_folder = "./Data/Dataset/"
data_folder = "./Data/"
extraction_folder = "./Extracted/"
train_data_folder = "./Data/Train_Data/"
test_data_folder = "./Data/Test_Data/"
Aug_train_data_folder = './Data/Augmented_Train_data/'

# Output Data Folders
train_Output_data = "./Model_Output/"
weightFile = train_Output_data + 'WeightFile_best.hdf5'
saved_model_File = train_Output_data + 'Saved_Model.h5'
prediction_fldr = train_Output_data + 'Prediction/'
cropped_fldr = train_Output_data + 'Cropped/'

# Import Main Data
data = np.load(data_folder + 'train_test_data_without_augmetation.npz') 
X_test_names = data['X_test_names'].astype(str)
X_test_bmp = data['X_test_bmp']
X_test_gray = data['X_test_gray']
X_test = data['X_test']
y_test = data['y_test']

#########################  Draw Troughs on Images  ############################
###############################################################################

def draw_troughs_with_bounding_boxes(image, points_pred, points_test, name,
                                     bounding_box_folder, cropped_fldr,
                                     height = 90, width = 70, th = 10):    
    top_left = points_pred[0]
    top_right = points_pred[1]
    
    # Find the angle to rotate the image
    angle  = (180/np.pi) * (np.arctan((top_left[1] - top_right[1])/
                            (top_left[0] - top_right[0])))
    
    # Rotate the image to cut rectangle from the images
    points_pred = points_pred.reshape((1, 2, 2))
    points_test = points_test.reshape((1, 2, 2))
    image = image.reshape((1, 240, 300, 3))
    image_rotated , keypoints_pred_rotated = iaa.Affine(rotate=-angle)(images=image, 
                                  keypoints=points_pred)
    _ , keypoints_test_rotated = iaa.Affine(rotate=-angle)(images=image, 
                                  keypoints=points_test)
    
    image_rotated = image_rotated.reshape((240, 300, 3))
    keypoints_pred_rotated = keypoints_pred_rotated.reshape((2, 2))
    keypoints_test_rotated = keypoints_test_rotated.reshape((2, 2))
    
    # Rotated Points
    top_left_ = keypoints_pred_rotated[0]    
    top_left_ = tuple(top_left_.reshape(1, -1)[0])
    
    center = np.zeros((2, )).astype(int)
    center[0] = top_left_[0] + int(width/2)  - th
    center[1] = top_left_[1] + int(height/2)
    center = tuple(center.reshape(1, -1)[0])
    
    # Crop the Vein Image
    crop = cv2.getRectSubPix(image_rotated, (width, height), center)
    cv2.imwrite(cropped_fldr + name, crop)
    
    # Draw Predicted Troughs
    points = keypoints_pred_rotated.reshape((2, 2))    
    for point in points:   
        point = np.array(point).astype(int)
        cv2.circle(image_rotated, (point[0], point[1]), 
                   5, (0, 0, 0), -1)
    
    # Draw Actual Troughs
    points = keypoints_test_rotated.reshape((2, 2))    
    for point in points:   
        point = np.array(point).astype(int)
        cv2.circle(image_rotated, (point[0], point[1]), 
                   5, (255, 0, 0), -1)
    
    # Points for Bounding Boxes
    tl = np.zeros((2, )).astype(int)
    tl[0] = center[0] - int(width/2)  
    tl[1] = center[1] - int(height/2)
    tl = tuple(tl.reshape(1, -1)[0])
    
    br = np.zeros((2, )).astype(int)
    br[0] = center[0] + int(width/2)  
    br[1] = center[1] + int(height/2)
    br = tuple(br.reshape(1, -1)[0])
    
    # Draw Bounding Boxes and Save the image
    image_rotated = cv2.rectangle(image_rotated, tl, br , (0,0,0), 2)
    cv2.imwrite(bounding_box_folder + name, image_rotated)

    return crop

#####################  Calculate loss from two points  ########################
###############################################################################

def cal_loss_from_points(X_test_bmp, y_test, y_pred, X_test_names,
                       data_folder, cropped_fldr, prediction_fldr,
                       height = 90, width = 70, th = 10,
                       thresh_h = 200, thresh_l = 70):
    vein_loss = []
    vein_images = []
    count = 0
    for image in X_test_bmp:         
        vein_image = draw_troughs_with_bounding_boxes(image = image, points_pred = y_pred[count], 
                                                      points_test = y_test[count], 
                                                      name = X_test_names[count],
                                                      bounding_box_folder = prediction_fldr,
                                                      cropped_fldr = cropped_fldr,
                                                      height = height, width = width, th = th)
        # Calculate loss from extracted Vein Image
        gray = cv2.cvtColor(vein_image, cv2.COLOR_BGR2GRAY)
        accu = ((gray <= thresh_h)  & (gray >= thresh_l))
        true = np.count_nonzero(accu)
        false = (accu.shape[0] * accu.shape[1]) - true
        vein_loss.append(false / (false + true))
        vein_images.append(vein_image)
        
        count += 1
    
    vein_loss = np.array(vein_loss)
    vein_images = np.array(vein_images)
    
    return vein_images, vein_loss

########################## Import libraries for Model  ########################
###############################################################################

from keras.models import load_model

import keras.backend as K
K.set_image_data_format('channels_last')

########################## Load Saved Model & Predict #########################
###############################################################################

model = load_model(saved_model_File)
model.load_weights(weightFile)

# summarize model.
model.summary()

# evaluate the model
X_test = X_test / 255
X_test = X_test.reshape((X_test.shape[0], 240, 300, 1))
score = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test)
y_pred = y_pred.reshape((y_pred.shape[0], 2, 2))

# Calculate and Save Predicted Data
vein_images, vein_loss = cal_loss_from_points(X_test_bmp = X_test_bmp, y_test = y_test,
                                              y_pred = y_pred, X_test_names = X_test_names,
                                              data_folder = data_folder, 
                                              cropped_fldr = cropped_fldr,
                                              prediction_fldr = prediction_fldr,
                                              height = 90, width = 70, th = 10,
                                              thresh_h = 200, thresh_l = 70)

np.savez(train_Output_data + "pred_data.npz",
         y_pred = y_pred,
         score = score,
         vein_images = vein_images,
         vein_loss = vein_loss)
