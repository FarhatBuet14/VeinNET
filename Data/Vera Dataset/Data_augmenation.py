############################  Import Libraries  ###############################
###############################################################################

import numpy as np
import imgaug.augmenters as iaa
import cv2
import os

##############################  Import Data  ##################################
###############################################################################

# Input Data Folders
bef = "./Data/Vera Dataset/"
Aug_train_data_folder = bef + 'Train_with_Augmentation/'
Aug_data_with_points_folder = bef + 'Aug_data_with_points/'
data_folder = bef + 'Train_without_Augmentation/'

data = np.load(data_folder + 'Train.npz')
X_train_names = data['names'].astype(str)
y_train = data['manual_points']

X_train_bmp = []
for name in X_train_names:
    X_train_bmp.append(cv2.imread(data_folder + name))

X_train_bmp = np.array(X_train_bmp)

########################################################

def draw_troughs(img, points):
    
    points = points.reshape((2, 2))
    
    for point in points:   
        point = np.array(point).astype(int)
        cv2.circle(img, (point[0], 
                   point[1]), 
                   5, (255, 255, 0), -1)

    return img

###############################  Main Code  ###################################
###############################################################################

keypoints = y_train.reshape((y_train.shape[0], 2, 2))
# Array to list for giving input to the iaa.Affine function
points = []
for point in keypoints:
    points.append(point)
keypoints = points

X_train_aug_names = []
X_train_aug_bmp = []
y_train_aug = []

total_sample = len(X_train_bmp)
num_aug_type = 8
sam_per_aug = int(len(X_train_bmp) / num_aug_type)

portion_count = 0
img_count = 0
for rotate_angle in [-80, -60, -50, -30, 80, 60, 50, 30]:
    # Take portion of samples for augmentation
    images_bmp = X_train_bmp[(portion_count * sam_per_aug) : ((portion_count+1) * sam_per_aug)]
    points = keypoints[(portion_count * sam_per_aug) : ((portion_count+1) * sam_per_aug)]
    
    # Take samples of that portion for augmentation
    for sample in range(0, len(images_bmp)):
        image = images_bmp[sample]
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        point = []
        point.append(points[sample])
        
        # Augmentation_1 (Rotation)
        X_aug , y_aug = iaa.Affine(rotate = rotate_angle)(images = image, 
                                                   keypoints = point)
        # Again Reshaping to before format
        X_aug = X_aug.reshape((image.shape[1], image.shape[2], image.shape[3]))
        y_aug = np.array(y_aug).reshape((2, 2))
        # Append samples after augmentation
        X_train_aug_bmp.append(X_aug)
        y_train_aug.append(y_aug)
    
        # Save the augmented pictures to the exact folder
        name = X_train_names[img_count].replace(".png", "")
        name = name + "_" + str(img_count) + "_rotVera_" + str(rotate_angle) + ".png"
        X_train_aug_names.append(name)
        cv2.imwrite(Aug_train_data_folder + name, X_aug)
        
        # Draw Troughs and Save Image
        X_aug = draw_troughs(X_aug, y_aug)
        cv2.imwrite(Aug_data_with_points_folder + name , X_aug)
        
        # Augmentation_2 (Fliplr + Rotation)
        images_lr , keypoints_lr = iaa.Fliplr(1.0)(images = image, 
                                             keypoints = point)
        X_aug , y_aug = iaa.Affine(rotate = rotate_angle)(images = images_lr, 
                                                   keypoints = keypoints_lr)
        # Again Reshaping to before format
        X_aug = X_aug.reshape((image.shape[1], image.shape[2], image.shape[3]))
        y_aug = np.array(y_aug).reshape((2, 2))
        # temp = y_aug[0, 0]
        # y_aug[0, 0] = y_aug[0, 1]
        # y_aug[0, 1] = temp
        # Append samples after augmentation
        X_train_aug_bmp.append(X_aug)
        y_train_aug.append(y_aug)
    
        # Save the augmented pictures to the exact folder
        name = X_train_names[img_count].replace(".png", "")
        name = name + "_" + str(img_count) + "_flrotVera_" + str(rotate_angle) + ".png"
        X_train_aug_names.append(name)
        cv2.imwrite(Aug_train_data_folder + name , X_aug)
        
        # Draw Troughs and Save Image
        X_aug = draw_troughs(X_aug, y_aug)
        cv2.imwrite(Aug_data_with_points_folder + name , X_aug)
        
        img_count += 1
    
    portion_count += 1

y_train_aug = np.array(y_train_aug)

# Save the Augmented Data to a .npz file
np.savez(Aug_train_data_folder + "Augmented_Train_data_vera.npz",
         X_train_aug_names = X_train_aug_names,
         y_train_aug = y_train_aug)
