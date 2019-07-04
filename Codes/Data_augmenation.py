############################  Import Libraries  ###############################
###############################################################################

import numpy as np
import imgaug.augmenters as iaa
import cv2

##############################  Import Data  ##################################
###############################################################################

from keras import backend as K
K.set_image_data_format('channels_last')

# Input Data Folders
data_folder = "./Data/"
Aug_train_data_folder = './Data/Augmented_Train_data/'
Aug_data_with_points_folder = './Data/Augmented_Train_data/Augmented_train_data_with_points/'

data = np.load(data_folder + 'train_test_data_without_augmetation.npz') 
X_train_bmp = data['X_train_bmp']
X_train_names = data['X_train_names'].astype(str)
y_train = data['y_train']

#########################  Find Accumulated Image  ############################
###############################################################################

def get_accumEdged(image):
    accumEdged = np.zeros(image.shape[:2], dtype="uint8")

    for chan in cv2.split(image):
        chan = cv2.medianBlur(chan, 3)
        edged = cv2.Canny(chan, 50, 150)
        accumEdged = cv2.bitwise_or(accumEdged, edged)
        
    return accumEdged

##########################  Shuffle Data Funtion  #############################
###############################################################################

def shuffleData(X,y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X_shuffled = X[randomize]
    y_shuffled = y[randomize]
    return X_shuffled,y_shuffled

#########################  Draw Troughs on Images  ############################
###############################################################################

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
X_train_aug_gray = []
X_train_aug_accu = []
y_train_aug = []

total_sample = len(X_train_bmp)
num_aug_type = 8
sam_per_aug = int(len(X_train_bmp) / num_aug_type)

portion_count = 0
img_count = 0
for rotate_angle in [-80, -60, -50, -30, -80, -60, -50, -30]:
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
        X_train_aug_gray.append(cv2.cvtColor(X_aug, cv2.COLOR_BGR2GRAY))
        X_train_aug_accu.append(get_accumEdged(X_aug))
        y_train_aug.append(y_aug)
    
        # Save the augmented pictures to the exact folder
        name = X_train_names[img_count].replace(".bmp", "")
        X_train_aug_names.append(name)
        cv2.imwrite(Aug_train_data_folder + name + "_" + str(img_count) +
                    "_rot_" + str(rotate_angle) + ".bmp", X_aug)
        
        # Draw Troughs and Save Image
        X_aug = draw_troughs(X_aug, y_aug)
        cv2.imwrite(Aug_data_with_points_folder + name + "_" + str(img_count) +
                    "_rot_" + str(rotate_angle) + ".bmp", X_aug)
        
        # Augmentation_2 (Fliplr + Rotation)
        images_lr , keypoints_lr = iaa.Fliplr(1.0)(images = image, 
                                             keypoints = point)
        X_aug , y_aug = iaa.Affine(rotate = rotate_angle)(images = image, 
                                                   keypoints = point)
        # Again Reshaping to before format
        X_aug = X_aug.reshape((image.shape[1], image.shape[2], image.shape[3]))
        y_aug = np.array(y_aug).reshape((2, 2))
        # Append samples after augmentation
        X_train_aug_bmp.append(X_aug)
        X_train_aug_gray.append(cv2.cvtColor(X_aug, cv2.COLOR_BGR2GRAY))
        X_train_aug_accu.append(get_accumEdged(X_aug))
        y_train_aug.append(y_aug)
    
        # Save the augmented pictures to the exact folder
        name = X_train_names[img_count].replace(".bmp", "")
        X_train_aug_names.append(name)
        cv2.imwrite(Aug_train_data_folder + name + "_" + str(img_count) +
                    "_flrot_" + str(rotate_angle) + ".bmp", X_aug)
        
        # Draw Troughs and Save Image
        X_aug = draw_troughs(X_aug, y_aug)
        cv2.imwrite(Aug_data_with_points_folder + name + "_" + str(img_count) +
                    "_flrot_" + str(rotate_angle) + ".bmp", X_aug)
        
        img_count += 1
    
    
    portion_count += 1

X_train_aug_bmp = np.array(X_train_aug_bmp)
X_train_aug_gray = np.array(X_train_aug_gray)
X_train_aug_accu = np.array(X_train_aug_accu)
y_train_aug = np.array(y_train_aug)

# Save the Augmented Data to a .npz file
np.savez(data_folder + "Augmented_Train_data.npz",
         X_train_aug_names = X_train_aug_names,
         X_train_aug_bmp = X_train_aug_bmp,
         X_train_aug_gray = X_train_aug_gray,
         X_train_aug_accu = X_train_aug_accu,
         y_train_aug = y_train_aug)
