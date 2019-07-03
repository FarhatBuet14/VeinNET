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


data_folder = "./Data/All/"
extraction_folder = "./Extracted/"
troughs_folder = "./Extracted/Troughs/"
vein_folder = "./Extracted/Vein_Images/"
bounding_box_folder = "./Extracted/Bounding_box/"
pos_vein_folder = "./Extracted/Vein_Images/pos_vein_img/"
neg_vein_folder = "./Extracted/Vein_Images/neg_vein_img"



############################  Second Part  ###############################
###############################################################################


data = np.load(extraction_folder + 'manual_selsction_data.npz')

X = data['X']
Y = data['Y']
train_names = data['train_names'].astype(str)


img = X[200]
cv2.imshow("Frame", img)
       
while True:    
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
        break











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
















