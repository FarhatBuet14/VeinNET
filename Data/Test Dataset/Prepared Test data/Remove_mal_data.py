############################  Import Libraries  ###############################
###############################################################################

import cv2
import numpy as np
import os

# --- Cascaded all dataset
dataset = "2"
data_folder = "./Final/"

# ---- Load past data
prepapred_data = np.load(data_folder + 'prepared_data_' + dataset + '.npz')
images = list(prepapred_data['images'])
points = list(prepapred_data['manual_points'])
names = list(prepapred_data['names'])


image = []
point = []
name = []


for index in range(0, len(names)):
    if(len(points[index]) == 4):
        image.append(images[index])
        point.append(np.array(points[index]))
        name.append(names[index])

point = np.array(point)

np.savez(data_folder + 'final_test_data_' + dataset + '.npz',
         images = image,
         manual_points = point,
         names = name)













