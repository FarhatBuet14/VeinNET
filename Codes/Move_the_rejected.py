# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:22:44 2019

@author: User
"""

import numpy as np
import cv2
import os
import pandas as pd 
import shutil


folder_path = "./HandVeinDatabase/right - 300/norm - 300/"
       
rejected_folder = "./Rejected/"
csv_file_name = "Vein_Image/Extarcted.csv"

def delete_the_rejected_images(csv_file_name, deleted_folder):
    
    file = pd.read_csv(csv_file_name) 
    file = np.array(file)
    image_count = 0
    for file_name in file[:, 0]:
        if(file[image_count, 1] == 2):
            shutil.move(folder_path + file_name, deleted_folder + file_name) 
            print(file_name + ' rejected, so removed - ' + str(file[image_count, 1]))
        
        image_count += 1

    return image_count

image_count = delete_the_rejected_images(csv_file_name, rejected_folder)

########################### Rename Filenames ##########################

#filenames = os.listdir(rejected_folder)
#
#for image_filename in filenames:
#    img_file = cv2.imread(rejected_folder + '/' + image_filename)
#    image_name = image_filename.replace('deleted_', '')
#    cv2.imwrite('./bla/' + image_name, img_file)






