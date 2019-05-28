# -*- coding: utf-8 -*-
"""
Created on Thu May 23 18:12:09 2019

@author: User
"""

import numpy as np
import pandas as pd 
import shutil


csv_file_name = './Troughs_Model/model_AccuEdges/prediction.csv'


#def measure_pred_acc(csv_file_name):
#    pos = 0
#    neg = 0
#    
#    file = pd.read_csv(csv_file_name) 
#    file = np.array(file)
#    
#    image_count = 0
#    for file_name in file[:, 0]:
#        if(file[image_count, 1] == 1):
#            pos += 1
#        elif(file[image_count, 1] == 2):
#            neg += 1
#        else:
#            print(str(file[image_count, 0]) + "is not decided")
#        
#        image_count += 1
#        
#    acc = (pos/image_count) * 100
#    
#    return acc, pos, neg, image_count
#
#
#acc, pos, neg, image_count = measure_pred_acc(csv_file_name_2)


       
rejected_folder = "./Rejected/"

rest_neg_folder = './Troughs_Model/model_AccuEdges/Rest_Neg_images/'


def remove_the_rejected_images(csv_file_name, rejected_folder, deleted_folder):
    
    file = pd.read_csv(csv_file_name) 
    file = np.array(file)
    image_count = 0
    for file_name in file[:, 0]:
        if(file[image_count, 1] == 2):
            shutil.move(rejected_folder + file_name, deleted_folder + file_name) 
            print(file_name + ' rejected, so removed - ' + str(file[image_count, 1]))
        
        image_count += 1

    return image_count

image_count = remove_the_rejected_images(csv_file_name, rejected_folder, rest_neg_folder)








