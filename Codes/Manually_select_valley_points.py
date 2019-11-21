############################  Import Libraries  ###############################
###############################################################################

import cv2
import numpy as np
import os

##############################  Data Collection  ##############################
###############################################################################
#dataset = "2"
#data_folder = "./Data/" + dataset + "/"
#extraction_folder = "./Extracted/"
#
#total_images = os.listdir(data_folder)
#total_data = len(total_images)
#
## ---- Load past data
#prepapred_data = np.load(extraction_folder + 'manual_selection_data_' + dataset + '.npz')
#images = list(prepapred_data['images'])
#points = list(prepapred_data['manual_points'])
#names = list(prepapred_data['names'])
#count = len(points)
#for name in names:
#    if(str(name) in total_images):
#       total_images.remove(str(name)) 
#if(total_data - len(total_images) == count):
#    print("Successfully added the past data")
#
## ---- Load completed data
#prepapred_data = np.load(extraction_folder + 'completed_data_' + dataset + '.npz')
#comp_names = list(prepapred_data['names'])
#count = count + len(comp_names)
#for name in comp_names:
#    if(str(name) in total_images):
#       total_images.remove(str(name)) 
#if(total_data - len(total_images) == count):
#    print("Successfully added the completed data")
#
## ---- Main Code
#point = []
#for i in range(0, len(total_images)):
#
#    # Define mouse event
#    def mouse_drawing(event, x, y, flags, params):
#        if event == cv2.EVENT_LBUTTONDOWN:
#            print(str(x) + " , " + str(y))
#            point.append(x)
#            point.append(y)
#    
#    cv2.namedWindow("Frame")
#    cv2.setMouseCallback("Frame", mouse_drawing)
#        
#    img = cv2.imread(data_folder + total_images[i])
#    cv2.imshow("Frame", img)
#    while True:    
#        key = cv2.waitKey(1)
#        if key == 27:
#            images.append(img)
#            points.append(point)
#            names.append(total_images[i])
#            point = []
#            count += 1
#            print("done - " + str(count))
#            cv2.destroyAllWindows()
#            break
#        
#        elif key == 32:
#            print("Exit.. Total - " + str(count))
#            cv2.destroyAllWindows()
#            break
#    if key == 32:
#        break
#
#points = np.array(points)
#
## ---- Save data
#np.savez(extraction_folder + 'manual_selection_data_' + dataset + '.npz',
#         images = images,
#         manual_points = points,
#         names = names)

dataset = "1"
extraction_folder = "./"

# ---- Load past data
prepapred_data = np.load(extraction_folder + 'manual_selection_data_' + dataset + '.npz')
images = list(prepapred_data['images'])
points = list(prepapred_data['manual_points'])
names = list(prepapred_data['names'])


prepapred_data = np.load(extraction_folder + 'manual_selection_data_' + dataset + '_2' + '.npz')
images = images + list(prepapred_data['images'])
points = points + list(prepapred_data['manual_points'])
names = names + list(prepapred_data['names'])


#prepapred_data = np.load(extraction_folder + 'manual_selection_data_' + dataset + '_3' + '.npz')
#images = images + list(prepapred_data['images'])
#points = points + list(prepapred_data['manual_points'])
#names = names + list(prepapred_data['names'])

points = np.array(points)

np.savez(extraction_folder + 'final_test_data_' + dataset + '.npz',
         images = images,
         manual_points = points,
         names = names)




















