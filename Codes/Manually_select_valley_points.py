############################  Import Libraries  ###############################
###############################################################################

import cv2
import numpy as np

data_folder = "./Data/All/"
extraction_folder = "./Extracted/"

error_data = np.load(extraction_folder + 'result.npz')
error_files = error_data['error_files']
neg_vein = error_data['neg_vein']

total_images = np.concatenate((error_files, neg_vein), axis = 0)

points = []
point = []
names = []
images = []
count = 0
for i in range(0, len(total_images)):

    # Define mouse event
    def mouse_drawing(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(str(x) + " , " + str(y))
            point.append(x)
            point.append(y)
    
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_drawing)
        
    img = cv2.imread(data_folder + total_images[i])    
    cv2.imshow("Frame", img)
    while True:    
        key = cv2.waitKey(1)
        if key == 27:
            images.append(img)
            points.append(point)
            names.append(total_images[i])
            point = []
            count += 1
            print("done - " + str(count))
            cv2.destroyAllWindows()
            break

points = np.array(points)

np.savez(extraction_folder + 'manual_selsction_data.npz',
         images = images,
         manual_points = points,
         names = names)

