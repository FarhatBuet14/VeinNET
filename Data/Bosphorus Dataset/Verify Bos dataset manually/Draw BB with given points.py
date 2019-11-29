import numpy as np
import cv2
import math
import imgaug.augmenters as iaa
import re
import imutils

pathDirData = "./Train/"
bounding_box_folder = "./Extracted/"

data = np.load(pathDirData + 'Bos_train_val_data_redefined_80_point_also.npz') 
X_names = data['names']
y = np.array(data['y'])
X_val_names = data['names_val']
y_val = np.array(data['y_val'])
img_name = np.concatenate((X_names, X_val_names), axis = 0)
output = np.concatenate((y, y_val), axis = 0)

# img_name = data['names'].astype(str)
# output = np.array(data['manual_points']).reshape((-1, 4))

# xt = []
# xv = []
# yt = []
# yv = []

# count = 0
# for index in range(0, len(X_names)):
#     if("80." not in X_names[index]):
#         xt.append(X_names[index])
#         yt.append(y[index])
#     else:
#         count += 1
#         print("deleting --- " + X_names[index])

# for index in range(0, len(X_val_names)):
#     if("80." not in X_val_names[index]):
#         xv.append(X_val_names[index])
#         yv.append(y_val[index])
#     else:
#         count += 1
#         print("deleting --- " + X_val_names[index])


# count = 0
# for index in range(0, len(X_names)):
#     xt.append(X_names[index])
#     if("fl" in X_names[index]):
#         temp = y[index].reshape((2, 2))
#         fs = []
#         fs.append(np.array(temp[1]))
#         fs.append(np.array(temp[0]))
#         yt.append(np.array(fs).reshape(4))
#         count += 1
#         print("swaping --- " + X_names[index])

#     else:
#         yt.append(y[index])

# for index in range(0, len(X_val_names)):
#     xv.append(X_val_names[index])
#     if("fl" in X_val_names[index]):
#         temp = y_val[index].reshape((2, 2))
#         fs = []
#         fs.append(np.array(temp[1]))
#         fs.append(np.array(temp[0]))
#         yv.append(np.array(fs).reshape(4))
#         count += 1
#         print("swaping --- " + X_val_names[index])
#     else:
#         yv.append(y_val[index])


# xt = np.array(xt)
# xv = np.array(xv)
# yt = np.array(yt)
# yv = np.array(yv)

# np.savez("Bos_train_val_data_redefined_80_point_also.npz",
#         names = xt,
#         y = yt,
#         names_val = xv,
#         y_val = yv)

# print("Finished...")


















height = 90
width = 70
th = 10
thresh_h = 200
thresh_l = 70

for sample in range(0, len(img_name)):
     # Error removing for augmented data---------------------
    # file, point = str(img_name[sample]), output[sample]
    # if((file.find('_flrot_') != -1) | (file.find('_flrotVera_') != -1)):
    #     point1 = np.array(point[0:2])
    #     point2 = np.array(point[2:4])
    #     point_changed = []
    #     point_changed.append(point2)
    #     point_changed.append(point1)
    #     output[sample] = np.array(point_changed).reshape((1, 4))
    # # -------------------------------------------------------
    
    # Draw actual Troughs & save
    img = cv2.imread(pathDirData + img_name[sample])
    points = output[sample]
    points = points.reshape((2, 2))
    color = [(255, 255, 255), (0, 0, 0)] # Left - First - White, # Right - Second - Black
    count = 0
    for point in points:   
        point = np.array(point).astype(int)
        cv2.circle(img, (point[0], point[1]), 
                5, color[count], -1)
        count += 1
    raw = img

    # Get points
    top_left = output[sample, 0:2]
    top_right = np.array(output[sample, 2:4])
    
    # Find the angle to rotate the image
    angle  = (180/np.pi) * (np.arctan((top_left[1] - top_right[1])/
                            (top_left[0] - top_right[0] + 1e-08)))
    
    # Rotate the image to cut rectangle from the images
    points_pred = (output[sample]).reshape((1, 2, 2))
    img = cv2.imread(pathDirData + img_name[sample])
    image = []
    image.append(img)
    image = np.array(image)
    image_rotated , keypoints_pred_rotated = iaa.Affine(rotate=-angle)(images=image, 
                                keypoints=points_pred)
    image_rotated = image_rotated[0]
    keypoints_pred_rotated = np.array(keypoints_pred_rotated[0]).reshape((2, 2))
    # Rotated Points
    top_left = keypoints_pred_rotated[0]
    top_left[0] = top_left[0] - th
    top_right = keypoints_pred_rotated[1]
    top_right[0] = top_right[0] + th
    width = int(abs(top_right - top_left)[0])
    height = int(width * (90/80))
    centre = tuple([top_left[0] + int(width/2), top_left[1] + int(height/2)])
    
    # Draw Troughs
    points = keypoints_pred_rotated.reshape((2, 2))  
    color = [(255, 255, 255), (0, 0, 0)] # Left - First - White, # Right - Second - Black
    count = 0
    for point in points:   
        point = np.array(point).astype(int)
        cv2.circle(image_rotated, (point[0], point[1]), 
                5, color[count], -1)
        count += 1

    bottom_right = [int(top_left[0] + width) , int(top_left[1] + height)]

    # Draw Bounding Boxes and Save the image
    image_rotated = cv2.rectangle(image_rotated, tuple(top_left), tuple(bottom_right) , (0,0,0), 2)
    
    all_img = cv2.hconcat((raw, image_rotated))
    
    cv2.imwrite(bounding_box_folder + img_name[sample], 
                all_img)
    
    if((sample + 1) % 50 == 0):
        print("Done - " + str(sample + 1))


print("Finished Save the boxes...")




















































