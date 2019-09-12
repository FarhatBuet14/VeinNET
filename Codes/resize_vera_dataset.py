import numpy as np 
import os
import shutil
import cv2

import imgaug.augmenters as iaa

# # Get the path of current working directory 
# path = os.getcwd() 


data_folder = "./Data/Vera Dataset/Test/"
data_file = data_folder + "All_raw_data_with_points_resized.npz"

# resized_folder = "all_resized/"

data = np.load(data_file)
points = list(data["manual_points"])
names = list(data["names"])


train_names = []
train_points = []

files = os.listdir(data_folder)

for file in files:
        if(file in names):
                train_names.append(names[names.index(str(file))])
                train_points.append(points[names.index(str(file))])
        else:
                print("File not found - " + str(file))

train_points = np.array(train_points).reshape((-1, 2, 2))

np.savez(data_folder + "Test.npz",
        manual_points = np.array(train_points),
        names = train_names)






# aug_names = []
# aug_points = []

# for index in range(0, len(names)):
#     image = cv2.imread(data_folder + names[index])
#     # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     point = np.array(points[index]).reshape((1, 2, 2))
#     image_aug, kps_aug = iaa.Resize({"height": 240, "width": 300})(image = image, keypoints = point)
#     cv2.imwrite(resized_folder + names[index], image_aug)
#     aug_names.append(names[index])
#     aug_points.append(kps_aug.reshape(4))

# aug_points = np.array(aug_points)
# np.savez(resized_folder + "test_data_resized.npz",
#         names = aug_names,
#         manual_points = aug_points)



print("Finished")



