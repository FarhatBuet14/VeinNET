#############################  IMPORT LIBRARIES  ############################
import numpy as np
import cv2
import math
import imgaug.augmenters as iaa
import re
import imutils

import torch
from torch.autograd import Variable

######################### Custom Vein Loss #########################
####################################################################

class Vein_loss_class(torch.nn.Module):
    def __init__(self, opt):
        super(Vein_loss_class, self).__init__()
        self.opt = opt
        self.height = 90
        self.width = 70
        self.th = 10
        self.thresh_h = 200
        self.thresh_l = 70

    def get_vein_img(self, save_vein_pic = True, save_bb = True):
        crop = []
        for sample in range(0, self.totalInput):
            
            # Draw actual Troughs & save
            img = cv2.imread(self.opt.pathDirData + self.img_name[sample])
            points = self.output[sample]
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
            top_left = self.output[sample, 0:2]
            top_right = self.output[sample, 2:4]
            
            # Find the angle to rotate the image
            angle  = (180/np.pi) * (np.arctan((top_left[1] - top_right[1])/
                                    (top_left[0] - top_right[0] + 1e-08)))
            
            # Rotate the image to cut rectangle from the images
            points_pred = (self.output[sample]).reshape((1, 2, 2))
            points_test = (self.target[sample]).reshape((1, 2, 2))
            img = cv2.imread(self.opt.pathDirData + self.img_name[sample])
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = []
            image.append(img)
            image = np.array(image)
            image_rotated , keypoints_pred_rotated = iaa.Affine(rotate=-angle)(images=image, 
                                        keypoints=points_pred)
            _ , keypoints_test_rotated = iaa.Affine(rotate=-angle)(images=image, 
                                        keypoints=points_test)
            
            # Check if the image is fully rotated that left goes to the right side of hand
            if(keypoints_pred_rotated[0, 0, 0] > keypoints_pred_rotated[0, 1, 0]):
                # Again rotate the picture to 180 with the points
                image = image_rotated
                image_rotated , keypoints_pred_rotated = iaa.Affine(rotate=180)(images=image, 
                                        keypoints=keypoints_pred_rotated)
                _ , keypoints_test_rotated = iaa.Affine(rotate=180)(images=image, 
                                            keypoints=keypoints_test_rotated)

            image_rotated = image_rotated[0]
            keypoints_pred_rotated = keypoints_pred_rotated.reshape((2, 2))
            keypoints_test_rotated = keypoints_test_rotated.reshape((2, 2))
            
            # Rotated Points
            top_left = keypoints_pred_rotated[0]
            top_left[0] = top_left[0] - self.th
            top_right = keypoints_pred_rotated[1]
            top_right[0] = top_right[0] + self.th
            self.width = int(abs(top_right - top_left)[0])
            if(self.org[sample] == 0): # Vera
                self.height = int(self.width * (82/80))
            elif(self.org[sample] == 1): # Bosphorous
                self.height = int(self.width * (90/80))
            centre = tuple([top_left[0] + int(self.width/2), top_left[1] + int(self.height/2)])

            # Crop the Vein Image
            cropped = cv2.getRectSubPix(image_rotated, (self.width, self.height), 
                                    centre)
            crop.append(cropped)
            if(save_vein_pic):
                cv2.imwrite(self.opt.cropped_fldr + self.img_name[sample], cropped)
            
            # Draw Predicted Troughs
            points = keypoints_pred_rotated.reshape((2, 2))  
            color = [(255, 255, 255), (0, 0, 0)] # Left - White, # Right - Black
            count = 0  
            for point in points:   
                point = np.array(point).astype(int)
                cv2.circle(image_rotated, (point[0], point[1]), 
                        5, color[count], -1)
                count += 1
            
            # Draw Actual Troughs
            points = keypoints_test_rotated.reshape((2, 2))    
            for point in points:   
                point = np.array(point).astype(int)
                cv2.circle(image_rotated, (point[0], point[1]), 
                        5, (255, 0, 0), -1)

            bottom_right = [int(top_left[0] + self.width) , int(top_left[1] + self.height)]

            # Draw Bounding Boxes and Save the image
            image_rotated = cv2.rectangle(image_rotated, tuple(top_left), tuple(bottom_right) , (0,0,0), 2)
            all_img = cv2.hconcat((raw, image_rotated))
            if(save_bb):
                cv2.imwrite(self.opt.bounding_box_folder + self.img_name[sample], 
                            all_img)
        # crop = np.array(crop)
        return crop
    
    def forward(self,target, output, input, img_name, ids, org):
        
        self.target = target.cpu().numpy()
        self.output = output.cpu().data.numpy()
        self.input = input.cpu().numpy()
        self.id = ids.cpu().numpy()
        self.id = np.array(self.id, dtype = 'int32')
        self.org = org.cpu().numpy()
        self.img_name = img_name
        self.totalInput = len(self.id)

        vein_image = self.get_vein_img()
        vein_loss = 0
        # Calculate loss from extracted Vein Image
        loss_logger = []
        names = []
        for sample in range(0, self.totalInput):
            image = cv2.cvtColor(vein_image[sample], cv2.COLOR_RGB2GRAY)
            if(org[sample] == 0):      # Vera Vein loss calculation
                self.thresh_l = 38
                self.thresh_h - 250
            accu = ((image <= self.thresh_h)  & (image >= self.thresh_l))
            true = np.count_nonzero(accu)
            false = (accu.shape[0] * accu.shape[1]) - true
            loss = Variable(torch.tensor((false / (false + true))), requires_grad=True)
            vein_loss += loss
            loss_logger.append(loss)
            names.append(self.img_name[sample])

        self.loss_logger = loss_logger
        self.names = names

        vein_loss = vein_loss / self.totalInput
        
        return vein_loss * 100

if __name__ == "__main__":
    pass
    # main()
