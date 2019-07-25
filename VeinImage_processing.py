import os
import cv2 
import numpy as np
import imutils
from skimage import morphology

path = './Data/Dataset/'
filenames = os.listdir(path)


#########################  Find Accumulated Image  ############################
###############################################################################

def get_accumEdged(image):
    accumEdged = np.zeros(image.shape[:2], dtype="uint8")

    for chan in cv2.split(image):
        chan = cv2.medianBlur(chan, 3)
        edged = cv2.Canny(chan, 50, 150)
        accumEdged = cv2.bitwise_or(accumEdged, edged)
        
    return accumEdged

###############################################################################
image = cv2.imread(path + filenames[70])
accumEdged = get_accumEdged(image)
############################# Stpe - 1 ########################################
pr_img = cv2.ximgproc.guidedFilter(image, image, 13, 70)
pr_img = np.array(pr_img)
############################# Stpe - 2 ########################################
#-----Converting image to LAB Color model
lab= cv2.cvtColor(pr_img, cv2.COLOR_BGR2LAB)
#-----Splitting the LAB image to different channels
l, a, b = cv2.split(lab)
#-----Applying CLAHE to L-channel
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
#-----Merge the CLAHE enhanced L-channel with the a and b channel
limg = cv2.merge((cl,a,b))
#-----Converting image from LAB Color model to RGB model
final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
############################## Stpe - 3 #######################################
img_bin=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,21,2)
img_bin = 255 - img_bin
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img_bin,kernel,iterations = 1)
kernel = np.ones((5,5),np.uint8)
img_dilation = cv2.dilate(erosion, kernel, iterations=1) 
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
# opened_mask = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)
# masked_img = cv2.bitwise_and(img_bin, img_bin, mask=opened_mask)



# opening = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# #find all your connected components (white blobs in your image)
# nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=4)
# #connectedComponentswithStats yields every seperated component with information on each of them, such as size
# #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
# sizes = stats[1:, -1]; nb_components = nb_components - 1

# # minimum size of particles we want to keep (number of pixels)
# #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
# min_size = 12000

# #your answer image
# mask = np.zeros((output.shape), dtype = 'uint8')
# #for every component in the image, you keep it only if it's above min_size
# for i in range(0, nb_components):
#     if sizes[i] >= min_size:
#         mask[output == i + 1] = 255
#
# opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

############################## Stpe - 4 #######################################
cv2.imshow("Preprocessed Image", accumEdged)
while(True):
    key = cv2.waitKey(0)
    if(key == 32):
        cv2.destroyAllWindows()

print("Finished")
