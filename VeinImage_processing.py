import os
import cv2 
import numpy as np
import imutils

path = './Data/Dataset/'
filenames = os.listdir(path)


#########################  Find Accumulated Image  ############################
###############################################################################

def get_cntrImg(image, cnt_length_thresh = 100):
    accumEdged = np.zeros(image.shape[:2], dtype="uint8")

    for chan in cv2.split(image):
        chan = cv2.medianBlur(chan, 3)
        edged = cv2.Canny(chan, 50, 150)
        accumEdged = cv2.bitwise_or(accumEdged, edged)
    
    cnts = cv2.findContours(accumEdged, cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # Delete the short contours less than length_threshold
    cnts_nedded = []
    Length = []
    for c in cnts:
        if( cv2.arcLength(c, False) > cnt_length_thresh ):
            Length.append(cv2.arcLength(c, False))
            cnts_nedded.append(c)
    all_cnts = np.zeros((1, 1, 2))    
    for cnt in cnts_nedded:
        all_cnts = np.append(all_cnts, cnt, axis = 0)
    all_cnts = all_cnts[1:, :, :]
    all_cnts = np.reshape(all_cnts, (all_cnts.shape[0], 2))
    blank = np.zeros((240, 300, 3))
    cnt_image = cv2.drawContours(blank,
                    np.array(all_cnts).reshape((-1,1,2)).astype(np.int32),
                    -1, (255,255,255), 2)
        
    return accumEdged, cnt_image

###############################################################################

image = cv2.imread(path + filenames[12])
pr_img = cv2.ximgproc.guidedFilter(image, image, 13, 70)
pr_img = np.array(pr_img)
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
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
accu, cnt = get_cntrImg(final, 400)
final = np.zeros((240, 300))
for chan in cv2.split(cnt):
    final += chan

cv2.imshow("Preprocessed Image", final)
while(True):
    key = cv2.waitKey(0)
    if(key == 32):
        cv2.destroyAllWindows()

print("Finished")
