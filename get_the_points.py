# USAGE
# python click_and_crop.py --image jurassic_park_kitchen.jpg

# import the necessary packages
import cv2
import numpy as np
import pandas as pd 


total_samples = 3
full = np.zeros((240*300*3 + 4, total_samples))
refPt = []

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt
    
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))

		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

k = 1
for i in range(1, int(total_samples/2)+1):
    for j in range(1, 4):
        # load the image, clone it, and setup the mouse callback function
        image = cv2.imread("p" + str(i) + "_left_act_" + str(j) + ".bmp")
        orig = image.copy()
        clone = image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_crop)
        
        # keep looping until the 'q' key is pressed
        while True:
        	# display the image and wait for a keypress
        	cv2.imshow("image", image)
        	key = cv2.waitKey(1) & 0xFF
        
        	# if the 'r' key is pressed, reset the cropping region
        	if key == ord("r"):
        		image = clone.copy()
        
        	# if the 'c' key is pressed, break from the loop
        	elif key == ord("c"):
        		break
        
        # close all open windows
        cv2.destroyAllWindows()
        
        orig = np.reshape(orig, (240*300*3, 1))
        full[:216000, k-1:k] = orig 
        full[216000+0] [k-1] = refPt[0][1]
        full[216000+1] [k-1] = refPt[1][1]
        full[216000+2] [k-1] = refPt[0][0]
        full[216000+3] [k-1] = refPt[1][0]
        
        k = k + 1

pd.DataFrame(full).to_csv("dataset.csv")




