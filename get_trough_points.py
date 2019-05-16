import numpy as np
import imutils
import cv2


def get_accumEdged(image):
    accumEdged = np.zeros(image.shape[:2], dtype="uint8")

    # loop over the blue, green, and red channels, respectively
    for chan in cv2.split(image):
        chan = cv2.medianBlur(chan, 3)
        edged = cv2.Canny(chan, 150, 200)
        accumEdged = cv2.bitwise_or(accumEdged, edged)
            
    cv2.imshow("Edge Map", accumEdged)
    
    while True:    
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        
    return accumEdged




def find_contour_needed(accumEdged, area_threshold = 100):
    
    contour_image = accumEdged
    
    cnts = cv2.findContours(contour_image, cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    cnts_nedded = []
    area = []
    for c in cnts:
        if(cv2.contourArea(c) > area_threshold):
            area.append(cv2.contourArea(c))
            cnts_nedded.append(c)

    return cnts_nedded, area



def get_trough_points(image, cnts_needed):
    troughs = []
    for cnt in cnts_nedded:
        
        cnt_x = cnt[:, 0, 0]
        cnt_y = cnt[:, 0, 1]
        
    #    plt.figure(1, figsize=(12, 6))
    #    plt.subplot(211)
    #    plt.plot(cnt_y)
    #    plt.ylabel('Controur_points')
    #    
        grad = np.gradient(cnt_y)
    #    plt.figure(1, figsize=(12, 6))
    #    plt.subplot(212)
    #    plt.plot(grad)
    #    plt.ylabel('Gradients')
    #    plt.show()
    
        troughs_arg = []
        for i in range(0, len(grad)):
            if((grad[i] < 0)  &  (grad[i-1] > 0) & (i>0)):
                troughs_arg.append(i)
        
        troughs_array = np.zeros((len(troughs_arg), 1, 2))
        for i in range(0, len(troughs_arg)):
            troughs_array[i, :, :] = [cnt_x[troughs_arg[i]], cnt_y[troughs_arg[i]]]
            cv2.circle(image, (int(troughs_array[i, 0, 0]), int(troughs_array[i, 0, 1])), 5, (0, 255, 0), -1)
            
        troughs.append(troughs_array)
    
    
        
    cv2.imshow("Trough point", image)
        
    while True:    
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break

    return troughs

############## Main Code ############



image_name = "p1_left_act_3.bmp"
image = cv2.imread(image_name)

accumEdged = get_accumEdged(image)
area_threshold = 40
[cnts_nedded, area_cnt] = find_contour_needed(accumEdged, area_threshold)
troughs = get_trough_points(image.copy(), cnts_nedded)
















