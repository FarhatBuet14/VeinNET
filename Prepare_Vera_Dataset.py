import numpy as np
import cv2
import os
import shutil

import torch

#############################  Gather All Data  ###############################
###############################################################################

def gather_all_data(Sep_data_folder, data_folder):
    persons = os.listdir(Sep_data_folder)
    for person in persons:
        
        then = Sep_data_folder + person + "/"
        sessions = os.listdir(then)
        for session in sessions:
            clicks = os.listdir(then + session + "/")
            for click in clicks:
                shutil.copy(then + session + "/" + click, data_folder)
    print("Finished gathering..")

#########################  Manual Selection of points  ########################
###############################################################################

def manual_selection(data_folder, extraction_folder, data_file = None):
    if(data_file):
        data = np.load(data_file)
        # images = list(data['images'])
        points = list(data['manual_points'])
        names = list(data['names'])
    
    else:
        points = []
        names = []
        # images = []
    
    count = len(names)
    point = []
    files = os.listdir(data_folder)

    for i in range(0, len(files)):
        if (files[i] in names):
            pass
        else:
            # Define mouse event
            def mouse_drawing(event, x, y, flags, params):
                if event == cv2.EVENT_LBUTTONDOWN:
                    print(str(x) + " , " + str(y))
                    point.append(x)
                    point.append(y)
            
            cv2.namedWindow("Frame")
            cv2.setMouseCallback("Frame", mouse_drawing)
                
            img = cv2.imread(data_folder + files[i])    
            cv2.imshow("Frame", img)
            while True:    
                key = cv2.waitKey(1)
                if key == 27: # Press Esc
                    if(len(point) == 4):
                        # images.append(img)
                        points.append(point)
                        names.append(files[i])
                        point = []
                        count += 1
                        print("done - " + str(count) + " - " + files[i])
                        cv2.destroyAllWindows()
                        if(count == 2200):
                            points = np.array(points)
                            np.savez(extraction_folder + 'manual_selsction_data.npz',
                                    manual_points = points,
                                    names = names)
                            print("Finished manual data selection..")
                            return
                        break
                    else:
                        print("Points Detected " + str(len(point)))
                        points = np.array(points)
                        np.savez(extraction_folder + 'manual_selsction_data.npz',
                                manual_points = points,
                                names = names)
                        cv2.destroyAllWindows()

                        return
                elif key == 32: # Press Space
                    points = np.array(points)
                    np.savez(extraction_folder + 'manual_selsction_data.npz',
                            manual_points = points,
                            names = names)
                    cv2.destroyAllWindows()
                    
                    return

    points = np.array(points)

    np.savez(extraction_folder + 'manual_selsction_data.npz',
            manual_points = points,
            names = names)

############################  Remove Manual data errors  ######################
###############################################################################

def remove_manual_data_error(data_file):
    data = np.load(data_file)
    points = list(data['manual_points'])
    names = list(data['names'])
    indeces = []
    for index in range(0, len(points)):
        if (len(points[index]) != 4):
            indeces.append(index)
    count = 0
    for index in indeces:
        i = index - count
        points.pop(i)
        names.pop(i)
        count += 1
    
    points = np.array(points)
    np.savez(extraction_folder + 'manual_selsction_data_after_correction.npz',
            manual_points = points,
            names = names)

#####################################  Main  ##################################
###############################################################################

if __name__ == "__main__":
    Sep_data_folder = "./Data/Vera Dataset/raw/"
    data_folder = "./Data/Vera Dataset/all/"
    extraction_folder = "./Data/Vera Dataset/"
    data_file = extraction_folder + 'manual_selsction_data.npz'
    # data_file = None

    # gather_all_data(Sep_data_folder, data_folder)

    # manual_selection(data_folder, extraction_folder, data_file)

    remove_manual_data_error(data_file)

    print("Finished")
