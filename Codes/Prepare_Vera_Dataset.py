import numpy as np
import cv2
import os
import shutil

import torch

import veinloss

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

##############################  Draw points on images  ########################
###############################################################################

def draw_poins_on_images(data_folder, data_file, image_with_point_folder):
    data = np.load(data_file)
    points = list(data['manual_points'])
    names = list(data['names'])
    for sample in range(0, len(names)):
        img = cv2.imread(data_folder + names[sample])
        for point in np.array(points[sample]).reshape(2, 2):   
            point = np.array(point).astype(int)
            cv2.circle(img, (point[0], point[1]), 
                    20, (255, 255, 255), -1)
        cv2.imwrite(image_with_point_folder + names[sample], img)

    print("Finished Drawing..")

###############################  Calculate Vein Loss  #########################
###############################################################################

def cal_vein_loss(data_folder, data_file, aug_data_file, 
                    vein_image_folder, loss_class, test_set):
    if(test_set == "Vera"):
        data = np.load(data_file)
        points = torch.tensor(data['manual_points']).cuda()
        names = data['names']
    
    elif(test_set == "Bosphorus"):
        data = np.load(data_file)
        X_names = data['X_train_names'].astype(str)
        y = data['y_train']
        data = np.load(aug_data_file)
        X_train_aug_names = data['X_train_aug_names'].astype(str)
        y_train_aug = data['y_train_aug'].reshape((-1, 4))

        # Concatenate main data and augmented data
        names = np.concatenate((X_names, X_train_aug_names), axis = 0)
        points = torch.tensor(np.concatenate((y, y_train_aug), axis = 0)).cuda()
    
    input = []
    ids = []
    for name in names:
        if(test_set == "Vera"):
            ids.append(int(name.split("_")[0]))
        elif(test_set == "Bosphorus"):
            import utils
            ids = utils.get_ID(names)
        input.append(cv2.imread(data_folder + name))
    
    input = torch.tensor(np.array(input)).cuda()
    ids = torch.tensor(np.array(ids)).cuda()

    loss = loss_class(points, points, input, names, ids)

    loss_logger = loss_class.loss_logger
    loss_names = loss_class.names

    return loss, loss_logger, loss_names

#####################################  Main  ##################################
###############################################################################

if __name__ == "__main__":
    
    test_set = "Vera"
    if(test_set == "Vera"):
        Sep_data_folder = "./Data/Vera Dataset/raw/"
        data_folder = "./Data/Vera Dataset/all/"
        extraction_folder = "./Data/Vera Dataset/Extraction/"
        data_file = extraction_folder + 'manual_selsction_data_after_correction.npz'
        aug_data_file = None
        image_with_point_folder = extraction_folder + "Data_with_points/"
        vein_image_folder = extraction_folder + "Extracted ROI/"
        bounding_box_folder = extraction_folder + "Extracted Bounding Box/"
    elif(test_set == "Bosphorus"):
        data_folder = "./Data/Train/"
        extraction_folder = "./Data/Extraction/"
        data_file = extraction_folder + 'Train_data_without_augmentation.npz'
        aug_data_file = extraction_folder + 'Augmented_Train_data.npz'
        image_with_point_folder = extraction_folder + "Data_with_points/"
        vein_image_folder = extraction_folder + "Extracted ROI/"
        bounding_box_folder = extraction_folder + "Extracted Bounding Box/"
    
    # data_file = None

    # gather_all_data(Sep_data_folder, data_folder)

    # manual_selection(data_folder, extraction_folder, data_file)

    # remove_manual_data_error(data_file)

    # draw_poins_on_images(data_folder, data_file, image_with_point_folder)
    
    loss_class = veinloss.Vein_loss_class(vein_image_folder, bounding_box_folder, 
                                        data_folder)
    loss , loss_logger, loss_names = cal_vein_loss(data_folder, data_file, aug_data_file, 
                                                    vein_image_folder, loss_class, test_set)

    print("Finished")
