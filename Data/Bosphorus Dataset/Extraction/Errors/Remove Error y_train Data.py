import numpy as np
import cv2
import os



#########################  Manual Selection of points  ########################
###############################################################################

def manual_selection(data_folder, extraction_folder, error):
    
    points = []
    names = []
    count = 0
    point = []
    files = os.listdir(data_folder)
    for i in range(0, len(files)):
        if (files[i] in error):
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
                        if(count == len(error)):
                            points = np.array(points)
                            np.savez(extraction_folder + 'error_data.npz',
                                    manual_points = points,
                                    names = names)
                            print("Finished manual data selection..")
                            return
                        break
                    else:
                        print("Points Detected " + str(len(point)))
                        points = np.array(points)
                        np.savez(extraction_folder + 'error_data.npz',
                                manual_points = points,
                                names = names)
                        cv2.destroyAllWindows()

                        return
                elif key == 32: # Press Space
                    points = np.array(points)
                    np.savez(extraction_folder + 'error_data.npz',
                            manual_points = points,
                            names = names)
                    cv2.destroyAllWindows()
                    
                    return

    points = np.array(points)

    np.savez(extraction_folder + 'error_data.npz',
            manual_points = points,
            names = names)

############################  Remove Manual data errors  ######################
###############################################################################

def remove_manual_data_error(data_file, error_corrected):
    data = np.load(data_file)
    names = list(data['X_train_aug_names'].astype(str))
    points = list(data['y_train_aug'])

    data = np.load(error_corrected)
    err_points = list(data['manual_points'])
    err_names = list(np.array(data['names']).astype(str))

    
    indeces = []
    err_indeces = []
    for index in range(0, len(names)):
        if (names[index] in err_names):
            indeces.append(index)
            err_indeces.append(err_names.index(names[index]))
    
    count = 0
    for index in indeces:
        i = index - count
        points.pop(i)
        names.pop(i)
        count += 1
    
    for index in err_indeces:
        points.append(err_points[index].reshape((2, 2))) # 
        names.append(err_names[index])
    
    count = 0
    err_indeces.sort()
    for index in err_indeces:
        i = index - count
        err_points.pop(i)
        err_names.pop(i)
        count += 1

    points = np.array(points)
    err_points = np.array(err_points)

    np.savez(extraction_folder + 'Augmented_Train_data.npz',
            y_train_aug = points,
            X_train_aug_names = names)
    np.savez(extraction_folder + 'error_data.npz',
            manual_points = err_points,
            names = err_names)
    

######################################  Main  #################################
###############################################################################

if __name__ == "__main__":

    data_folder = "./Data/Train/"
    extraction_folder = "./Data/Extraction/"
    data_file = extraction_folder + 'Train_data_without_augmentation.npz'
    aug_data_file = extraction_folder + 'Augmented_Train_data.npz'
    error_corrected = extraction_folder + 'error_data.npz'

    # data = np.load(error_corrected)
    # manual_points = data['manual_points']
    # names = np.array(data['names']).astype(str)
    
    
    # data = np.load(data_file)
    # X_names = data['X_train_names'].astype(str)
    # y = data['y_train']
    # data = np.load(aug_data_file)
    # X_train_aug_names = data['X_train_aug_names'].astype(str)
    # y_train_aug = data['y_train_aug'].reshape((-1, 4))

    # names = np.concatenate((X_names, X_train_aug_names), axis = 0)
    # points = np.concatenate((y, y_train_aug), axis = 0)

    error_folder = extraction_folder + "Errors/Reverse Error/"

    error = os.listdir(error_folder)

    # for sample in range(0, len(names)):
    #     if(str(names[sample]) in error):
    #         img = cv2.imread(data_folder + str(names[sample]))
    #         # print(names[sample])
    #         # print(points[sample])
    #         po = np.array(points[sample]).reshape((2, 2))
    #         color = [(255, 255, 255), (0, 0, 0)] # Left - White, # Right - Black
    #         count = 0
    #         for point in po:
    #             point = np.array(point).astype(int)
    #             cv2.circle(img, (point[0], point[1]), 
    #                     5, color[count], -1)
    #             count += 1
    #         cv2.imwrite(error_folder + "/test/" + names[sample], img)

    # manual_selection(data_folder, extraction_folder, error)

    remove_manual_data_error(aug_data_file, error_corrected)

    print("Finished..")



