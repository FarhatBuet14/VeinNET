import numpy as np 
import os
import cv2
import glob
import torch
from torchvision import transforms
from PIL import Image
import argparse


def resizing_inputs(test_folder):
    resized_folder = test_folder + "Resized/"
    files = os.listdir(test_folder)

    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)
    else:
        test = resized_folder + '*'
        r = glob.glob(test)
        for i in r:
            os.remove(i)
    count = 0
    for file in files:
        if(len(file.split(".")) == 2): 
            image = cv2.imread(test_folder + file)
            image = cv2.resize(image, (300, 240))
        
            cv2.imwrite(resized_folder + file, image)
            count += 1
            if(count % 50 == 0): print(count)
            
    print("Finished Resizing..")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training VeinNetTrainer...')
    parser.add_argument("--pathDirData", type=str, metavar = "", help="Train/Test/Validation Data Directory", default = "./2/")
    parser.add_argument("--nnArchitecture", type=str, metavar = "", help="Name of Model Architechture", default = "resnet50")
    parser.add_argument("--nnIsTrained", action = "store_true", help="Use Trained Network or not", default = True)
    parser.add_argument("--nnInChanCount", type=int, metavar = "", help="Number of Input channel", default = 3)
    parser.add_argument("--nnClassCount", type=int, metavar = "", help="Number of predicted values", default = 4)
    parser.add_argument("--gpu", action = "store_true", help="Use GPU or not", default = True)
    parser.add_argument("--checkpoint", type=str, metavar = "", help="Checkpoint File", 
    default = './Model_Output/' + '73_____2.162804449172247_____17.516155242919922_____3.787345668247768_____8.933741569519043_____-0.35563125610351565_____-0.34951141476631165.pth.tar')
    
    opt = parser.parse_args()
    print(opt)

    resizing_inputs(opt.pathDirData)
    
    print("Finished")


