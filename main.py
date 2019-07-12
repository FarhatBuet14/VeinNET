############################  Import Libraries  ######################

import os
import numpy as np
import time
import sys
import cv2
from os.path import join
from PIL import Image
import pandas as pd
import matplotlib as plt

########################## Import Model Class ########################

from VeinNetTrainer import VeinNetTrainer

######################### Run TRAINING  #########################

def runTrain(nnArchitecture, gpu = False):
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    pathDirData = "./Data/Full X_train_data/"
    pathFileTest = "./Data/Test_Data/"
    
    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnInChanCount = 3
    nnIsTrained = True
    nnClassCount = 4
    
    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 1
    trMaxEpoch = 20
    checkpoint = None
    
    pathModel = 'm-' + timestampLaunch + '.pth.tar'
    
    print ('Training NN architecture = ', nnArchitecture)
    vein = VeinNetTrainer(gpu)
    vein.training(pathDirData, nnArchitecture, nnIsTrained, 
                nnInChanCount, nnClassCount, trBatchSize, 
                trMaxEpoch, timestampLaunch, checkpoint)
    # vein.test(pathFileTest, pathModel, nnArchitecture, 
    #         nnInChanCount, nnClassCount, nnIsTrained, 
    #         trBatchSize, timestampLaunch)

######################### RUN TESTING #########################

def runTest(nnArchitecture, gpu = False):

    pathFileTest = "./Data/Test_Data/"
    nnIsTrained = True
    
    nnInChanCount = 3
    nnClassCount = 4
    trBatchSize = 16
    
    pathModel = './Model_Output/m-25012018-123527.pth.tar'
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    vein = VeinNetTrainer(gpu)
    vein.test(pathFileTest, pathModel, nnArchitecture, 
            nnInChanCount, nnClassCount, nnIsTrained, 
            trBatchSize, timestampLaunch)

######################### Main Function  #########################

nnArchitecture = 'resnet18'
runTest(nnArchitecture, gpu = True)
#runTrain(nnArchitecture, gpu = True)

print("finished")
