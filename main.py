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
import torch

########################## Import Model Class ########################

from VeinNetTrainer import VeinNetTrainer

######################### Run TRAINING  #########################

def runTrain(nnArchitecture, gpu = False):
    
    torch.cuda.empty_cache()
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    pathDirData = "./Data/Full X_train_data/"
    train_Output_data = "./Model_Output/"

    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnInChanCount = 3
    nnIsTrained = True
    nnClassCount = 4
    
    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 32
    trMaxEpoch = 25
    loss_weights = [0.2, 0.8]
    # checkpoint = None
    pathModel = train_Output_data + 'm_' + timestampLaunch +  '.pth.tar'

    #---- Training settings: Vein Loss
    vein_loss = True
    cropped_fldr = train_Output_data + 'Cropped/'
    bounding_box_folder = train_Output_data + 'Prediction/'
    print('-' * 100)
    print ('Training NN architecture = ', nnArchitecture)
    vein = VeinNetTrainer(gpu)
    vein.training(pathDirData, pathModel, nnArchitecture, nnIsTrained, 
                nnInChanCount, nnClassCount, trBatchSize, 
                trMaxEpoch, loss_weights, timestampLaunch, None,
                vein_loss, cropped_fldr, bounding_box_folder)
    del vein

######################### RUN TESTING #########################

def runTest(nnArchitecture, gpu = False):

    pathFileTest = "./Data/Test_Data/"
    train_Output_data = "./Model_Output/"
    nnIsTrained = True
    
    nnInChanCount = 3
    nnClassCount = 4
    trBatchSize = 64
    loss_weights = [0.2, 0.8]
    
    pathModel = './Model_Output/m-25012018-123527.pth.tar'
    
    # timestampTime = time.strftime("%H%M%S")
    # timestampDate = time.strftime("%d%m%Y")
    # timestampLaunch = timestampDate + '-' + timestampTime

    vein_loss = True
    cropped_fldr = train_Output_data + 'Cropped/'
    bounding_box_folder = train_Output_data + 'Prediction/'
    
    vein = VeinNetTrainer(gpu)
    vein.test(pathFileTest, pathModel, nnArchitecture, 
            nnInChanCount, nnClassCount, nnIsTrained, 
            trBatchSize, loss_weights, vein_loss, 
            cropped_fldr, bounding_box_folder)

######################### Main Function  #########################

torch.cuda.empty_cache()
nnArchitecture = 'resnet18'
# runTest(nnArchitecture, gpu = True)
runTrain(nnArchitecture, gpu = True)
