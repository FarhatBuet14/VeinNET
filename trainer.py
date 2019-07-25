#############################  IMPORT LIBRARIES  ############################
import numpy as np
import time
import pandas as pd
from os.path import join
from PIL import Image
import cv2
import math
import imgaug.augmenters as iaa
import re
import imutils
import argparse

import torch
import torch.backends.cudnn as cudnn
from torchsummary import summary
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as func
from torchvision import transforms, datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import Module

import dataset_builder, utils, model

#######################  Define VeinBetTrainer Class  #########################

class VeinNetTrainer():

    #---- Train the VeinNet network 
    #---- pathDirData - path to the directory that contains images
    #---- nnArchitecture - model architecture 'resnet50', 'resnet34' or 'resnet18'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- SeedlingDataset - Dataset Generator 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training


    ######################### Initialization #########################
    ##################################################################

    def __init__(self, gpu = False):
        self.gpu = gpu
        torch.cuda.empty_cache()
    
    ######################### Epoch Train #########################
    ###############################################################
    
    def epochTrain (self, model, dataLoader, optimizer, scheduler, trBatchSize,
                    epochMax, classCount, loss_class, loss_weights, vein_loss = False,
                    cropped_fldr = None, bounding_box_folder = None, data_folder = None):
        
        model.train()
        w_mae, w_veinLoss = loss_weights
        loss_v = 0
        for batchID, (input, target, img_name) in enumerate (dataLoader):
            # torch.cuda.empty_cache()
            
            id = target[:, 0]
            target = target[:, 1:]
            varInput = torch.autograd.Variable(input.cuda())
            varOutput = model(varInput)
            output = (Variable(varOutput).data).cpu()

            del varInput, varOutput
            loss = Variable(func.mse_loss(output, target).data, requires_grad=True)
            # loss += loss_class.calculate(target, output)
            # if(vein_loss):
            #     vein_loss_class = utils.Vein_loss_class(target, output, img_name,
            #                                             input, cropped_fldr, 
            #                                             bounding_box_folder,
            #                                             data_folder, id)
            #     vein_loss_class.get_vein_img(save_vein_pic = True,
            #                                             save_bb = True)
            #     loss_v += vein_loss_class.cal_vein_loss()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            del input, target, id, output#, vein_loss_class

        loss = (loss / (batchID+1))  * w_mae
        # loss_v = (loss_v / (batchID+1))  * w_veinLoss
        return loss, loss_v

    ######################### Epoch Validation #########################
    ####################################################################
    
    def epochVal (self, model, dataLoader, optimizer, scheduler, trBatchSize,
                    epochMax, classCount, loss_class, loss_weights, vein_loss = False,
                    cropped_fldr = None, bounding_box_folder = None, data_folder = None):
        
        model.eval ()
        w_mae, w_veinLoss = loss_weights
        loss = 0
        loss_v = 0
        with torch.no_grad():
            for batchID, (input, target, img_name) in enumerate (dataLoader):
                id = target[:, 0]
                target = target[:, 1:]
                varInput = torch.autograd.Variable(input.cuda())
                varOutput = model(varInput)
                output = (Variable(varOutput).data).cpu()

                del varInput, varOutput
                loss += func.mse_loss(output, target)
                # loss += loss_class.calculate(target, output)
                # if(vein_loss):
                #     vein_loss_class = utils.Vein_loss_class(target, output, img_name,
                #                                             input, cropped_fldr, 
                #                                             bounding_box_folder, 
                #                                             data_folder, id)
                #     vein_loss_class.get_vein_img(save_vein_pic = True,
                #                                             save_bb = True)
                #     loss_v += vein_loss_class.cal_vein_loss()

                del input, target, output, id#, vein_loss_class

            loss = (loss / (batchID+1))  * w_mae
            # loss_v = (loss_v / (batchID+1))  * w_veinLoss
        return loss, loss_v
    
    ######################### Train Function #########################
    ##################################################################
    
    def training (self, pathDirData, pathModel, nnArchitecture, 
                nnIsTrained, nnInChanCount, nnClassCount,
                trBatchSize, trMaxEpoch, loss_weights, launchTimestamp = None, 
                checkpoint = None, vein_loss = False, 
                cropped_fldr = None, bounding_box_folder = None):
        
        training_model = model.load_model(nnArchitecture, nnIsTrained, 
                                        nnInChanCount, nnClassCount)

        #-------------------- SETTINGS: DATASET BUILDERS
        trans = transforms.Compose([transforms.ToTensor()])
        train_data, valid_data = dataset_builder.dataset_builder(pathDirData)
        train_set = dataset_builder.SeedlingDataset(train_data, pathDirData, 
                                    transform = trans, normalize = True)
        val_set = dataset_builder.SeedlingDataset(valid_data, pathDirData, 
                                transform = trans, normalize = True)

        train_loader = DataLoader(train_set, batch_size = trBatchSize, shuffle = True)
        valid_loader = DataLoader(val_set, batch_size = trBatchSize, shuffle = True)
        
        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (training_model.parameters(), lr  =1e-3)#, betas=(0.9, 0.999), 
                                #eps=1e-08, weight_decay=1e-5)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, 
                                                    patience = 5, mode = 'min')
                
        #-------------------- SETTINGS: LOSS
        # loss_class = utils.Cal_loss(loss_type = 'mae')
        loss_class = None

        # #---- Load checkpoint 
        # if checkpoint != None:
        #    modelCheckpoint = torch.load(checkpoint)
        #    model.load_state_dict(modelCheckpoint['state_dict'])
        #    optimizer.load_state_dict(modelCheckpoint['optimizer'])

        
        #---- TRAIN THE NETWORK
        lossMIN = 100000
        print('-' * 50 + 'Start Training' + '-' * 50)

        for epochID in range (0, trMaxEpoch):

            timestampSTART = time.strftime("%d%m%Y") + '-' + time.strftime("%H%M%S")
            lossTrain, lossTrain_v = self.epochTrain (training_model, train_loader, optimizer, 
                                    scheduler, trBatchSize, trMaxEpoch, nnClassCount, 
                                    loss_class, loss_weights, vein_loss,
                                    cropped_fldr, bounding_box_folder, pathDirData)
            
            lossVal, lossVal_v = self.epochVal (training_model, valid_loader, optimizer, 
                                            scheduler, trBatchSize, trMaxEpoch, 
                                            nnClassCount, loss_class, loss_weights, 
                                            vein_loss, cropped_fldr, 
                                            bounding_box_folder, pathDirData)
            
            timestampEND = time.strftime("%d%m%Y") + '-' + time.strftime("%H%M%S")
            
            scheduler.step(lossVal.data)
            
            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save({'epoch': epochID + 1, 'state_dict': training_model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, pathModel)
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + ']')
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + ']')
            print('Train_loss= ' + str(lossTrain.data) + 'Val_loss= ' + str(lossVal.data))
            # print('Train_loss_vein= ' + str(lossTrain_v.data) + 'Val_loss_vein= ' + str(lossVal_v.data))
            print('-' * 100)
        
        # torch.cuda.empty_cache()
        print('-' * 50 + 'Finished Training' + '-' * 50)
        print('-' * 100)

    ######################### Test Function #########################
    #################################################################
    
    def testing (self, pathFileTest, pathModel, nnArchitecture, nnInChanCount, 
                nnClassCount, nnIsTrained, trBatchSize, loss_weights, 
                launchTimeStamp, vein_loss = False, cropped_fldr = None, 
                bounding_box_folder = None):   
        
        cudnn.benchmark = True
        
        model = model.load_model(nnArchitecture, nnIsTrained, 
                                nnInChanCount, nnClassCount)
        
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: IMPORT DATA
        # Import Main Data
        data = np.load(pathFileTest + 'train_test_data_without_augmetation.npz') 
        X_test_names = data['X_test_names'].astype(str)
        y_test = data['y_test']
        
        ID = dataset_builder.get_ID(X_test_names)

        data = []
        for index in range(0, len(X_test_names)):
            data.append([X_test_names[index], ID[index], y_test[index, 0], y_test[index, 1],
                        y_test[index, 2], y_test[index, 3]])

        test_data = pd.DataFrame(data, columns=['file_name', 'id', 'point_1x', 'point_1y',
                                                'point_2x', 'point_2y']) 

        #-------------------- SETTINGS: DATASET BUILDERS
        trans = transforms.Compose([transforms.ToTensor()])
        test_set = dataset_builder.SeedlingDataset(test_data, pathFileTest, 
                                    transform = trans, normalize = True)

        test_loader = DataLoader(test_set, batch_size = trBatchSize, shuffle = True)
        loss_class = utils.Cal_loss(loss_type = 'mae')
        model.eval() 
        # torch.cuda.empty_cache()
        print('-' * 50 + 'Start Testing' + '-' * 50)
        valid_loss = 0
        vein_loss = 0
        with torch.no_grad():      
            for i, (input, target) in enumerate(test_loader):
                
                if(self.gpu):
                    input = input.cuda()
                    ids = target[:, 0].cuda()
                    target = target[:, 1:].cuda()
                    varInput = torch.autograd.Variable(input)
                    varTarget = torch.autograd.Variable(target)
                    varOutput = model(varInput)
                    valid_loss += loss_class.calculate(varTarget, varOutput)
                    if(vein_loss):
                        vein_loss_class = utils.Vein_loss_class(varTarget, varOutput, 
                                                                varInput, cropped_fldr, 
                                                                bounding_box_folder, ids)
                        vein_img = vein_loss_class.get_vein_img()
                        valid_loss += vein_loss_class.cal_vein_loss()
        print('-' * 50 + 'Finished Testing' + '-' * 50)
        print('-' * 100)
        return valid_loss


if __name__ == "__main__":

    #-------------------- Parse the arguments
    parser = argparse.ArgumentParser(description='Training VeinNetTrainer...')
    parser.add_argument("--pathDirData", type=str, metavar = "",
                        help="Train/Test/Validation Data Directory")
    parser.add_argument("--Output_dir", type=str, metavar = "",
                        help="Output Folder Directory")
    parser.add_argument("--pathModel", type=str, metavar = "",
                        help="Test model Directory")
    parser.add_argument("--nnArchitecture", type=str, metavar = "",
                        help="Name of Model Architechture")
    parser.add_argument("--nnInChanCount", type=int, metavar = "",
                        help="Number of Input channel")
    parser.add_argument("--nnClassCount", type=int, metavar = "",
                        help="Number of predicted values")
    parser.add_argument("--trBatchSize", type=int, metavar = "",
                        help="Batch Size")
    parser.add_argument("--trMaxEpoch", type=int, metavar = "",
                        help="Number of epochs for training")
    parser.add_argument("--loss_weights", type=float, metavar = "",
                        help="Weights of different losses (MSE, Vein)")
    parser.add_argument("-gpu", "--gpu", action = "store_true",
                        help="Use GPU or not")
    parser.add_argument("-tr", "--train_mode", action = "store_true",
                        help="Activate Train Mode")
    parser.add_argument("-ts", "--test_mode", action = "store_true",
                        help="Activate Test Mode")
    parser.add_argument("-td", "--nnIsTrained", action = "store_true",
                        help="Use Trained Network or not")
    parser.add_argument("-vl", "--vein_loss", action = "store_true",
                        help="Calculate Vein loss or not")
    parser.add_argument("-ch", "--checkpoint", action = "store_true",
                        help="Check Points or not")
    args = parser.parse_args()
    
    #-------------------- Store the arguements
    if(args.nnArchitecture):
        nnArchitecture = args.nnArchitecture
    else:
        nnArchitecture = "resnet18"
    if args.trMaxEpoch:
        trMaxEpoch = args.trMaxEpoch
    else:
        trMaxEpoch = 10
    if args.trBatchSize:
        trBatchSize = args.trBatchSize
    else:
        trBatchSize = 8
    if args.pathDirData:
        pathDirData = args.pathDirData
    else:
        pathDirData = "./Data/Full X_train_data/"
    if args.Output_dir:
        Output_dir = args.Output_dir
    else:
        Output_dir = "./Model_Output/"
    if args.pathModel:
        test_pathModel = args.pathModel
    else:
        test_pathModel = None
    if args.nnInChanCount:
        nnInChanCount = args.nnInChanCount
    else:
        nnInChanCount = 3
    if args.nnClassCount:
        nnClassCount = args.nnClassCount
    else:
        nnClassCount = 4
    if args.loss_weights:
        loss_weights = args.loss_weights
    else:
        loss_weights = [0.2, 0.8]
    if(args.gpu):
        gpu = True
    else:
        gpu = False
    if(args.nnIsTrained):
        nnIsTrained = True
    else:
        nnIsTrained = False
    if(args.vein_loss):
        vein_loss = True
    else:
        vein_loss = False
    if(args.checkpoint):
        checkpoint = True
    else:
        checkpoint = False


    cropped_fldr = Output_dir + 'Cropped/'
    bounding_box_folder = Output_dir + 'Prediction/'
    print('-' * 100)
    print ('Training NN architecture = ', nnArchitecture)
    vein = VeinNetTrainer(args.gpu)
    if(args.train_mode):
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        timestampLaunch = timestampDate + '-' + timestampTime
        pathModel = Output_dir + 'm_' + timestampLaunch +  '.pth.tar'
        vein.training(pathDirData, pathModel, nnArchitecture, args.nnIsTrained, 
                    nnInChanCount, nnClassCount, trBatchSize, 
                    trMaxEpoch, loss_weights, timestampLaunch, args.checkpoint,
                    args.vein_loss, cropped_fldr, bounding_box_folder)
    if(args.test_mode):
        vein.testing(pathDirData, test_pathModel, nnArchitecture, 
                    nnInChanCount, nnClassCount, nnIsTrained, 
                    trBatchSize, loss_weights, vein_loss, 
                    cropped_fldr, bounding_box_folder)
    
    print('-' * 100)
    print ('Finished')
