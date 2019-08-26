############################  IMPORT LIBRARIES  ############################
import numpy as np
import time
from os.path import join
import argparse
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as func
import torch.optim as optim
from torch.optim import lr_scheduler

import utils, losses, model

#######################  Define VeinBetTrainer Class  #########################

class VeinNetTrainer():

    #---- Train the VeinNet network 
    #---- pathDirData - path to the directory that contains train images
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
        loss = 0
        loss_v = 0
        loader = tqdm(dataLoader, total=len(dataLoader))
        for batchID, (input, target, img_name) in enumerate (loader):
            
            id = target[:, 0]
            target = target[:, 1:]
            varInput = torch.autograd.Variable(input.cuda())
            varOutput = model(varInput)
            output = (Variable(varOutput).data).cpu()
            # loss = Variable(func.mse_loss(output, target), requires_grad = True)
            loss = loss_class.calculate(target, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

                # loss += func.mse_loss(output, target)
                loss += loss_class.calculate(target, output)
        
            loss = loss/len(dataLoader)
            
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
        trans = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    #                         std=[0.229, 0.224, 0.225])])
        train_data, valid_data = utils.dataset_builder(pathDirData, 'Train')
        train_set = utils.SeedlingDataset(train_data, pathDirData, 
                                    transform = trans, normalize = True)
        val_set = utils.SeedlingDataset(valid_data, pathDirData, 
                                transform = trans, normalize = True)

        train_loader = DataLoader(train_set, batch_size = trBatchSize, shuffle = True)
        valid_loader = DataLoader(val_set, batch_size = trBatchSize, shuffle = True)
        
        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(training_model.parameters(), lr=1e-3, betas=(0.9, 0.999), 
                                eps=1e-08, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                    patience=5, verbose=False, threshold=0.0001, 
                                                    threshold_mode='rel', cooldown=0, min_lr=0, 
                                                    eps=1e-08)
                
        #-------------------- SETTINGS: LOSS
        loss_class = losses.Cal_loss(loss_type = 'mae')
  
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
            if lossVal < lossMIN: # Save the minimum validation point data
                lossMIN = lossVal   
                path = pathModel + 't_' + timestampLaunch + '_ltr_' + str(lossTrain.data) + '_lvl_' + str(lossVal) + '.pth.tar'
                torch.save({'epoch': epochID + 1, 'state_dict': training_model.state_dict(), 
                            'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, path)
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + ']')
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + ']')
            # Print the losses
            print('Train_loss= ' + str(lossTrain.data))
            print('Val_loss= ' + str(lossVal))
            if(vein_loss):
                print('Train_loss_vein= ' + str(lossTrain_v))
                print('Val_loss_vein= ' + str(lossVal_v))
        
        print('-' * 50 + 'Finished Training' + '-' * 50)


######################### Main Function #########################
#################################################################

if __name__ == "__main__":

    #-------------------- Parse the arguments
    parser = argparse.ArgumentParser(description='Training VeinNetTrainer...')
    parser.add_argument("--pathDirData", type=str, metavar = "",
                        help="Train/Test/Validation Data Directory")
    parser.add_argument("--Output_dir", type=str, metavar = "",
                        help="Output Folder Directory")
    parser.add_argument("--pathModel", type=str, metavar = "",
                        help="Test model Directory")
    parser.add_argument("-arc", "--nnArchitecture", type=str, metavar = "",
                        help="Name of Model Architechture")
    parser.add_argument("--nnInChanCount", type=int, metavar = "",
                        help="Number of Input channel")
    parser.add_argument("--nnClassCount", type=int, metavar = "",
                        help="Number of predicted values")
    parser.add_argument("--trBatchSize", type=int, metavar = "",
                        help="Batch Size")
    parser.add_argument("-ep", "--trMaxEpoch", type=int, metavar = "",
                        help="Number of epochs for training")
    parser.add_argument("--loss_weights", type=float, metavar = "",
                        help="Weights of different losses (MSE, Vein)")
    parser.add_argument("-gpu", "--gpu", action = "store_true",
                        help="Use GPU or not")
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
        nnArchitecture = "DENSE-NET-201"
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
        loss_weights = [1, 1]
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
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    pathModel = Output_dir
    
    vein = VeinNetTrainer(gpu)

    vein.training(pathDirData, pathModel, nnArchitecture, args.nnIsTrained, 
                nnInChanCount, nnClassCount, trBatchSize, 
                trMaxEpoch, loss_weights, timestampLaunch, args.checkpoint,
                args.vein_loss, cropped_fldr, bounding_box_folder)
