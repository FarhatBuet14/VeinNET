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

import utils, losses, veinloss, model

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

    def __init__(self, gpu = True):
        self.gpu = gpu
        torch.cuda.empty_cache()
    
    ######################### Epoch Train #########################
    ###############################################################
    
    def epochTrain (self, model, dataLoader, optimizer, scheduler, trBatchSize,
                    epochMax, classCount, loss_class, loss_weights, vein_loss = False,
                    vein_loss_class = None, cropped_fldr = None, 
                    bounding_box_folder = None, data_folder = None):
        
        model.train()
        runningLoss = torch.tensor(0).to(dtype = torch.float32, device = torch.device('cuda'))
        runningLoss_v = torch.tensor(0).to(dtype = torch.float32, device = torch.device('cuda'))
        runningTotalLoss = torch.tensor(0).to(dtype = torch.float32, device = torch.device('cuda'))

        loader = tqdm(dataLoader, total=len(dataLoader))
        for batchID, (input, target, img_name) in enumerate (loader):
            
            id = target[:, 0]
            target = target[:, 1:]
            input, target = Variable(input), Variable(target)
            if(self.gpu):
                input = input.type(torch.FloatTensor).to(device = torch.device('cuda'))
                target = target.float().to(device= torch.device('cuda'))
            else:
                input = input.type(torch.DoubleTensor)
                target = target.float()
            
            output = model(input)
            # loss = func.mse_loss(output, target).type(torch.FloatTensor)
            loss = loss_class(target, output, input, img_name, id,
                            vein_loss, vein_loss_class, 
                            loss_weights).type(torch.FloatTensor)
            
            runningLoss += loss_class.point_loss_value
            runningLoss_v += loss_class.vein_loss_value
            runningTotalLoss += loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        runningLoss = runningLoss.item()/len(dataLoader)
        runningLoss_v = runningLoss_v.item()/len(dataLoader)
        runningTotalLoss = runningTotalLoss.item()/len(dataLoader)

        return runningTotalLoss, runningLoss, runningLoss_v

    ######################### Epoch Validation #########################
    ####################################################################
    
    def epochVal (self, model, dataLoader, optimizer, scheduler, trBatchSize,
                    epochMax, classCount, loss_class, loss_weights, vein_loss = False,
                    vein_loss_class = None, cropped_fldr = None, 
                    bounding_box_folder = None, data_folder = None):
        
        model.eval ()
        runningLoss = 0
        runningLoss_v = 0
        with torch.no_grad():
            for batchID, (input, target, img_name) in enumerate (dataLoader):
                
                id = target[:, 0]
                target = target[:, 1:]
                if(self.gpu):
                    input = input.type(torch.FloatTensor).to(device = torch.device('cuda'))
                    target = target.float().to(device = torch.device('cuda'))
                else:
                    input = input.type(torch.DoubleTensor),
                    target = target.float()
                
                output = model(input)
                # loss = func.mse_loss(output, target)
                loss = loss_class(target, output, input, img_name, id,
                                vein_loss, vein_loss_class, 
                                loss_weights).type(torch.FloatTensor)
            
                # Calculate loss explicitly
                loss_p = loss_class.point_loss_value
                loss_v = loss_class.vein_loss_value

                runningLoss += loss_p
                runningLoss_v += loss_v
        
            runningLoss = runningLoss/len(dataLoader)
            runningLoss_v = runningLoss_v/len(dataLoader)
            
        return loss, runningLoss, runningLoss_v
    
    ######################### Train Function #########################
    ##################################################################
    
    def training (self, pathDirData, pathModel, nnArchitecture, 
                nnIsTrained, nnInChanCount, nnClassCount,
                trBatchSize, trMaxEpoch, loss_weights,
                checkpoint = None, vein_loss = False, 
                cropped_fldr = None, bounding_box_folder = None):
        
        training_model = model.load_model(nnArchitecture, nnIsTrained, 
                                        nnInChanCount, nnClassCount, self.gpu)

        #-------------------- SETTINGS: DATASET BUILDERS
        trans_pipeline = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        train_data, valid_data = utils.dataset_builder(pathDirData, 'Train')
        train_set = utils.SeedlingDataset(train_data, pathDirData, 
                                    trans_pipeline = trans_pipeline, normalize = False)
        val_set = utils.SeedlingDataset(valid_data, pathDirData, 
                                trans_pipeline = trans_pipeline, normalize = False)

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
        vein_loss_class = veinloss.Vein_loss_class(cropped_fldr, bounding_box_folder, pathDirData)
  
        #---- TRAIN THE NETWORK
        lossMIN = 100000
        print('-' * 50 + 'Start Training' + '-' * 50)

        for epochID in range (0, trMaxEpoch):

            totalLossTrain, lossTrain, lossTrain_v = self.epochTrain (training_model, train_loader, optimizer, 
                                    scheduler, trBatchSize, trMaxEpoch, nnClassCount, 
                                    loss_class, loss_weights, vein_loss, vein_loss_class,
                                    cropped_fldr, bounding_box_folder, pathDirData)
            
            totalLossVal, lossVal, lossVal_v = self.epochVal (training_model, valid_loader, optimizer, 
                                            scheduler, trBatchSize, trMaxEpoch, 
                                            nnClassCount, loss_class, loss_weights, 
                                            vein_loss, vein_loss_class, cropped_fldr, 
                                            bounding_box_folder, pathDirData)
            
            scheduler.step(totalLossVal.data)
            
            # Save the minimum validation point data
            if totalLossVal < lossMIN:
                lossMIN = totalLossVal
                path = pathModel + '_ltr_' + str(totalLossTrain) + '_lvl_' + str(totalLossVal.item()) + '.pth.tar'
                torch.save({'epoch': epochID + 1, 'state_dict': training_model.state_dict(), 
                            'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, path)
                print ('Epoch [' + str(epochID + 1) + '] [save]')
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----]')
            
            # Print the losses
            print('Train_loss = ' + str(totalLossTrain))
            print('Val_loss   = ' + str(np.array(totalLossVal.cpu())))
            print('-' * 20)
            if(vein_loss):
                print('Train_loss_point = ' + str(lossTrain))
                print('Val_loss_point   = ' + str(lossVal.item()))
                print('-' * 20)
                print('Train_loss_vein = ' + str(lossTrain_v))
                print('Val_loss_vein   = ' + str(lossVal_v.item()))
        
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
        nnArchitecture = "resnet18"
    if args.trMaxEpoch:
        trMaxEpoch = args.trMaxEpoch
    else:
        trMaxEpoch = 100
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
        gpu = args.gpu
    else:
        gpu = True
    if(args.nnIsTrained):
        nnIsTrained = True
    else:
        nnIsTrained = False
    if(args.vein_loss):
        vein_loss = args.vein_loss
    else:
        vein_loss = True
    if(args.checkpoint):
        checkpoint = True
    else:
        checkpoint = False


    cropped_fldr = Output_dir + 'Cropped/'
    bounding_box_folder = Output_dir + 'Prediction/'
    print('-' * 100)
    print ('Training NN architecture = ', nnArchitecture)
    pathModel = Output_dir
    
    vein = VeinNetTrainer(gpu)

    vein.training(pathDirData, pathModel, nnArchitecture, args.nnIsTrained, 
                nnInChanCount, nnClassCount, trBatchSize, 
                trMaxEpoch, loss_weights, checkpoint,
                vein_loss, cropped_fldr, bounding_box_folder)
