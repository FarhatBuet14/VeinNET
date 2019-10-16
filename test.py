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

import utils, losses, veinloss , model

#######################  Define VeinBetTrainer Class  #########################

class VeinNetTester():

    #---- Test the VeinNet network 
    #---- pathFileTest - path to the directory that contains test images
    #---- nnArchitecture - model architecture 'resnet50', 'resnet34' or 'resnet18'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- SeedlingDataset - Dataset Generator 
    
    ######################### Initialization #########################
    ##################################################################

    def __init__(self, gpu = False):
        self.gpu = gpu
        torch.cuda.empty_cache()
    
    ########################## Test Function #########################
    ##################################################################
    
    def testing(self, pathFileTest, pathModel, Output_dir, nnArchitecture, 
                nnInChanCount, nnClassCount, nnIsTrained, 
                trBatchSize, loss_weights, vein_loss,
                cropped_fldr, bounding_box_folder):
        
        cudnn.benchmark = True

        #-------------------- SETTINGS: LOAD DATA
        test_model = model.load_model(nnArchitecture, nnIsTrained, 
                            nnInChanCount, nnClassCount)
        
        modelCheckpoint = torch.load(pathModel)
        test_model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: DATASET BUILDERS
        trans_pipeline = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_data = utils.dataset_builder(pathFileTest, 'Test')
        test_set = utils.SeedlingDataset(test_data, pathFileTest,
                                    trans_pipeline = trans_pipeline, normalize = True)
        test_loader = DataLoader(test_set, batch_size = trBatchSize, shuffle = True)

        loss_class = losses.Cal_loss(loss_type = 'mae')
        vein_loss_class = veinloss.Vein_loss_class(cropped_fldr, bounding_box_folder, pathFileTest)
        test_model.eval() 
        
        #-------------------- TESTING BEIGINS
        print('-' * 50 + 'Start Testing' + '-' * 50)
        print('-' * 113)
        
        runningLoss = 0
        runningLoss_v = 0
        runningTotalLoss = 0
        loss_logger = []
        names = []
        batch_loss_logger = []
        with torch.no_grad():
            
            loader = tqdm(test_loader, total=len(test_loader))
            for batchID, (input, target, img_name) in enumerate (loader):
                batch_loss = []
                id = target[:, 0:2]
                target = target[:, 2:]
                if(self.gpu):
                    input = input.type(torch.FloatTensor).to(device = torch.device('cuda'))
                    target = target.float().to(device = torch.device('cuda'))
                else:
                    input = input.type(torch.DoubleTensor),
                    target = target.float()
                
                output = test_model(input)
                # loss = func.mse_loss(output, target)
                loss = loss_class(target, output, input, img_name, id,
                                vein_loss, vein_loss_class, 
                                loss_weights).type(torch.FloatTensor)

                # Loss Logger
                loss_logger.append(loss_class.loss_logger)
                names.append(loss_class.names)
                batch_loss.append(loss_class.point_loss_value)
                batch_loss.append(loss_class.vein_loss_value)
                batch_loss.append(loss)
                batch_loss_logger.append(batch_loss)
            
                runningLoss += loss_class.point_loss_value
                runningLoss_v += loss_class.vein_loss_value
                runningTotalLoss += loss
        
            runningLoss = runningLoss/len(test_loader)
            runningLoss_v = runningLoss_v/len(test_loader)
            runningTotalLoss = runningTotalLoss/len(test_loader)

        # Print the losses
        print('Test_loss      = ' + str(np.array(runningTotalLoss.cpu())))
        print('-' * 20)
        if(vein_loss):
            print('Test_loss_point   = ' + str(runningLoss.item()))
            print('-' * 20)
            print('Test_loss_vein   = ' + str(runningLoss_v.item()))

        np.savez(Output_dir + "Loss_logger_test.npz",
                loss_logger = loss_logger,
                names = names,
                batch_loss_logger = batch_loss_logger)

        print('-' * 50 + 'Finished Testing' + '-' * 50)
        print('-' * 113)


######################### Main Function #########################
#################################################################

if __name__ == "__main__":

    #-------------------- Parse the arguments
    parser = argparse.ArgumentParser(description='Training VeinNetTrainer...')
    parser.add_argument("-ts", "--testSet", type=str, metavar = "",
                        help="Test Set Name")
    parser.add_argument("--pathDirData", type=str, metavar = "",
                        help="Train/Test/Validation Data Directory")
    parser.add_argument("--Output_dir", type=str, metavar = "",
                        help="Output Folder Directory")
    parser.add_argument('-pm', "--pathModel", type=str, metavar = "",
                        help="Test model Directory")
    parser.add_argument("--nnArchitecture", type=str, metavar = "",
                        help="Name of Model Architechture")
    parser.add_argument("--nnInChanCount", type=int, metavar = "",
                        help="Number of Input channel")
    parser.add_argument("--nnClassCount", type=int, metavar = "",
                        help="Number of predicted values")
    parser.add_argument("--trBatchSize", type=int, metavar = "",
                        help="Batch Size")
    parser.add_argument("--loss_weights", type=float, metavar = "",
                        help="Weights of different losses (MSE, Vein)")
    parser.add_argument("-gpu", "--gpu", action = "store_true",
                        help="Use GPU or not")
    parser.add_argument("-td", "--nnIsTrained", action = "store_true",
                        help="Use Trained Network or not")
    parser.add_argument("-vl", "--vein_loss", action = "store_true",
                        help="Calculate Vein loss or not")
    args = parser.parse_args()
    
    #-------------------- Store the arguements
    if(args.nnArchitecture):
        nnArchitecture = args.nnArchitecture
    else:
        nnArchitecture = "resnet50"
    if args.pathDirData:
        pathFileTest = args.pathDirData
    else:
        pathFileTest = "./Data/All/Test/"
    if args.Output_dir:
        Output_dir = args.Output_dir
    else:
        Output_dir = "./Model_Output/"
    if args.pathModel:
        test_pathModel = args.pathModel
    else:
        test_pathModel = "59_____1.7215716044108074_____3.3250184059143066_____16.050174967447916_____15.23857307434082.pth.tar"
    test_pathModel = Output_dir + test_pathModel
    if args.nnInChanCount:
        nnInChanCount = args.nnInChanCount
    else:
        nnInChanCount = 3
    if args.nnClassCount:
        nnClassCount = args.nnClassCount
    else:
        nnClassCount = 4
    if args.trBatchSize:
        trBatchSize = args.trBatchSize
    else:
        trBatchSize = 32
    if args.loss_weights:
        loss_weights = args.loss_weights
    else:
        loss_weights = [1, 1]
    if(args.gpu):
        gpu = args.gpu
    else:
        gpu = True
    if(args.vein_loss):
        vein_loss = args.vein_loss
    else:
        vein_loss = True
    if(args.nnIsTrained): 
        nnIsTrained = args.nnIsTrained
    else:
        nnIsTrained = False

    cropped_fldr = Output_dir + 'Cropped/Test/'
    bounding_box_folder = Output_dir + 'Prediction/Test/'
    print('-' * 100)
    print ('Training NN architecture = ', nnArchitecture)

    vein = VeinNetTester(gpu)
    vein.testing(pathFileTest, test_pathModel, Output_dir, nnArchitecture, 
                nnInChanCount, nnClassCount, nnIsTrained, 
                trBatchSize, loss_weights, vein_loss, 
                cropped_fldr, bounding_box_folder)
