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

import utils, losses, model

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
    
    def testing(self, pathFileTest, pathModel, nnArchitecture, 
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
        trans = transforms.Compose([transforms.ToTensor()])
        test_data = utils.dataset_builder(pathFileTest, 'Test')
        test_set = utils.SeedlingDataset(test_data, pathFileTest,
                                    transform = trans, normalize = True)

        test_loader = DataLoader(test_set, batch_size = trBatchSize, shuffle = True)

        loss_class = losses.Cal_loss(loss_type = 'mse')
        test_model.eval() 
        
        #-------------------- TESTING BEIGINS
        print('-' * 50 + 'Start Testing' + '-' * 50)
        print('-' * 113)
        loss = 0
        loss_v = 0
        with torch.no_grad():
            loader = tqdm(test_loader, total=len(test_loader))
            for batchID, (input, target, img_name) in enumerate(loader):
                id = target[:, 0]
                target = target[:, 1:]
                varInput = torch.autograd.Variable(input.cuda())
                varOutput = test_model(varInput)
                output = (Variable(varOutput).data).cpu()

                # loss += func.mse_loss(output, target)
                loss += loss_class.calculate(target, output)
                if(vein_loss):
                    vein_loss_class = losses.Vein_loss_class(target, output, img_name,
                                                            input, cropped_fldr,
                                                            bounding_box_folder, 
                                                            pathFileTest, id)
                    _ = vein_loss_class.get_vein_img()
                    loss_v += vein_loss_class.cal_vein_loss()
            loss = loss / len(test_loader)
        # Print the losses
        print('Test_loss= ' + str(loss.data))
        if(vein_loss):
            print('Test_loss_vein= ' + str(loss_v))
        
        print('-' * 50 + 'Finished Testing' + '-' * 50)
        print('-' * 113)


######################### Main Function #########################
#################################################################

if __name__ == "__main__":

    #-------------------- Parse the arguments
    parser = argparse.ArgumentParser(description='Training VeinNetTrainer...')
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
        pathFileTest = "./Data/Test_data/"
    if args.Output_dir:
        Output_dir = args.Output_dir
    else:
        Output_dir = "./Model_Output/"
    if args.pathModel:
        test_pathModel = args.pathModel
    else:
        test_pathModel = "t_29072019-010056_ltr_tensor(20448.8496)_lvl_tensor(17930.2539).pth.tar"
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
        trBatchSize = 8
    if args.loss_weights:
        loss_weights = args.loss_weights
    else:
        loss_weights = [1, 1]
    if(args.gpu):
        gpu = True
    else:
        gpu = False
    if(args.vein_loss):
        vein_loss = True
    else:
        vein_loss = False
    if(args.nnIsTrained):
        nnIsTrained = True
    else:
        nnIsTrained = False

    cropped_fldr = Output_dir + 'Cropped/'
    bounding_box_folder = Output_dir + 'Prediction/'
    print('-' * 100)
    print ('Training NN architecture = ', nnArchitecture)

    vein = VeinNetTester(gpu)
    vein.testing(pathFileTest, test_pathModel, nnArchitecture, 
                nnInChanCount, nnClassCount, nnIsTrained, 
                trBatchSize, loss_weights, vein_loss, 
                cropped_fldr, bounding_box_folder)
   