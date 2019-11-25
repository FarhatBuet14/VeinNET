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

import utils, losses, veinloss , models

#######################  Define VeinBetTrainer Class  #########################

class VeinNetTester():
    
    ######################### Initialization #########################
    ##################################################################

    def __init__(self, opt):
        self.gpu = opt.gpu
        self.opt = opt
        torch.cuda.empty_cache()
    
    ########################## Test Function #########################
    ##################################################################
    
    def testing(self):
        
        cudnn.benchmark = True

        #-------------------- SETTINGS: LOAD DATA
        test_model = models.resnet50_DANN(self.opt)
        
        modelCheckpoint = torch.load(opt.checkpoint)
        test_model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: DATASET BUILDERS
        trans_pipeline = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_data= utils.dataset_builder(self.opt.pathDirData, 'Test')
        test_set = utils.SeedlingDataset(test_data, self.opt.pathDirData,
                                    trans_pipeline = trans_pipeline, normalize = True)
        test_loader = DataLoader(test_set, batch_size = self.opt.trBatchSize, shuffle = True)

        loss_class = losses.Cal_loss(loss_type = 'mae')
        vein_loss_class = veinloss.Vein_loss_class(self.opt)
        loss_domain = "bce"
        
        loss_class = loss_class.cuda()
        vein_loss_class = vein_loss_class.cuda()

        test_model.eval() 
        
        #-------------------- TESTING BEIGINS
        print('-' * 50 + 'Start Testing' + '-' * 50)
        print('-' * 113)
        
        runningLoss = 0
        runningLoss_v = 0
        runningLoss_d = 0
        runningTotalLoss = 0
        loss_logger = []
        names = []
        batch_loss_logger = []
        with torch.no_grad():
            
            loader = tqdm(test_loader, total=len(test_loader))
            for batchID, (input, target, img_name) in enumerate (loader):
                batch_loss = []
                id = target[:, 0:1]
                org = target[:, 1:2]
                target = target[:, 2:]
                if(self.gpu):
                    input = input.type(torch.FloatTensor).to(device = torch.device('cuda'))
                    target = target.float().to(device = torch.device('cuda'))
                    id = id.float().to(device = torch.device('cuda'))
                    org = org.float().to(device = torch.device('cuda'))
                else:
                    input = input.type(torch.DoubleTensor),
                    target = target.float()
                    id = id.float()
                    org = org.float()
                
                # ---------- Calculate alpha
                # p = float(batchID + epoch * len(dataLoader)) / self.opt.trMaxEpoch / len(dataLoader)
                # alpha = 2. / (1. + np.exp(-10 * p)) - 1

                output, domain_output = test_model(input, alpha = 0)
                loss = loss_class(target, output, domain_output, input, img_name, id, org,
                                vein_loss_class, self.opt.loss_weights, loss_domain)

                # Loss Logger
                loss_logger.append(loss_class.loss_logger)
                names.append(loss_class.names)
                batch_loss.append(loss_class.point_loss_value)
                batch_loss.append(loss_class.vein_loss_value)
                batch_loss.append(loss_class.domain_loss)
                batch_loss.append(loss)
                batch_loss_logger.append(batch_loss)
            
                runningLoss += loss_class.point_loss_value
                runningLoss_v += loss_class.vein_loss_value
                runningLoss_d += loss_class.domain_loss
                runningTotalLoss += loss
        
            runningLoss = runningLoss/len(test_loader)
            runningLoss_v = runningLoss_v/len(test_loader)
            runningLoss_d = runningLoss_d/len(test_loader)
            runningTotalLoss = runningTotalLoss/len(test_loader)

        # Print the losses
        print('Test_loss      = ' + str(np.array(runningTotalLoss.cpu())))
        print('-' * 20)
        print('Test_loss_point   = ' + str(runningLoss.item()))
        print('-' * 20)
        print('Test_loss_vein   = ' + str(runningLoss_v.item()))
        print('-' * 20)
        print('Test_loss_domain   = ' + str(runningLoss_d.item()))

        np.savez(opt.Output_dir + "Loss_logger_test.npz",
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
    
    parser.add_argument("--pathDirData", type=str, metavar = "", help="Train/Test/Validation Data Directory", default = "./Data/Test/")
    parser.add_argument("--Output_dir", type=str, metavar = "", help="Output Folder Directory", default = "./Model_Output/")
    parser.add_argument("--cropped_fldr", type=str, metavar = "", help="Test model Directory", default = "./Model_Output/Cropped/Test/")
    parser.add_argument("--bounding_box_folder", type=str, metavar = "", help="Test model Directory", default = "./Model_Output/Prediction/Test/")
    parser.add_argument("--image_size", type = int, default = [240, 300])

    parser.add_argument("--nnArchitecture", type=str, metavar = "", help="Name of Model Architechture", default = "resnet50")
    parser.add_argument("--nnIsTrained", action = "store_true", help="Use Trained Network or not", default = True)
    parser.add_argument("--nnInChanCount", type=int, metavar = "", help="Number of Input channel", default = 3)
    parser.add_argument("--nnClassCount", type=int, metavar = "", help="Number of predicted values", default = 4)

    parser.add_argument("--trBatchSize", type=int, metavar = "", help="Batch Size", default = 32)
    parser.add_argument("--trMaxEpoch", type=int, metavar = "", help="Number of epochs for training", default = 800)
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--loss_weights", type=float, metavar = "", help="Weights of different losses (MSE, Vein, Domain)", default = [1, 0, 1])
    
    parser.add_argument("--gpu", action = "store_true", help="Use GPU or not", default = True)
    parser.add_argument("--checkpoint", type=str, metavar = "", help="Checkpoint File", 
    default = './Model_Output/' + '556_____3.3099144345238094_____17.011796951293945_____3.720998709542411_____10.303921699523926_____0.0036639082999456495_____0.00019180364324711263.pth.tar')
    
    opt = parser.parse_args()
    print(opt)
    
    # ------------------------ Start Training
    
    print('-' * 113)
    print ('Testing NN architecture = ', opt.nnArchitecture)

    vein = VeinNetTester(opt)
    vein.testing()
