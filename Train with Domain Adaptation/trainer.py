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
import torchvision
from torchvision import transforms 
import torch.nn.functional as func
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.utils.tensorboard import SummaryWriter

import models, utils, losses, veinloss

#######################  Define VeinBetTrainer Class  #########################

class VeinNetTrainer():

    ######################### Initialization #########################
    ##################################################################

    def __init__(self, opt):
        self.gpu = opt.gpu
        self.opt = opt
        torch.cuda.empty_cache()
    
    ######################### Epoch Train #########################
    ###############################################################
    
    def epochTrain(self, model, dataLoader, optimizer, 
                    scheduler, loss_class, epoch, vein_loss_class, loss_domain):
        
        model.train()
        runningLoss = torch.tensor(0).to(dtype = torch.float32, device = torch.device('cuda'))
        runningLoss_v = torch.tensor(0).to(dtype = torch.float32, device = torch.device('cuda'))
        runningLoss_d = torch.tensor(0).to(dtype = torch.float32, device = torch.device('cuda'))
        runningTotalLoss = torch.tensor(0).to(dtype = torch.float32, device = torch.device('cuda'))

        loader = tqdm(dataLoader, total=len(dataLoader))
        for batchID, (input, target, img_name) in enumerate (loader):
            
            id = target[:, 0:1]
            org = target[:, 1:2]
            target = target[:, 2:]
            input, target = Variable(input), Variable(target)
            if(self.gpu):
                input = input.type(torch.FloatTensor).to(device = torch.device('cuda'))
                target = target.float().to(device= torch.device('cuda'))
                id = id.float().to(device= torch.device('cuda'))
                org = org.float().to(device= torch.device('cuda'))
            else:
                input = input.type(torch.DoubleTensor)
                target = target.float()
                id = id.float()
                org = org.float()
            
            p = float(batchID + epoch * len(dataLoader)) / self.opt.trMaxEpoch / len(dataLoader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            output, domain_output = model(input, alpha)
            loss = loss_class(target, output, domain_output, input, img_name, id, org, 
                            vein_loss_class, self.opt.loss_weights, loss_domain)
            
            runningLoss += loss_class.point_loss_value
            runningLoss_v += loss_class.vein_loss_value
            runningLoss_d += loss_class.domain_loss
            runningTotalLoss += loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        runningLoss = runningLoss.item()/len(dataLoader)
        runningLoss_v = runningLoss_v.item()/len(dataLoader)
        runningLoss_d = runningLoss_d.item()/len(dataLoader)
        runningTotalLoss = runningTotalLoss.item()/len(dataLoader)

        return runningTotalLoss, runningLoss, runningLoss_v, runningLoss_d

    ######################### Epoch Validation #########################
    ####################################################################
    
    def epochVal (self, model, dataLoader, optimizer, 
                scheduler, loss_class, epoch, vein_loss_class, loss_domain):
        
        model.eval ()
        runningLoss = 0
        runningLoss_v = 0
        runningLoss_d = 0
        runningTotalLoss = 0
        with torch.no_grad():
            for batchID, (input, target, img_name) in enumerate (dataLoader):
                
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
                
                p = float(batchID + epoch * len(dataLoader)) / self.opt.trMaxEpoch / len(dataLoader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                output, domain_output = model(input, alpha)
                loss = loss_class(target, output, domain_output, input, img_name, id, org,
                                vein_loss_class, self.opt.loss_weights, loss_domain)

                runningLoss += loss_class.point_loss_value
                runningLoss_v += loss_class.vein_loss_value
                runningLoss_d += loss_class.domain_loss
                runningTotalLoss += loss
            
            runningLoss = runningLoss/len(dataLoader)
            runningLoss_v = runningLoss_v/len(dataLoader)
            runningLoss_d = runningLoss_d/len(dataLoader)
            runningTotalLoss = runningTotalLoss/len(dataLoader)
            
        return runningTotalLoss, runningLoss, runningLoss_v, runningLoss_d
    
    ######################### Train Function #########################
    ##################################################################
    
    def training (self):
        
        training_model = models.resnet50_DANN(self.opt)
        for p in training_model.parameters():
            p.requires_grad = True

        #-------------------- SETTINGS: DATASET BUILDERS
        trans_pipeline = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        train_data, valid_data = utils.dataset_builder(self.opt.pathDirData, 'Train')
        train_set = utils.SeedlingDataset(train_data, self.opt.pathDirData, 
                                    trans_pipeline = trans_pipeline, normalize = False)
        val_set = utils.SeedlingDataset(valid_data, self.opt.pathDirData, 
                                trans_pipeline = trans_pipeline, normalize = False)

        train_loader = DataLoader(train_set, batch_size = self.opt.trBatchSize, shuffle = True)
        valid_loader = DataLoader(val_set, batch_size = self.opt.trBatchSize, shuffle = True)
        
        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(training_model.parameters(), lr = self.opt.lr, betas = (0.9, 0.999), 
                                eps = 1e-08, weight_decay = 1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, 
                                                    patience = 5, verbose = True, threshold = 0.0001, 
                                                    threshold_mode = 'rel', cooldown = 0, min_lr = 0, 
                                                    eps = 1e-08)
                
        #-------------------- SETTINGS: LOSS
        loss_class = losses.Cal_loss(loss_type = 'mae')
        vein_loss_class = veinloss.Vein_loss_class(self.opt)
        loss_domain = torch.nn.BCELoss()
        
        loss_class = loss_class.cuda()
        vein_loss_class = vein_loss_class.cuda()
        loss_domain = loss_domain.cuda()
  
        #-------------------- SETTINGS: TENSORBOARD
        tb = SummaryWriter()
        
        # images, _, _ = next(iter(train_loader))
        # images_val, _, _ = next(iter(valid_loader))
        # grid = torchvision.utils.make_grid(images)
        # grid_val = torchvision.utils.make_grid(images_val)

        # tb.add_image('images', grid)
        # tb.add_image('images_val', grid_val)
        # tb.add_graph(training_model.cpu(), images.reshape((1, trBatchSize, nnInChanCount, 240, 300)))
        # tb.add_graph(training_model.cpu(), images_val.reshape((1, trBatchSize, nnInChanCount, 240, 300)))
        
        #---- TRAIN THE NETWORK
        lossMIN = 100000
        start_epoch = 0
        # Load the checkpoint
        if(self.opt.checkpoint):
            training_model, optimizer, start_epoch, lossMIN = utils.load_checkpoint(training_model, optimizer, 
                                                                                    lossMIN, self.opt.checkpoint)
            training_model = training_model.cuda()
            
            # now individually transfer the optimizer parts...
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(torch.device('cuda'))
        
        print('-' * 50 + 'Start Training' + '-' * 50)

        loss_epoch = []
        self.loss_logger = []

        for epochID in range (start_epoch, self.opt.trMaxEpoch):

            totalLossTrain, lossTrain, lossTrain_v, lossTrain_d = self.epochTrain(training_model, train_loader, optimizer, 
                                                 scheduler, loss_class, epochID + 1, vein_loss_class, loss_domain)
            
            tb.add_scalar('Train_loss', totalLossTrain, epochID + 1)
            tb.add_scalar('Train_loss_point', lossTrain, epochID + 1)
            tb.add_scalar('Train_loss_vein', lossTrain_v, epochID + 1)
            tb.add_scalar('Train_loss_vein', lossTrain_d, epochID + 1)
            loss_epoch.append(totalLossTrain)
            loss_epoch.append(lossTrain)
            loss_epoch.append(lossTrain_v)
            loss_epoch.append(lossTrain_d)

            totalLossVal, lossVal, lossVal_v, lossVal_d = self.epochVal(training_model, valid_loader, optimizer, 
                                                scheduler, loss_class, epochID + 1, vein_loss_class, loss_domain)
            
            tb.add_scalar('Val_loss', totalLossVal, epochID + 1)
            tb.add_scalar('Val_loss_point', lossVal, epochID + 1)
            tb.add_scalar('Val_loss_vein', lossVal_v, epochID + 1)
            tb.add_scalar('Val_loss_vein', lossVal_d, epochID + 1)
            loss_epoch.append(totalLossVal)
            loss_epoch.append(lossVal)
            loss_epoch.append(lossVal_v)
            loss_epoch.append(lossVal_d)

            # tb.add_histogram('conv1.bias', training_model.conv1.bias, epochID + 1)
            # tb.add_histogram('conv1.weight', training_model.conv1.weight, epochID + 1)
            # tb.add_histogram('conv1.weight.grad', training_model.conv1.weight.grad, epochID + 1)
            
            self.loss_logger.append(loss_epoch)

            scheduler.step(totalLossVal.data)

            # Save the minimum validation point data
            if lossVal < lossMIN : 
                lossMIN = lossVal
                path = self.opt.Output_dir + str(epochID + 1) + '_____' + str(lossTrain) + '_____' + str(lossVal.item()) + '_____' + str(lossTrain_v) + '_____' + str(lossVal_v.item()) + '_____' + str(lossTrain_d) + '_____' + str(lossVal_d.item()) + '.pth.tar'
                torch.save({'epoch': epochID + 1, 'state_dict': training_model.state_dict(), 
                            'best_loss': lossMIN, 'optimizer' : optimizer.state_dict(), 
                            'loss_logger' : self.loss_logger }, path)
                print ('Epoch [' + str(epochID + 1) + '] [save]')
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----]')
            
            # Print the losses
            print('Train_loss = ' + str(totalLossTrain))
            print('Val_loss   = ' + str(np.array(totalLossVal.cpu())))
            print('-' * 20)
            print('Train_loss_point = ' + str(lossTrain))
            print('Val_loss_point   = ' + str(lossVal.item()))
            print('-' * 20)
            print('Train_loss_vein = ' + str(lossTrain_v))
            print('Val_loss_vein   = ' + str(lossVal_v.item()))
            print('-' * 20)
            print('Train_loss_domain = ' + str(lossTrain_d))
            print('Val_loss_domain   = ' + str(lossVal_d.item()))
            print('-' * 50)


            loss_epoch = []
        
        tb.close()
        np.savez(self.opt.Output_dir + 'loss_logger.npz', loss_logger = np.array(self.loss_logger))
        
        print('-' * 50 + 'Finished Training' + '-' * 50)


######################### Main Function #########################
#################################################################

if __name__ == "__main__":

    #-------------------- Parse the arguments
    parser = argparse.ArgumentParser(description='Training VeinNetTrainer...')
    
    parser.add_argument("--pathDirData", type=str, metavar = "", help="Train/Test/Validation Data Directory", default = "./Data/Train/")
    parser.add_argument("--Output_dir", type=str, metavar = "", help="Output Folder Directory", default = "./Model_Output/")
    parser.add_argument("--cropped_fldr", type=str, metavar = "", help="Test model Directory", default = "./Model_Output/Cropped/")
    parser.add_argument("--bounding_box_folder", type=str, metavar = "", help="Test model Directory", default = "./Model_Output/Prediction/")
    parser.add_argument("--image_size", type = int, default = [240, 300])

    parser.add_argument("--nnArchitecture", type=str, metavar = "", help="Name of Model Architechture", default = "resnet50")
    parser.add_argument("--nnIsTrained", action = "store_true", help="Use Trained Network or not", default = True)
    parser.add_argument("--nnInChanCount", type=int, metavar = "", help="Number of Input channel", default = 3)
    parser.add_argument("--nnClassCount", type=int, metavar = "", help="Number of predicted values", default = 4)

    parser.add_argument("--trBatchSize", type=int, metavar = "", help="Batch Size", default = 32)
    parser.add_argument("--trMaxEpoch", type=int, metavar = "", help="Number of epochs for training", default = 900)
    parser.add_argument("--lr", type = float, default = 0.001)
    parser.add_argument("--loss_weights", type=float, metavar = "", help="Weights of different losses (MSE, Vein)", default = [1, 0.25])
    
    parser.add_argument("--gpu", action = "store_true", help="Use GPU or not", default = True)
    parser.add_argument("--checkpoint", type=str, metavar = "", help="Checkpoint File", 
    default = './Model_Output/' + '35_____4.604841541108631_____6.364306449890137_____22.350406901041666_____18.335926055908203_____1.5730550130208334_____2.3439548015594482.pth.tar')

    opt = parser.parse_args()
    print(opt)
    
    vein = VeinNetTrainer(opt)

    vein.training()
