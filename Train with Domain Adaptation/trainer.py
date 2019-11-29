############################  IMPORT LIBRARIES  ############################
import numpy as np
import argparse
from tqdm import tqdm
import random

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms 
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
    
    def epochTrain(self, model, dataloader_s, dataloader_t, optimizer, 
                    scheduler, loss_class, epoch, vein_loss_class, loss_domain):
        
        model.train()
        runningLoss_s = torch.tensor(0).to(dtype = torch.float32, device = torch.device('cuda'))
        runningLoss_v_s = torch.tensor(0).to(dtype = torch.float32, device = torch.device('cuda'))
        runningLoss_d_s = torch.tensor(0).to(dtype = torch.float32, device = torch.device('cuda'))
        runningTotalLoss_s = torch.tensor(0).to(dtype = torch.float32, device = torch.device('cuda'))

        runningLoss_t = torch.tensor(0).to(dtype = torch.float32, device = torch.device('cuda'))
        runningLoss_v_t = torch.tensor(0).to(dtype = torch.float32, device = torch.device('cuda'))
        runningLoss_d_t = torch.tensor(0).to(dtype = torch.float32, device = torch.device('cuda'))
        runningTotalLoss_t = torch.tensor(0).to(dtype = torch.float32, device = torch.device('cuda'))

        len_dataloader = min(len(dataloader_s), len(dataloader_t))
        data_source_iter = iter(dataloader_s)
        data_target_iter = iter(dataloader_t)
        
        batchID = 0
        while batchID < len_dataloader:
            p = float(batchID + epoch * len_dataloader) / self.opt.trMaxEpoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            loader = data_source_iter.next()
            input, target, img_name = loader
            
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
            
            model.zero_grad()
            output, domain_output = model(input, alpha)
            loss_s = loss_class("source", target, output, domain_output, input, img_name, id, org, 
                            vein_loss_class, self.opt.loss_weights, loss_domain)
            
            runningLoss_s += loss_class.point_loss_value
            runningLoss_v_s += loss_class.vein_loss_value
            runningLoss_d_s += loss_class.domain_loss
            runningTotalLoss_s += loss_s
            
            # training model using target data
            loader = data_target_iter.next()
            input, target, img_name = loader
            
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
            
            output, domain_output = model(input, alpha)
            loss_t = loss_class("target", target, output, domain_output, input, img_name, id, org, 
                            vein_loss_class, self.opt.loss_weights, loss_domain)
            
            runningLoss_t += loss_class.point_loss_value
            runningLoss_v_t += loss_class.vein_loss_value
            runningLoss_d_t += loss_class.domain_loss
            runningTotalLoss_t += loss_t

            # calculate total loss and backpropagate
            if((batchID + 1) % 5 == 0):
                print("batch-" + str(batchID + 1) + "--- Source point Loss - " + str(loss_s.item()) + " ---------- Target Loss - " + str(loss_t.item()))
            loss = loss_s + loss_t
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchID += 1
        
        runningLoss_s = runningLoss_s.item()/len_dataloader
        runningLoss_v_s = runningLoss_v_s.item()/len_dataloader
        runningLoss_d_s = runningLoss_d_s.item()/len_dataloader
        runningTotalLoss_s = runningTotalLoss_s.item()/len_dataloader
        
        runningLoss_t = runningLoss_t.item()/len_dataloader
        runningLoss_v_t = runningLoss_v_t.item()/len_dataloader
        runningLoss_d_t = runningLoss_d_t.item()/len_dataloader
        runningTotalLoss_t = runningTotalLoss_t.item()/len_dataloader

        return runningTotalLoss_s, runningLoss_s, runningLoss_v_s, runningLoss_d_s, runningTotalLoss_t, runningLoss_t, runningLoss_v_t, runningLoss_d_t

    ######################### Epoch Validation #########################
    ####################################################################
    
    def epochVal (self, model, method, dataLoader, optimizer, 
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
                
                # ---------- Calculate alpha
                p = float(batchID + epoch * len(dataLoader)) / self.opt.trMaxEpoch / len(dataLoader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                output, domain_output = model(input, alpha)
                loss = loss_class(method, target, output, domain_output, input, img_name, id, org,
                                vein_loss_class, self.opt.loss_weights, loss_domain)

                runningLoss += loss_class.point_loss_value
                runningLoss_v += loss_class.vein_loss_value
                runningLoss_d += loss_class.domain_loss
                runningTotalLoss += loss

                if((batchID + 1) % 10 == 0):
                    print("batch-" + str(batchID + 1) + "--- " + method + " Loss - " + str(loss.item()))
            
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
        
        #---------- Source Data
        train_data, valid_data = utils.dataset_builder(self.opt.pathDirData, 'Train', opt.source)
        train_set = utils.SeedlingDataset(train_data, self.opt.pathDirData, 
                                    trans_pipeline = trans_pipeline, normalize = False)
        val_set = utils.SeedlingDataset(valid_data, self.opt.pathDirData, 
                                trans_pipeline = trans_pipeline, normalize = False)

        train_loader_s = DataLoader(train_set, batch_size = self.opt.trBatchSize, shuffle = True)
        valid_loader_s = DataLoader(val_set, batch_size = self.opt.trBatchSize, shuffle = True)

        #---------- Target Data
        train_data, valid_data = utils.dataset_builder(self.opt.pathDirData, 'Train', opt.target)
        train_set = utils.SeedlingDataset(train_data, self.opt.pathDirData, 
                                    trans_pipeline = trans_pipeline, normalize = False)
        val_set = utils.SeedlingDataset(valid_data, self.opt.pathDirData, 
                                trans_pipeline = trans_pipeline, normalize = False)

        train_loader_t = DataLoader(train_set, batch_size = self.opt.trBatchSize, shuffle = True)
        valid_loader_t = DataLoader(val_set, batch_size = self.opt.trBatchSize, shuffle = True)

        
        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(training_model.parameters(), lr = self.opt.lr, betas = (0.9, 0.999), 
                                eps = 1e-08, weight_decay = 1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, 
                                                    patience = 5, verbose = True, threshold = 0.0001, 
                                                    threshold_mode = 'rel', cooldown = 0, min_lr = 1e-10, 
                                                    eps = 1e-08)
                
        #-------------------- SETTINGS: LOSS
        loss_class = losses.Cal_loss(loss_type = 'mae')
        vein_loss_class = veinloss.Vein_loss_class(self.opt)
        loss_domain = "bce"
        
        loss_class = loss_class.cuda()
        vein_loss_class = vein_loss_class.cuda()
  
        #-------------------- SETTINGS: TENSORBOARD
        # #-- Source
        # tb_s = SummaryWriter()
        # tb_val_s = SummaryWriter()
        # #-- Target
        # tb_t = SummaryWriter()
        # tb_val_t = SummaryWriter()
        
        #---- TRAIN THE NETWORK
        lossMIN = 100000000
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
            
            # ------------------------------------------ Train
            print("------ Training ------ Epoch - " + str(epochID + 1) + " ----" * 10)
            totalLossTrain_s, lossTrain_s, lossTrain_v_s, lossTrain_d_s, totalLossTrain, lossTrain, lossTrain_v, lossTrain_d = self.epochTrain(training_model,
                                                                    train_loader_s, train_loader_t,
                                                                    optimizer, scheduler, loss_class, epochID + 1, 
                                                                    vein_loss_class, loss_domain)
            
            # tb_s.add_scalar('Train_loss_s', totalLossTrain_s, epochID + 1)
            # tb_s.add_scalar('Train_loss_point_s', lossTrain_s, epochID + 1)
            # tb_s.add_scalar('Train_loss_vein_s', lossTrain_v_s, epochID + 1)
            # tb_s.add_scalar('Train_loss_domain_s', lossTrain_d_s, epochID + 1)
            loss_epoch.append(totalLossTrain_s)
            loss_epoch.append(lossTrain_s)
            loss_epoch.append(lossTrain_v_s)
            loss_epoch.append(lossTrain_d_s)

            
            # tb_t.add_scalar('Train_loss_t', totalLossTrain, epochID + 1)
            # tb_t.add_scalar('Train_loss_point_t', lossTrain, epochID + 1)
            # tb_t.add_scalar('Train_loss_vein_t', lossTrain_v, epochID + 1)
            # tb_t.add_scalar('Train_loss_domain_t', lossTrain_d, epochID + 1)
            loss_epoch.append(totalLossTrain)
            loss_epoch.append(lossTrain)
            loss_epoch.append(lossTrain_v)
            loss_epoch.append(lossTrain_d)

            # ------------------------------------------ Validation
            print("------ Validation ------ Epoch - " + str(epochID + 1) + " ----" * 10)
            # ----- Source Validation
            method = "source"
            totalLossVal_s, lossVal_s, lossVal_v_s, lossVal_d_s = self.epochVal(training_model, method, valid_loader_s, optimizer, 
                                                            scheduler, loss_class, epochID + 1, vein_loss_class, loss_domain)
            
            # tb_val_s.add_scalar('Val_loss_s', totalLossVal_s, epochID + 1)
            # tb_val_s.add_scalar('Val_loss_point_s', lossVal_s, epochID + 1)
            # tb_val_s.add_scalar('Val_loss_vein_s', lossVal_v_s, epochID + 1)
            # tb_val_s.add_scalar('Val_loss_domain_s', lossVal_d_s, epochID + 1)
            loss_epoch.append(totalLossVal_s)
            loss_epoch.append(lossVal_s)
            loss_epoch.append(lossVal_v_s)
            loss_epoch.append(lossVal_d_s)

            # ----- Target Validation
            print(' -- ' * 10)
            method = "target"
            totalLossVal, lossVal, lossVal_v, lossVal_d = self.epochVal(training_model, method, valid_loader_t, optimizer, 
                                                scheduler, loss_class, epochID + 1, vein_loss_class, loss_domain)
            
            # tb_val_t.add_scalar('Val_loss_t', totalLossVal, epochID + 1)
            # tb_val_t.add_scalar('Val_loss_point_t', lossVal, epochID + 1)
            # tb_val_t.add_scalar('Val_loss_vein_t', lossVal_v, epochID + 1)
            # tb_val_t.add_scalar('Val_loss_domain_t', lossVal_d, epochID + 1)
            loss_epoch.append(totalLossVal)
            loss_epoch.append(lossVal)
            loss_epoch.append(lossVal_v)
            loss_epoch.append(lossVal_d)
            
            self.loss_logger.append(loss_epoch)

            scheduler.step(totalLossVal_s.data)

            # Save the minimum validation point data
            print('-' * 50)
            print('-' * 50)
            if lossVal_s < lossMIN :
                lossMIN = lossVal_s
                path = self.opt.Output_dir + str(epochID + 1) + '_____' + str(lossTrain_s) + '_____' + str(lossVal_s.item()) + '_____' + str(lossTrain_v_s) + '_____' + str(lossVal_v_s) + '_____' + str(lossTrain_d_s) + '_____' + str(lossVal_d_s.item()) + '.pth.tar'
                torch.save({'epoch': epochID + 1, 'state_dict': training_model.state_dict(), 
                            'best_loss': lossMIN, 'optimizer' : optimizer.state_dict(), 
                            'loss_logger' : self.loss_logger }, path)
                print ('Epoch [' + str(epochID + 1) + '] [save]')
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----]')
            
            # Print the losses
            print("-------- Source Data Results --------")
            print('Train_loss = ' + str(totalLossTrain_s))
            print('Val_loss   = ' + str(np.array(totalLossVal_s.cpu())))
            print('-' * 20)
            print('Train_loss_point = ' + str(lossTrain_s))
            print('Val_loss_point   = ' + str(lossVal_s.item()))
            # print('-' * 20)
            # print('Train_loss_vein = ' + str(lossTrain_v_s))
            # print('Val_loss_vein   = ' + str(lossVal_v_s.item()))
            print('-' * 20)
            print('Train_loss_domain = ' + str(lossTrain_d_s))
            print('Val_loss_domain   = ' + str(lossVal_d_s.item()))
            print('-' * 20)
            print("-------- Target Data Results --------")
            print('Train_loss = ' + str(totalLossTrain))
            print('Val_loss   = ' + str(np.array(totalLossVal.cpu())))
            print('-' * 20)
            print('Train_loss_point = ' + str(lossTrain))
            print('Val_loss_point   = ' + str(lossVal.item()))
            # print('-' * 20)
            # print('Train_loss_vein = ' + str(lossTrain_v))
            # print('Val_loss_vein   = ' + str(lossVal_v_s.item()))
            print('-' * 20)
            print('Train_loss_domain = ' + str(lossTrain_d))
            print('Val_loss_domain   = ' + str(lossVal_d_s.item()))
            print('-' * 50)

            np.savez(self.opt.Output_dir + 'loss_logger.npz', loss_logger = np.array(self.loss_logger))
            loss_epoch = []
        
        # tb_s.close()
        # tb_val_s.close()
        # tb_t.close()
        # tb_val_t.close()
        
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
    parser.add_argument("--source", type=str, metavar = "", help="Source Data", default = "Bosphorus")
    parser.add_argument("--target", type=str, metavar = "", help="Target Data", default = "Vera")

    parser.add_argument("--nnArchitecture", type=str, metavar = "", help="Name of Model Architechture", default = "resnet50")
    parser.add_argument("--nnIsTrained", action = "store_true", help="Use Trained Network or not", default = True)
    parser.add_argument("--nnInChanCount", type=int, metavar = "", help="Number of Input channel", default = 3)
    parser.add_argument("--nnClassCount", type=int, metavar = "", help="Number of predicted values", default = 4)

    parser.add_argument("--trBatchSize", type=int, metavar = "", help="Batch Size", default = 16)
    parser.add_argument("--trMaxEpoch", type=int, metavar = "", help="Number of epochs for training", default = 800)
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--loss_weights", type=float, metavar = "", help="Weights of different losses (MSE, Vein, Domain)", default = [1, 0, 1])
    
    parser.add_argument("--gpu", action = "store_true", help="Use GPU or not", default = True)
    parser.add_argument("--checkpoint", type=str, metavar = "", help="Checkpoint File", 
    default = None ) # './Model_Output/' + '42_____7.746071508375265_____8.61897087097168_____0.0_____0.0_____0.6471325502557269_____0.7307562828063965.pth.tar')
    
    opt = parser.parse_args()
    print("-" * 50)
    print(opt)
    print("-" * 50)
    
    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    vein = VeinNetTrainer(opt)

    vein.training()
