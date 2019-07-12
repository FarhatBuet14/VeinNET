import numpy as np
import time
import pandas as pd
from os.path import join
from PIL import Image
import cv2

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

#######################  Define VeinBetTrainer Class  #########################
###############################################################################
###############################################################################
###############################################################################

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

    def __init__(self, gpu = False):
        self.gpu = gpu
    
    ######################### Data Generator #########################

    class SeedlingDataset(Dataset):
        def __init__(self, labels, root_dir, subset=False, 
                    transform = None, normalize=True):
            self.labels = labels
            self.root_dir = root_dir
            self.transform = transform
            self.normalize = normalize
        
        def get_accumEdged(self, image):
            image = cv2.medianBlur(np.array(image), 3)       
            return torch.tensor(cv2.Canny(image, 50, 150))
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            img_name = self.labels.iloc[idx, 0]
            fullname = join(self.root_dir, img_name)
            image = np.array(Image.open(fullname).convert('RGB'))
            labels = torch.tensor(self.labels.iloc[idx, 1:], 
                                dtype = torch.float32)
            if self.transform:
                image = self.transform(image)
            return image, labels
    
    ######################### MAE/MSE Loss #########################

    class Cal_loss():
        def __init__(self, loss_type):
            self.loss_type = loss_type
        
        def mae(self, target, pred):
            self.target = target
            self.pred = pred
            return torch.sum(torch.abs(pred - target)) / target.numel()
        
        def mse(self, target, pred):
            self.target = target
            self.pred = pred
            return torch.sum((pred - target)**2) / target.numel()

        def calculate(self, target, pred):
            if(self.loss_type == 'mae'):
                return self.mae(target, pred)
            elif(self.loss_type == 'mse'):
                return self.mse(target, pred)
    
    ######################### Custom Vein Loss #########################

    # #---- Computes area under ROC curve 
    # #---- dataGT - ground truth data
    # #---- dataPRED - predicted data
    # #---- classCount - number of classes
    
    # def computeAUROC (self, dataGT, dataPRED, classCount):
        
    #     outAUROC = []
        
    #     datanpGT = dataGT.cpu().numpy()
    #     datanpPRED = dataPRED.cpu().numpy()
        
    #     for i in range(classCount):
    #         outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
    #     return outAUROC

    ######################### Simple CNN Model #########################

    # class SimpleCNN(torch.nn.Module):
    
    #     #Our batch shape for input x is (3, 240, 300)
        
    #     def __init__(self):
    #         super(VeinNetTrainer.SimpleCNN, self).__init__()
    #         self.layer1 = torch.nn.Sequential(
    #             torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
    #             torch.nn.ReLU(),
    #             torch.nn.MaxPool2d(kernel_size=1, stride=1))
    #         self.layer2 = torch.nn.Sequential(
    #             torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
    #             torch.nn.ReLU(),
    #             torch.nn.MaxPool2d(kernel_size=1, stride=1))
    #         self.drop_out = torch.nn.Dropout()
    #         self.fc1 = torch.nn.Linear(236 * 296 * 64, 1000)
    #         self.fc2 = torch.nn.Linear(1000, 4)
            
    #     def forward(self, x):
    #         out = self.layer1(x)
    #         out = self.layer2(out)
    #         out = out.reshape(out.size(0), -1)
    #         out = self.drop_out(out)
    #         out = self.fc1(out)
    #         out = self.fc2(out)
    #         return out
    
    ######################### Epoch Train #########################

    def epochTrain (self, model, dataLoader, optimizer, scheduler, 
                    epochMax, classCount, loss_class):
        
        model.train()

        for batchID, (input, target) in enumerate (dataLoader):
            
            if(self.gpu):
                input = input.cuda()
                target = target.cuda()
                 
            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)         
            varOutput = model(varInput)
            
            loss = loss_class.calculate(varTarget, varOutput)
                       
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    ######################### Epoch Validation #########################

    def epochVal (self, model, dataLoader, optimizer, scheduler, 
                    epochMax, classCount, loss_class):
        
        model.eval ()
        
        lossVal = 0
        lossValNorm = 0    
        losstensorMean = 0
        with torch.no_grad():
            for i, (input, target) in enumerate (dataLoader):
                
                if(self.gpu):
                    input = input.cuda()
                    target = target.cuda()
                    
                varInput = torch.autograd.Variable(input, volatile=True)
                varTarget = torch.autograd.Variable(target, volatile=True)    
                varOutput = model(varInput)
                
                losstensor = loss_class.calculate(varOutput, varTarget)
                losstensorMean += losstensor
                
                lossVal += losstensor.data
                lossValNorm += 1
                
            outLoss = lossVal / lossValNorm
            losstensorMean = losstensorMean / lossValNorm
        
        return outLoss, losstensorMean
    
    ######################### Train Function #########################

    def training (self, pathDirData, nnArchitecture, nnIsTrained, 
                nnInChanCount, nnClassCount, trBatchSize, 
                trMaxEpoch, launchTimestamp, checkpoint):
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'resnet18': model = models.resnet18(nnIsTrained)
        elif nnArchitecture == 'resnet34': model = models.resnet34(nnIsTrained)
        elif nnArchitecture == 'resnet50': model = models.resnet50(nnIsTrained)
        elif nnArchitecture == 'alexnet': model = models.alexnet(nnIsTrained)
        elif nnArchitecture == 'vgg19': model = models.vgg19(nnIsTrained)
        else: 
            model = self.SimpleCNN()
        
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, nnClassCount)
        #model.classifier._modules['6'] = torch.nn.Linear(4096, nnClassCount)

        for idx, m in enumerate(model.modules()):
            print("{} is {}".format(idx, m))

        total_parameter = sum([param.nelement() for param in model.parameters()])
        print("total_parameter - {}".format(total_parameter))
        
        if(self.gpu):
            model = model.cuda()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
        
        #-------------------- SETTINGS: IMPORT DATA
        # Import Main Data
        data = np.load(pathDirData + 'train_test_data_without_augmetation.npz') 
        X_names = data['X_train_names'].astype(str)
        y = data['y_train']

        # # Import Augmented Data
        # data = np.load(pathDirData + "Augmented_Train_data.npz") 
        # X_train_aug_names = data['X_train_aug_names'].astype(str)
        # y_train_aug = data['y_train_aug'].reshape((-1, 4))

        # # Concatenate main data and augmented data
        # X_names = np.concatenate((X_train_names, X_train_aug_names), axis = 0)
        # y = np.concatenate((y_train, y_train_aug), axis = 0)

        data = []
        for index in range(0, len(X_names)):
            data.append([X_names[index], y[index, 0], y[index, 1],
                        y[index, 2], y[index, 3]])

        data_df = pd.DataFrame(data, columns=['file_name', 'point_1x', 'point_1y',
                                                'point_2x', 'point_2y']) 

        train_data = data_df.sample(frac=0.7)
        valid_data = data_df[~data_df['file_name'].isin(train_data['file_name'])]


        #-------------------- SETTINGS: DATASET BUILDERS
        trans = transforms.Compose([transforms.ToTensor()])
        train_set = self.SeedlingDataset(train_data, pathDirData, 
                                    transform = trans, normalize = True)
        val_set = self.SeedlingDataset(valid_data, pathDirData, 
                                transform = trans, normalize = True)

        train_loader = DataLoader(train_set, batch_size = trBatchSize, shuffle = True)
        valid_loader = DataLoader(val_set, batch_size = trBatchSize, shuffle = True)

        
        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
                
        #-------------------- SETTINGS: LOSS
        loss_class = self.Cal_loss(loss_type = 'mae')
        
        #---- Load checkpoint 
        if checkpoint != None:
           modelCheckpoint = torch.load(checkpoint)
           model.load_state_dict(modelCheckpoint['state_dict'])
           optimizer.load_state_dict(modelCheckpoint['optimizer'])

        
        #---- TRAIN THE NETWORK
        
        lossMIN = 100000
        
        for epochID in range (0, trMaxEpoch):
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime

            torch.cuda.empty_cache()             
            self.epochTrain (model, train_loader, optimizer, scheduler, 
                            trMaxEpoch, nnClassCount, loss_class)
            torch.cuda.empty_cache()
            lossVal, losstensor = self.epochVal (model, valid_loader, optimizer, 
                                                scheduler, trMaxEpoch, nnClassCount, loss_class)
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            scheduler.step(losstensor.data)
            
            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
     
    ######################### Test Function #########################
    
    def test (self, pathFileTest, pathModel, nnArchitecture, nnInChanCount, 
                nnClassCount, nnIsTrained, trBatchSize, launchTimeStamp):   
        
        cudnn.benchmark = True
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'resnet18': model = models.resnet18(nnClassCount, nnIsTrained)
        elif nnArchitecture == 'resnet34': model = models.resnet34(nnClassCount, nnIsTrained)
        elif nnArchitecture == 'resnet50': model = models.resnet50(nnClassCount, nnIsTrained)
        elif nnArchitecture == 'alexnet': model = models.alexnet(nnIsTrained)
        elif nnArchitecture == 'vgg19': model = models.vgg19(nnIsTrained)
        else: 
            model = self.SimpleCNN()
        
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, nnClassCount)
        #model.classifier._modules['6'] = torch.nn.Linear(4096, nnClassCount)
        
        if(self.gpu):
            model = model.cuda()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
        
        total_parameter = sum([param.nelement() for param in model.parameters()])
        print("total_parameter - {}".format(total_parameter))
        
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: IMPORT DATA
        # Import Main Data
        data = np.load(pathFileTest + 'train_test_data_without_augmetation.npz') 
        X_test_names = data['X_test_names'].astype(str)
        y_test = data['y_test']
        
        data = []
        for index in range(0, len(X_test_names)):
            data.append([X_test_names[index], y_test[index, 0], y_test[index, 1],
                        y_test[index, 2], y_test[index, 3]])

        test_data = pd.DataFrame(data, columns=['file_name', 'point_1x', 'point_1y',
                                                'point_2x', 'point_2y']) 

        #-------------------- SETTINGS: DATASET BUILDERS
        trans = transforms.Compose([transforms.ToTensor()])
        test_set = self.SeedlingDataset(test_data, pathFileTest, 
                                    transform = trans, normalize = True)

        test_loader = DataLoader(test_set, batch_size = trBatchSize, shuffle = True)
        loss_class = self.Cal_loss(loss_type = 'mae')
        model.eval() 
        torch.cuda.empty_cache()
        valid_loss = 0
        with torch.no_grad():      
            for i, (input, target) in enumerate(test_loader):
                
                target = target.cuda()
                varInput = torch.autograd.Variable(input.cuda(), volatile=True)
                outPRED = model(varInput)
                valid_loss += loss_class.calculate(target, outPRED)
        
        return valid_loss
#-------------------------------------------------------------------------------- 
