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
    
    ######################### Get the IDs  #########################
    ##################################################################

    def get_ID(self, img_names):
        IDs = []
        for name in img_names:
            temp = name.split("_")[0].split()[0]
            IDs.append(np.array(re.findall(r'\d+', temp)).astype(int)[0])
            
        return IDs

    ######################### Data Generator #########################
    ##################################################################

    class SeedlingDataset(Dataset):
        def __init__(self, labels, root_dir, subset=False, 
                    transform = None, normalize=True):
            self.labels = labels
            self.root_dir = root_dir
            self.transform = transform
            self.normalize = normalize
        
        def get_processed(self, image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            accumEdged = np.zeros(image.shape[:2], dtype="uint8")
            for chan in cv2.split(image):
                chan = cv2.medianBlur(chan, 3)
                chan = cv2.Canny(chan, 50, 150)
                accumEdged = cv2.bitwise_or(accumEdged, chan) 
            image = np.zeros((240, 300, 3), dtype="float32")
            image[:, :, 0] = gray
            image[:, :, 1] = accumEdged
            image[:, :, 2] = accumEdged # Will add a precessed image
            image = image/255
            return image
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            img_name = self.labels.iloc[idx, 0]
            fullname = join(self.root_dir, img_name)
            image = np.array(Image.open(fullname).convert('RGB'))
            image = self.get_processed(image)
            labels = torch.tensor(self.labels.iloc[idx, 1:], 
                                dtype = torch.float32)
            if self.transform:
                image = self.transform(image)
            return image, labels
    
    ######################### MAE/MSE Loss #########################
    ################################################################

    class Cal_loss():
        def __init__(self, loss_type):
            self.loss_type = loss_type
        
        def mae(self, target, pred):
            loss = torch.sum(torch.abs(pred - target)) / target.numel()
            loss = Variable(loss, requires_grad=True)
            return loss
        
        def mse(self, target, pred):
            loss = torch.sum((pred - target)**2) / target.numel()
            loss = Variable(loss, requires_grad=True)
            return loss

        def calculate(self, target, pred):
            if(self.loss_type == 'mae'):
                return self.mae(target, pred)
            elif(self.loss_type == 'mse'):
                return self.mse(target, pred)
    
    ######################### Custom Vein Loss #########################
    ####################################################################

    class Vein_loss_class():    
        def __init__(self, varTarget, varOutput, varInput, 
                    cropped_fldr, bounding_box_folder, ids):
            self.target = varTarget[0].cpu().numpy()
            self.output = varOutput[0].cpu().numpy()
            self.input = varInput[0].cpu().numpy()
            self.id = ids.cpu().numpy()
            
            del varInput, varOutput, varTarget, ids
            
            self.bounding_box_folder = bounding_box_folder
            self.cropped_fldr = cropped_fldr
            self.height = 90
            self.width = 70
            self.th = 10
            self.thresh_h = 200
            self.thresh_l = 70

        def get_vein_img(self, save_vein_pic = False,
                        save_bb = False):    
            
            top_left = self.output[0:2]
            top_right = self.output[2:4]
            
            # Find the angle to rotate the image
            angle  = (180/np.pi) * (np.arctan((top_left[1] - top_right[1])/
                                    (top_left[0] - top_right[0])))
            
            # Rotate the image to cut rectangle from the images
            points_pred = self.output.reshape((1, 2, 2))
            points_test = self.target.reshape((1, 2, 2))
            image = self.input[0, :, :] * 255
            image = image.reshape((1, 240, 300))
            image_rotated , keypoints_pred_rotated = iaa.Affine(rotate=-angle)(images=image, 
                                        keypoints=points_pred)
            _ , keypoints_test_rotated = iaa.Affine(rotate=-angle)(images=image, 
                                        keypoints=points_test)
            
            image_rotated = image_rotated.reshape((240, 300))
            keypoints_pred_rotated = keypoints_pred_rotated.reshape((2, 2))
            keypoints_test_rotated = keypoints_test_rotated.reshape((2, 2))
            
            # Rotated Points
            top_left_ = keypoints_pred_rotated[0]    
            top_left_ = tuple(top_left_.reshape(1, -1)[0])
            
            center = np.zeros((2, )).astype(int)
            center[0] = top_left_[0] + int(self.width/2)  - self.th
            center[1] = top_left_[1] + int(self.height/2)
            center = tuple(center.reshape(1, -1)[0])
            
            # Crop the Vein Image
            crop = cv2.getRectSubPix(image_rotated, (self.width, self.height), 
                                    center)
            if(save_vein_pic):
                cv2.imwrite(self.cropped_fldr + str(self.id) + '.bmp', crop)
            
            # Draw Predicted Troughs
            points = keypoints_pred_rotated.reshape((2, 2))    
            for point in points:   
                point = np.array(point).astype(int)
                cv2.circle(image_rotated, (point[0], point[1]), 
                        5, (0, 0, 0), -1)
            
            # Draw Actual Troughs
            points = keypoints_test_rotated.reshape((2, 2))    
            for point in points:   
                point = np.array(point).astype(int)
                cv2.circle(image_rotated, (point[0], point[1]), 
                        5, (255, 0, 0), -1)
            
            # Points for Bounding Boxes
            tl = np.zeros((2, )).astype(int)
            tl[0] = center[0] - int(self.width/2)  
            tl[1] = center[1] - int(self.height/2)
            tl = tuple(tl.reshape(1, -1)[0])
            
            br = np.zeros((2, )).astype(int)
            br[0] = center[0] + int(self.width/2)  
            br[1] = center[1] + int(self.height/2)
            br = tuple(br.reshape(1, -1)[0])
            
            # Draw Bounding Boxes and Save the image
            image_rotated = cv2.rectangle(image_rotated, tl, br , (0,0,0), 2)
            if(save_bb):
                cv2.imwrite(self.bounding_box_folder + str(self.id) + '.bmp', 
                            image_rotated)
            return crop
        
        def cal_vein_loss(self):

            vein_image = self.get_vein_img()
            
            # Calculate loss from extracted Vein Image
            accu = ((vein_image <= self.thresh_h)  & (vein_image >= self.thresh_l))
            true = np.count_nonzero(accu)
            false = (accu.shape[0] * accu.shape[1]) - true
            vein_loss = Variable(torch.tensor((false / (false + true))), requires_grad=True)
            return vein_loss

    ######################### Simple CNN Model #########################
    ####################################################################
    
    class SimpleCNN(torch.nn.Module):
    
        #Our batch shape for input x is (3, 240, 300)
        
        def __init__(self):
            super(VeinNetTrainer.SimpleCNN, self).__init__()
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=1, stride=1))
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=1, stride=1))
            self.drop_out = torch.nn.Dropout()
            self.fc1 = torch.nn.Linear(236 * 296 * 64, 1000)
            self.fc2 = torch.nn.Linear(1000, 4)
            
        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.drop_out(out)
            out = self.fc1(out)
            out = self.fc2(out)
            return out
    
    ######################### Epoch Train #########################
    ###############################################################
    
    def epochTrain (self, model, dataLoader, optimizer, scheduler, trBatchSize,
                    epochMax, classCount, loss_class, vein_loss = False,
                    cropped_fldr = None, bounding_box_folder = None):
        
        model.train()
        loss = 0
        for batchID, (input, target) in enumerate (dataLoader):
            # torch.cuda.empty_cache()
            
            id = target[:, 0]
            target = target[:, 1:]
            varInput = torch.autograd.Variable(input.cuda())
            varOutput = model(varInput)
            output = (Variable(varOutput).data).cpu()

            del varInput, varOutput
            
            loss += loss_class.calculate(target, output)
            if(vein_loss):
                vein_loss_class = self.Vein_loss_class(target, output, 
                                                        input, cropped_fldr, 
                                                        bounding_box_folder, id)
                vein_loss_class.get_vein_img(save_vein_pic = True,
                                                        save_bb = True)
                loss += vein_loss_class.cal_vein_loss()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            del input, target, id, output, vein_loss_class
        loss_mean = loss / trBatchSize
        return loss_mean

    ######################### Epoch Validation #########################
    ####################################################################
    
    def epochVal (self, model, dataLoader, optimizer, scheduler, trBatchSize,
                    epochMax, classCount, loss_class,  vein_loss = False,
                    cropped_fldr = None, bounding_box_folder = None):
        
        model.eval ()
        loss = 0
        with torch.no_grad():
            for i, (input, target) in enumerate (dataLoader):
                # torch.cuda.empty_cache()
                id = target[:, 0]
                target = target[:, 1:]
                varInput = torch.autograd.Variable(input.cuda())
                varOutput = model(varInput)
                output = (Variable(varOutput).data).cpu()

                del varInput, varOutput
                
                loss += loss_class.calculate(target, output)
                if(vein_loss):
                    vein_loss_class = self.Vein_loss_class(target, output, 
                                                            input, cropped_fldr, 
                                                            bounding_box_folder, id)
                    vein_loss_class.get_vein_img(save_vein_pic = True,
                                                            save_bb = True)
                    loss += vein_loss_class.cal_vein_loss()

                del input, target, output, id, vein_loss_class
            loss = loss / trBatchSize
        return loss
    
    ########################### Load Model ###########################
    ##################################################################
    
    def load_model (self, nnArchitecture, nnIsTrained, 
                nnInChanCount, nnClassCount):
        
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

        print('-' * 100)
        for idx, m in enumerate(model.modules()):
            print("{} is {}".format(idx, m))
        print('-' * 100)
        
        if(self.gpu):
            model = model.cuda()
        
        # # Freeze model weights
        # for param in model.parameters():
        #     param.requires_grad = False
        
        # Print Trainable and Non-Trainable Parameters
        print('-' * 100)
        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        
        return model
    
    ######################### Train Function #########################
    ##################################################################
    
    def training (self, pathDirData, pathModel, nnArchitecture, 
                nnIsTrained, nnInChanCount, nnClassCount, 
                trBatchSize, trMaxEpoch, launchTimestamp = None, 
                checkpoint = None, vein_loss = False, 
                cropped_fldr = None, bounding_box_folder = None):
        
        model = self.load_model(nnArchitecture, nnIsTrained, 
                                nnInChanCount, nnClassCount)
        
        #-------------------- SETTINGS: IMPORT DATA
        # Import Main Data
        data = np.load(pathDirData + 'train_test_data_without_augmetation.npz') 
        X_names = data['X_train_names'].astype(str)
        y = data['y_train']

        # Import Augmented Data
        data = np.load(pathDirData + "Augmented_Train_data.npz") 
        X_train_aug_names = data['X_train_aug_names'].astype(str)
        y_train_aug = data['y_train_aug'].reshape((-1, 4))

        # Concatenate main data and augmented data
        X_names = np.concatenate((X_names, X_train_aug_names), axis = 0)
        y = np.concatenate((y, y_train_aug), axis = 0)

        ID = self.get_ID(X_names)
        
        data = []
        for index in range(0, len(X_names)):
            data.append([X_names[index], ID[index], y[index, 0], y[index, 1],
                        y[index, 2], y[index, 3]])

        data_df = pd.DataFrame(data, columns=['file_name', 'id', 'point_1x', 
                                            'point_1y', 'point_2x', 'point_2y']) 

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
        
        # torch.cuda.empty_cache()
        del train_data, valid_data, X_names, y, ID, data_df
        
        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), 
                                eps=1e-08, weight_decay=1e-5)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, 
                                                    patience = 5, mode = 'min')
                
        #-------------------- SETTINGS: LOSS
        loss_class = self.Cal_loss(loss_type = 'mae')

        # #---- Load checkpoint 
        # if checkpoint != None:
        #    modelCheckpoint = torch.load(checkpoint)
        #    model.load_state_dict(modelCheckpoint['state_dict'])
        #    optimizer.load_state_dict(modelCheckpoint['optimizer'])

        
        #---- TRAIN THE NETWORK
        
        lossMIN = 100000
        # torch.cuda.empty_cache()
        print('-' * 50 + 'Start Training' + '-' * 50)
        for epochID in range (0, trMaxEpoch):

            timestampSTART = time.strftime("%d%m%Y") + '-' + time.strftime("%H%M%S")
            lossTrain = self.epochTrain (model, train_loader, optimizer, 
                            scheduler, trBatchSize, trMaxEpoch, nnClassCount, 
                            loss_class,  vein_loss,
                            cropped_fldr, bounding_box_folder)
            
            lossVal = self.epochVal (model, valid_loader, optimizer, 
                                    scheduler, trBatchSize, trMaxEpoch, 
                                    nnClassCount, loss_class,  vein_loss,
                                    cropped_fldr, bounding_box_folder)
            
            timestampEND = time.strftime("%d%m%Y") + '-' + time.strftime("%H%M%S")
            
            scheduler.step(lossVal.data)
            
            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, pathModel)
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossTrain) + str(lossVal))
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossTrain) + str(lossVal))
        
        # torch.cuda.empty_cache()
        print('-' * 50 + 'Finished Training' + '-' * 50)
        print('-' * 100)


    ######################### Test Function #########################
    #################################################################
    
    def test (self, pathFileTest, pathModel, nnArchitecture, nnInChanCount, 
                nnClassCount, nnIsTrained, trBatchSize, launchTimeStamp,
                vein_loss = False, cropped_fldr = None, 
                bounding_box_folder = None):   
        
        cudnn.benchmark = True
        
        model = self.load_model(nnArchitecture, nnIsTrained, 
                                nnInChanCount, nnClassCount)
        
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: IMPORT DATA
        # Import Main Data
        data = np.load(pathFileTest + 'train_test_data_without_augmetation.npz') 
        X_test_names = data['X_test_names'].astype(str)
        y_test = data['y_test']
        
        ID = self.get_ID(X_test_names)

        data = []
        for index in range(0, len(X_test_names)):
            data.append([X_test_names[index], ID[index], y_test[index, 0], y_test[index, 1],
                        y_test[index, 2], y_test[index, 3]])

        test_data = pd.DataFrame(data, columns=['file_name', 'id', 'point_1x', 'point_1y',
                                                'point_2x', 'point_2y']) 

        #-------------------- SETTINGS: DATASET BUILDERS
        trans = transforms.Compose([transforms.ToTensor()])
        test_set = self.SeedlingDataset(test_data, pathFileTest, 
                                    transform = trans, normalize = True)

        test_loader = DataLoader(test_set, batch_size = trBatchSize, shuffle = True)
        loss_class = self.Cal_loss(loss_type = 'mae')
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
                        vein_loss_class = self.Vein_loss_class(varTarget, varOutput, 
                                                                varInput, cropped_fldr, 
                                                                bounding_box_folder, ids)
                        vein_img = vein_loss_class.get_vein_img()
                        valid_loss += vein_loss_class.cal_vein_loss()
        print('-' * 50 + 'Finished Testing' + '-' * 50)
        print('-' * 100)
        return valid_loss
#-------------------------------------------------------------------------------- 
#################################################################################