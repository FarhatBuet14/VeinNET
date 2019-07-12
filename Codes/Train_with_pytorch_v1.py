############################  Import Libraries  ###############################
###############################################################################

import numpy as np
import time
import cv2
from os.path import join
from PIL import Image
import pandas as pd
import matplotlib as plt

########################## Import libraries for Model  ########################
###############################################################################

import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as func
import torchvision
from torchvision import transforms, datasets, models


######################### Extraction Data to Dataset  #########################
###############################################################################

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

############################# Define training method  #########################
###############################################################################

def train_model(dataloaders, model, cal_loss, optimizer, 
                scheduler, batch_size = 2, num_epochs = 10, use_gpu = True):
    since = time.time()
    train_losses = []
    val_losses = []
    best_model_wts = model.state_dict()
    best_loss = 100.0

    for epoch in range(num_epochs+1):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_batch = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                labels = labels
                
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda(), requires_grad = True)
                    labels = Variable(labels.cuda(), requires_grad = True)
                else:
                    inputs = Variable(inputs, requires_grad = True)
                    labels = Variable(labels, requires_grad = True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                if(cal_loss.loss_type == 'mae'):
                    loss = cal_loss.mae(labels, outputs)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data.item()
                running_batch +=1
            epoch_loss = running_loss / running_batch            
            # Save the loss history
            if phase == 'train': train_losses.append(epoch_loss)
            else: val_losses.append(epoch_loss)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses

###########################  Define Loss Funtion ###############################
################################################################################

class cal_loss():
    def __init__(self, loss_type):
        self.loss_type = loss_type
    
    def mae(self, target, pred):
        self.target = target
        self.pred = pred
        return torch.sum(torch.abs(pred - target)) / target.numel()


##############################  Import Data  ##################################
###############################################################################

# Input Data Folders
train_Output_data = "./Model_Output/"
data_folder = "./Data/"
train_data_folder = "./Data/Full X_train_data/"


# Output Data Folders
weightFile = train_Output_data + 'WeightFile_best.hdf5'
saved_model_File = train_Output_data + 'Saved_Model.h5'

# Import Main Data
data = np.load(data_folder + 'train_test_data_without_augmetation.npz') 
X_train_names = data['X_train_names'].astype(str)
y_train = data['y_train']

# Import Augmented Data
data = np.load(data_folder + "Augmented_Train_data.npz") 
X_train_aug_names = data['X_train_aug_names'].astype(str)
y_train_aug = data['y_train_aug'].reshape((-1, 4))

# Concatenate main data and augmented data
X_names = np.concatenate((X_train_names, X_train_aug_names), axis = 0)
y = np.concatenate((y_train, y_train_aug), axis = 0)

data = []
for index in range(0, len(X_names)):
    data.append([X_names[index], y[index, 0], y[index, 1],
                  y[index, 2], y[index, 3]])

data_df = pd.DataFrame(data, columns=['file_name', 'point_1x', 'point_1y',
                                        'point_2x', 'point_2y']) 

train_data = data_df.sample(frac=0.7)
valid_data = data_df[~data_df['file_name'].isin(train_data['file_name'])]

trans = transforms.Compose([transforms.ToTensor()])

train_set = SeedlingDataset(train_data, train_data_folder, 
                            transform = trans, normalize = True)
val_set = SeedlingDataset(valid_data, train_data_folder, 
                          transform = trans, normalize = True)

batch_size = 16
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(val_set, batch_size = batch_size, shuffle = True)


dataset_sizes = {
    'train': len(train_loader.dataset), 
    'valid': len(valid_loader.dataset)
}

num_pred_value = y_train.shape[1]


################################## Model Train  ###############################
###############################################################################

model = models.resnet50(pretrained=True)
# model = models.resnet18(pretrained=True)
model.conv1.in_channels = 1

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_pred_value)


# # training with these layers unfrozen for a couple of epochs after the initial frozen training
# # Freeze model parameters
# for param in resnet50.parameters():
#     param.requires_grad = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model_loss = cal_loss(loss_type = 'mae')
optimizer = torch.optim.RMSprop(model.fc.parameters(),lr=1e-2, 
                                alpha=0.99, eps=1e-8, weight_decay=0, 
                                momentum=0, centered=False)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, 
                                       step_size=7,
                                       gamma=0.1)


loaders = {'train':train_loader, 'valid':valid_loader}

gpu_available = torch.cuda.is_available()
num_epochs = 20
model, train_losses, val_losses = train_model(loaders, model, model_loss, optimizer, 
                    exp_lr_scheduler, batch_size, num_epochs, 
                    gpu_available)


torch.save(model, 'resnet50_mine.pth')

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()













