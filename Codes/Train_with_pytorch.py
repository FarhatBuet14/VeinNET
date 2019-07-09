############################  Import Libraries  ###############################
###############################################################################

import numpy as np
import time
import cv2
from os.path import join
from PIL import Image

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
    def __init__(self, names, labels, root_dir, subset=False, 
                 transform = None, normalize=True):
        self.names = names
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        self.normalize = normalize
    
    def get_accumEdged(self, image):
        image = cv2.medianBlur(np.array(image), 3)       
        return torch.tensor(cv2.Canny(image, 50, 150))
    
    def __len__(self):
        return self.labels.shape[0]
    
    def get_data(self):
        for img_name in self.names:
            fullname = join(self.root_dir, img_name)
            image = Image.open(fullname).convert('RGB')
            if self.transform:
                image = self.transform(image)
            trans = transforms.ToPILImage()
            trans1 = transforms.ToTensor()
            image = self.get_accumEdged(trans(trans1(image)))
            if self.normalize:
                image = image / 255
            labels = torch.as_tensor(self.labels)
            image = torch.as_tensor(image, dtype = torch.int32)
        return image, labels

############################# Define training method  #########################
###############################################################################

def train_model(dataloaders, model, loss, optimizer, 
                scheduler, num_epochs = 10):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_batch = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                labels = labels.view(-1)
                
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = loss(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                running_batch +=1

            epoch_loss = running_loss / running_batch
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

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

# Train Test Splitting
random_seed = 0
from sklearn.model_selection import train_test_split
X_train_names, X_val_names, y_train, y_val = train_test_split(X_names, y, test_size = 0.3, 
                                                  random_state=random_seed)

trans = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

train_set = SeedlingDataset(X_train_names, y_train, train_data_folder, trans, normalize=True)
val_set = SeedlingDataset(X_val_names, y_val, train_data_folder, trans, normalize=True)

batch_size = 32
train_loader = DataLoader(train_set.get_data(), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_set.get_data(), batch_size=batch_size, shuffle=True)

dataset_sizes = {
    'train': len(train_loader.dataset), 
    'valid': len(valid_loader.dataset)
}

num_pred_value = y_train.shape[1]


################################## Model Train  ###############################
###############################################################################

model = models.resnet50(pretrained=True)

# training with these layers unfrozen for a couple of epochs after the initial frozen training
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_pred_value)

use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), 
                            lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, 
                                       step_size=7,
                                       gamma=0.1)


loaders = {'train':train_loader, 'valid':valid_loader}

model = train_model(loaders, model, loss, optimizer, exp_lr_scheduler, num_epochs=1)
