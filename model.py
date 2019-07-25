#############################  IMPORT LIBRARIES  ############################
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as func
from torchvision import models

########################### Load Model ###########################
##################################################################

def load_model (nnArchitecture, nnIsTrained, 
                nnInChanCount, nnClassCount, gpu = True):
    
    if nnArchitecture == 'resnet18': model = models.resnet18(nnIsTrained)
    elif nnArchitecture == 'resnet34': model = models.resnet34(nnIsTrained)
    elif nnArchitecture == 'resnet50': model = models.resnet50(nnIsTrained)
    elif nnArchitecture == 'alexnet': model = models.alexnet(nnIsTrained)
    elif nnArchitecture == 'vgg19': model = models.vgg19(nnIsTrained)
    else: 
        model = SimpleCNN()
    
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, nnClassCount)
    #model.classifier._modules['6'] = torch.nn.Linear(4096, nnClassCount)

    # let's make our model work with 6 channels
    trained_kernel = model.conv1.weight
    new_conv = torch.nn.Conv2d(nnInChanCount, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*nnInChanCount, dim=1)
    model.conv1 = new_conv

    print('-' * 100)
    for idx, m in enumerate(model.modules()):
        print("{} is {}".format(idx, m))
    print('-' * 100)
    
    if(gpu):
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

######################### Simple CNN Model #########################
####################################################################

class SimpleCNN(torch.nn.Module):

    #Our batch shape for input x is (3, 240, 300)
    def __init__(self):
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
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


if __name__ == "__main__":
    pass
    # main()
