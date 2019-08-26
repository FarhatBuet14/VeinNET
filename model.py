#############################  IMPORT LIBRARIES  ############################
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as func
from torchvision import models
import User_defined_model

########################### Load Model ###########################
##################################################################

def load_model (nnArchitecture, nnIsTrained, 
                nnInChanCount, nnClassCount, gpu = True):
    
    if nnArchitecture == 'resnet18': model = models.resnet18(nnClassCount, nnIsTrained)
    elif nnArchitecture == 'resnet34': model = models.resnet34(nnClassCount, nnIsTrained)
    elif nnArchitecture == 'resnet50': model = models.resnet50(nnClassCount, nnIsTrained)
    elif nnArchitecture == 'alexnet': model = models.alexnet(nnClassCount, nnIsTrained)
    elif nnArchitecture == 'vgg19': model = models.vgg19(nnClassCount, nnIsTrained)
    elif nnArchitecture == 'DENSE-NET-121': 
        from DensenetModels import DenseNet121
        model = DenseNet121(nnClassCount, nnIsTrained)
    elif nnArchitecture == 'DENSE-NET-169': 
        from DensenetModels import DenseNet169
        model = DenseNet169(nnClassCount, nnIsTrained)
    elif nnArchitecture == 'DENSE-NET-201': 
        from DensenetModels import DenseNet201
        model = DenseNet201(nnClassCount, nnIsTrained)
    elif nnArchitecture == 'mine': model = User_defined_model.SimpleCNN()
    
    # num_ftrs = model.fc.in_features
    # model.fc = torch.nn.Linear(num_ftrs, nnClassCount)
    #model.classifier._modules['6'] = torch.nn.Linear(4096, nnClassCount)

    # # let's make our model work with channels we want
    # trained_kernel = model.conv1.weight
    # new_conv = torch.nn.Conv2d(nnInChanCount, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # with torch.no_grad():
    #     new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*nnInChanCount, dim=1)
    # model.conv1 = new_conv

    print('-' * 100)
    # for idx, m in enumerate(model.modules()):
    #     print("{} is {}".format(idx, m))
    # print('-' * 100)
    
    if(gpu):
        model = model.cuda()
    
    # # Freeze model weights
    for param in model.parameters():
        param.requires_grad = True
    
    # Print Trainable and Non-Trainable Parameters
    print('-' * 100)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    
    return model


if __name__ == "__main__":
    pass
    # main()
