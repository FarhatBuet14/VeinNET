#############################  IMPORT LIBRARIES  ############################
import torch
import torch.backends.cudnn as cudnn

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
