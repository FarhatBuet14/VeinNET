#############################  IMPORT LIBRARIES  ############################
import numpy as np
import torch
import torch.nn.functional as F

import veinloss

######################### MAE/MSE Loss #########################
################################################################

class Cal_loss(torch.nn.Module):
    def __init__(self, loss_type): 
        super(Cal_loss, self).__init__()
        self.loss_type = loss_type
    
    def mae(self, target, pred):
        loss = torch.sum(torch.abs(pred - target)) / target.numel()
        return loss.cpu()
    
    def mse(self, target, pred):
        loss = torch.sum((pred - target)**2) / target.numel()
        return loss.cpu()

    def kl_divergence(self, output, target):
        epsilon = 0.00001
        output = (output + epsilon)
        target = (target + epsilon)
        return F.kl_div(output, target)

    def bceLoss(self, out, tar, threshold = 0):
        bce = torch.nn.BCELoss().cuda()
        return abs(bce(out, tar).cpu() - threshold)

    def forward(self, target, pred, input, img_name, id, org, 
                vein_loss_class = None, loss_weights = [1, 1]):
        # ------- Point Loss
        w_point, w_veinLoss = loss_weights
        if(self.loss_type == 'mae'): self.point_loss_value = self.mae(target, pred)
        elif(self.loss_type == 'mse'): self.point_loss_value = self.mse(target, pred)
        total_loss = w_point * self.point_loss_value
        # ------- Vein Loss
        if(vein_loss_class):
            self.vein_loss_value = vein_loss_class(target, pred, input, img_name, id, org)
            self.loss_logger = vein_loss_class.loss_logger
            self.names = vein_loss_class.names

            total_loss += w_veinLoss * self.vein_loss_value

        return total_loss

if __name__ == "__main__":
    pass
    # main()
