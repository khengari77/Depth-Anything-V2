import torch
from torch import nn
import torch.nn.functional as F

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        print("pred", pred.shape)
        print("target", target.shape)
        print("valid_mask", valid_mask.shape)
        valid_mask = valid_mask.detach()
#        height, width = target.shape[-2:]
#        pred = F.interpolate(pred[:, None], (height, width), mode="bilinear", align_corners=True)[0,0]
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss
