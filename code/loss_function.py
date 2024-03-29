import torch
import torch.nn as nn
import numpy as np
import torch.functional as F
    
#PyTorch
class WeightBCELoss(torch.nn.Module):
    
    def __init__(self, w_p = 1, w_n = 5):
        super(WeightBCELoss, self).__init__()
        
        self.w_p = w_p
        self.w_n = w_n
        
    def forward(self, inputs, targets, epsilon = 1e-7):
        
        
        loss_pos = -1 * torch.mean(self.w_p * targets * torch.log(inputs + epsilon))
        loss_neg = -1 * torch.mean(self.w_n * (1-targets) * torch.log((1-inputs) + epsilon))
        
        loss = loss_pos + loss_neg
        
        return loss
    
class PunishingBCEloss(torch.nn.Module):
    def __init__(self):
        super(PunishingBCEloss, self).__init__()

    def forward(self,inputs, targets, epsilon=1e-7):
        
        lane_pixel = targets
        background_pixel = 1-targets


        true_positive = -1 * torch.mean(inputs * lane_pixel) # 얘는 커야하고
        true_negative = -1 * torch.mean(inputs * background_pixel) # 예는 작아야하고

        return 0
    
def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()


    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score

class DiceLoss(nn.Module):

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(y_pr, y_gt, beta=1., 
                           eps=self.eps, threshold=None, 
                           activation=self.activation)


class BCEDiceLoss(DiceLoss):

    def __init__(self, eps=1e-7, activation='sigmoid', lambda_dice=1.0, lambda_bce=1.0):
        super().__init__(eps, activation)
        if activation == None:
            self.bce = nn.BCELoss(reduction='mean')
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.lambda_dice=lambda_dice
        self.lambda_bce=lambda_bce

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return (self.lambda_dice*dice) + (self.lambda_bce* bce)