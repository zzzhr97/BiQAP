import torch
import torch.nn.functional as F

from src.utils.utils import unsqueeze_bs


def get_bce_weight(target):
    bs, n1, n2 = target.shape
    w0 = n1*n2 / (n1*n2 - n1)
    w1 = n2
    weight_matrix = torch.full_like(target, w0, dtype=target.dtype, device=target.device)
    weight_matrix[target == 1] = w1
    weight_matrix /= 2
    return weight_matrix

def bce_loss(pred, data, **kwargs):
    target = data['gt_X']
    bs, n1, n2 = target.shape
    
    target = target.unsqueeze(1).repeat(1, pred.shape[0]//bs, 1, 1).view(-1, n1, n2)
    
    bce_weight = get_bce_weight(target)
    loss = F.binary_cross_entropy(pred, target, weight=bce_weight)
    return loss