import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


class Hungarian(object):
    def __init__(self):
        pass
    
    def __call__(self, X, *args, **kwargs):
        device = X.device
        X = X.cpu().detach().numpy()
        bs, n1, n2 = X.shape
        
        pred_X = torch.zeros((bs,n1,n2), dtype=torch.float32)
        for i in range(bs):
            pred_permute = linear_sum_assignment(-X[i])[1].tolist()
            pred_X[i] = torch.eye(n2, dtype=torch.float32)[pred_permute]
        return pred_X.to(device)