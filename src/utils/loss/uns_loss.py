import torch
import torch.nn.functional as F

from src.utils.problem import get_obj_KBQAP, KBQAP_norm


def uns_KBQAP_loss(pred, data, maximize=True):
    bs, n1, n2 = pred.shape
    F1, F2, Kp, sbs = data['F1'], data['F2'], data['Kp'], data['sbs']
    F1, F2, Kp = KBQAP_norm(F1, F2, Kp)
    obj = get_obj_KBQAP(F1, F2, Kp, pred)
    
    gt_X = data['gt_X']
    assert gt_X.shape[0] == data['F1'].shape[0]
    gt_obj = get_obj_KBQAP(F1, F2, Kp, gt_X).unsqueeze(1).repeat(1, bs//gt_X.shape[0]).view(-1) 
    
    if maximize:
        loss = gt_obj - obj
    else:
        loss = obj - gt_obj

    # normalization
    norm_loss = torch.div(loss, n1+n2)

    # select loss
    final_loss = torch.mean(norm_loss)
    
    return final_loss