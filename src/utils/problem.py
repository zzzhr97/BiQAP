import numpy as np
import torch
from functools import partial

from src.utils.config import INFO, DEBUG
from src.utils.utils import unsqueeze_bs


def list_to_tensor(data: list):
    converted = []
    for x in data:
        if type(x) == np.ndarray or type(x) == list:
            x = torch.tensor(x, dtype=torch.float32)
        converted.append(x)
    return converted

def kronecker_product(A, B):
        """
        Computes the Kronecker product of two matrices A and B.
        """
        Ash, Bsh = len(A.shape), len(B.shape)
        if Ash == Bsh == 2:
            m, n = A.shape
            p, q = B.shape
            res = torch.mul(A.view(m,1,n,1), B.view(1,p,1,q)).view(m*p, n*q)
        elif Ash == Bsh == 3 and A.shape[0] == B.shape[0]:
            bs, m, n = A.shape
            bs, p, q = B.shape
            res = torch.mul(A.view(bs,m,1,n,1), B.view(bs,1,p,1,q)).view(bs, m*p, n*q)
        else:
            assert False, DEBUG(f'Error: A.shape {A.shape}, B.shape {B.shape}')
        return res

def KB2LL(F1, F2, Kp):
    if len(Kp.shape) == 3:
        bs, n1, n2 = Kp.shape
        K = kronecker_product(F1, F2) + torch.diag_embed(Kp.view(bs, -1))
    elif len(Kp.shape) == 2:
        n1, n2 = Kp.shape
        K = kronecker_product(F1, F2) + torch.diag_embed(Kp.view(-1))
    return K

def get_obj_KBQAP(F1, F2, Kp, pred_X):
    F1, F2, Kp, pred_X = list_to_tensor([F1, F2, Kp, pred_X])
    dim = len(pred_X.shape)
    
    if dim == 2: pred_X = unsqueeze_bs(pred_X, 1)
    bs, n1, n2 = pred_X.shape
    assert len(F1.shape) == 3, DEBUG(f'F1.shape {F1.shape}')
    if F1.shape[0] > 1:
        unsqueeze = partial(torch.unsqueeze, dim=1)
        repeat_interleave = partial(torch.repeat_interleave, repeats=bs//F1.shape[0], dim=1)
        F1, F2, Kp = map(unsqueeze, [F1, F2, Kp])
        F1, F2, Kp = map(repeat_interleave, [F1, F2, Kp])
        F1 = F1.view(bs, n1, n1)
        F2 = F2.view(bs, n2, n2)
        Kp = Kp.view(bs, n1, n2)
    else:
        F1, F2, Kp = unsqueeze_bs([F1, F2, Kp], bs)
    
    m1 = torch.mul(pred_X, torch.bmm(F1, torch.bmm(pred_X, F2)))
    m2 = torch.mul(pred_X, Kp)
    obj1 = torch.sum(m1, dim=(1, 2))
    obj2 = torch.sum(m2, dim=(1, 2))
    pred_obj = obj1 + obj2
    
    return pred_obj

def get_obj_LLQAP(K, pred_X):
    K, pred_X = list_to_tensor([K, pred_X])
    if len(pred_X.shape) == 2: pred_X = unsqueeze_bs(pred_X, 1)
    bs, n1, n2 = pred_X.shape
    K = unsqueeze_bs(K, bs)
    
    pred_obj = torch.bmm(pred_X.reshape(bs,1,-1), 
        torch.bmm(K, pred_X.reshape(bs,-1,1))).view(bs)
    return pred_obj   

def get_obj(pred_X, data, problem='KBQAP'):
    assert len(pred_X.shape) == 3, DEBUG(f'pred_X.shape {pred_X.shape}')
    if problem == 'KBQAP':
        pred_obj = get_obj_KBQAP(data['F1'], data['F2'], data['Kp'], pred_X)
    elif problem == 'LLQAP':
        pass
    else:
        assert False, DEBUG(f'Unknown problem {problem}')
    return pred_obj

def eval_obj(pred_X, data, maximize=True):
    bs, n1, n2 = pred_X.shape
    gt_X = data['gt_X']
    gt_obj = data['gt_obj']
    pred_objs = get_obj(pred_X, data)
    assert pred_objs.shape == (bs,), DEBUG(f'pred_objs.shape {pred_objs.shape}')
    
    F1, F2, Kp = data['F1'], data['F2'], data['Kp']
    
    if maximize:
        pred_obj, idx = torch.max(pred_objs, dim=0, keepdim=True)
        res = gt_obj - pred_obj
    else:
        pred_obj, idx = torch.min(pred_objs, dim=0, keepdim=True)
        res = pred_obj - gt_obj
         
    mae = torch.abs(res)
    mse = torch.pow(res, 2)
    
    acc = torch.sum(gt_X*pred_X, dim=-1)
    acc = torch.mean(acc, dim=-1)
    acc = torch.max(acc, dim=-1)[0]
    
    return {
        'pred_obj': pred_obj,
        'gt_obj': gt_obj,
        'res': res,
        'mae': mae,
        'mse': mse,
        'acc': acc,
    }
    
def KBQAP_norm(F1, F2, Kp):
    if len(Kp.shape) == 3:
        bs, n1, n2 = Kp.shape
        F1v_abs = torch.abs(F1).view(bs, -1)
        F2v_abs = torch.abs(F2).view(bs, -1)
        Kpv_abs = torch.abs(Kp).view(bs, -1)
        F1norm = torch.max(F1v_abs, dim=1)[0]
        F2norm = torch.max(F2v_abs, dim=1)[0]
        Kpnorm = F1norm * F2norm
        F1 = torch.div(F1, F1norm.view(bs, 1, 1))
        F2 = torch.div(F2, F2norm.view(bs, 1, 1))
        Kp = torch.div(Kp, Kpnorm.view(bs, 1, 1))
        return F1, F2, Kp
    elif len(Kp.shape) == 2:
        n1, n2 = Kp.shape
        F1v_abs = torch.abs(F1).view(-1)
        F2v_abs = torch.abs(F2).view(-1)
        Kpv_abs = torch.abs(Kp).view(-1)
        F1norm = torch.max(F1v_abs)
        F2norm = torch.max(F2v_abs)
        Kpnorm = F1norm * F2norm
        F1 = torch.div(F1, F1norm)
        F2 = torch.div(F2, F2norm)
        Kp = torch.div(Kp, Kpnorm)
        return F1, F2, Kp
    else:
        assert False, DEBUG(f'Error: Kp.shape {Kp.shape}')
    