import torch
import numpy as np
import random
from src.utils.logger import INFO, DEBUG 


def unsqueeze_bs(X, bs):
    if type(X) == torch.Tensor:
        if len(X.shape) == 3:
            if X.shape[0] == 1: X = X.repeat(bs, 1, 1)
            elif X.shape[0] == bs: pass
            else: assert False, DEBUG(f'Unsqueeze BS Error: X.shape = {X.shape}')
        elif len(X.shape) == 2:
            X = X.unsqueeze(0).expand(bs, -1, -1)
        elif len(X.shape) == 4:
            if X.shape[0] == 1: X = X.repeat(bs, 1, 1, 1)
            elif X.shape[0] == bs: pass
            else: assert False, DEBUG(f'Unsqueeze BS Error: X.shape = {X.shape}')
        else:
            assert False, DEBUG(f'Unsqueeze BS Error: X.shape = {X.shape}')
    elif type(X) == np.ndarray:
        if len(X.shape) == 3:
            if X.shape[0] == 1: return np.repeat(X, bs, axis=0)
            elif X.shape[0] == bs: pass
            else: assert False, DEBUG(f'Unsqueeze BS Error: X.shape = {X.shape}')
        elif len(X.shape) == 2:
            X = np.expand_dims(X, axis=0).repeat(bs, axis=0)
        elif len(X.shape) == 4:
            if X.shape[0] == 1: return np.repeat(X, bs, axis=0)
            elif X.shape[0] == bs: pass
            else: assert False, DEBUG(f'Unsqueeze BS Error: X.shape = {X.shape}')

    elif type(X) == list or type(X) == tuple:
        X = [unsqueeze_bs(_, bs) for _ in X]
    else:
        assert False, DEBUG(f'Unsqueeze BS Error: type(X) = {type(X)}')
    return X

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)