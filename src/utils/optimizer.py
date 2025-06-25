import torch
import torch.nn as nn

def get_optimizer(optimizer_cfg, model):
    _cfg = optimizer_cfg
    name = _cfg['name']
    
    if name == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=_cfg['lr'], weight_decay=_cfg['wd'],
            betas=(_cfg['betas'][0], _cfg['betas'][1]))
    elif name == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=_cfg['lr'], weight_decay=_cfg['wd'], 
            momentum=_cfg['momentum'])
    else:
        raise ValueError(f'Optimizer {name} not supported')
    
    return optimizer