import torch
import torch.nn as nn

def get_scheduler(scheduler_cfg, optimizer):
    _cfg = scheduler_cfg
    name = _cfg['name']
    
    if name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=_cfg['step_size'], gamma=_cfg['gamma'])
    elif name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=_cfg['milestones'], gamma=_cfg['gamma'])
    elif name == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=_cfg['gamma'])
    elif name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=_cfg['mode'], factor=_cfg['factor'], patience=_cfg['patience'], 
            threshold=_cfg['threshold'], threshold_mode=_cfg['threshold_mode'], 
            cooldown=_cfg['cooldown'], min_lr=_cfg['min_lr'], eps=_cfg['eps'])
    else:
        raise ValueError(f'Scheduler {name} not supported')
    
    return scheduler