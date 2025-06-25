import os
import os.path as osp
import torch

from .BiQAP import *
from src.utils.path import get_project_path
from src.utils.logger import INFO


PROJECT_PATH = get_project_path(3)

def get_model(model_cfg, model_setting_cfg, device):
    name = model_cfg['name']
    pretrained_path = model_cfg.get('pretrained_path', None)
    
    model = eval(name)(model_setting_cfg)
    if pretrained_path is not None:
        total_path = osp.join(PROJECT_PATH, pretrained_path)
        model.load_state_dict(torch.load(total_path, map_location=device))
        INFO(f'Pretrained model loaded from {total_path}')
        
    model = model.to(device)
    return model