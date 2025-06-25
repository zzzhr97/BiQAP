import torch.nn as nn

from .hungarian import Hungarian


def get_post_process(pp_cfg):
    _cfg = pp_cfg
    name = _cfg['name']
    
    if name == 'Hungarian':
        pp = Hungarian()
    elif name is None:
        pp = nn.Identity()
    else:
        raise ValueError(f'Post-process {name} not supported')
    
    return pp