from .common_loss import *
from .uns_loss import *


def get_loss(loss_cfg):
    loss_name = loss_cfg['name']
    if loss_name == 'bce':
        return bce_loss
    elif loss_name == 'uns_KBQAP':
        return uns_KBQAP_loss
    else:
        raise NotImplementedError