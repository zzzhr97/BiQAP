import torch
import torch.nn as nn

from src.models.BiQAP.gromov import GromovWasserstein
from src.utils.logger import INFO, DEBUG


class GWBaseModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        # compatibility with old models
        if not hasattr(self, "gromov"):
            self.gromov = GromovWasserstein()
        
    def forward(self, data):
        raise NotImplementedError
    
    def _set_param(self, setting):
        gromov_cfg = {
            "gromov_iters": setting["gromov_iters"] if "gromov_iters" in setting else 15,
            "psinkhorn_iters": setting["psinkhorn_iters"] if "psinkhorn_iters" in setting else 15,
            "psinkhorn_tau": setting["psinkhorn_tau"] if "psinkhorn_tau" in setting else 0.1,
            "norm": setting["norm"] if "norm" in setting else "Linf",
            "gumbel": setting["gumbel"] if "gumbel" in setting else False,
            "gumbel_size": setting["gumbel_size"] if "gumbel_size" in setting else 0,
            "gumbel_lambda": setting["gumbel_lambda"] if "gumbel_lambda" in setting else 0.05,
            "tau": 0.1,
            "epsilon": 0.1,
        }

        self.gromov._set_param(**gromov_cfg)