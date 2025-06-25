import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from mamba_ssm import Mamba
from functools import partial

from src.models.BiQAP.gromov import GromovWasserstein
from src.models.BiQAP.basemodel_GW import GWBaseModel
from src.utils.utils import unsqueeze_bs
from src.utils.problem import KBQAP_norm
from src.utils.logger import INFO, DEBUG
from src.utils.state import STATE


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2, ln="11", use_chk=False):
        super().__init__()
        self.dim = dim
        self.use_chk = use_chk
        
        assert len(ln) == 2
        if ln[0] == '1': self.ln1 = nn.LayerNorm(dim, eps=1e-6)
        else: self.ln1 = nn.Identity()
        if ln[1] == '1': self.ln2 = nn.LayerNorm(dim, eps=1e-6)
        else: self.ln2 = nn.Identity()
        
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor, typically 64 or 128
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
        )
        
    def forward(self, x):
        """
        Input: x: (bs, n1, n2, d)
        Output: x: (bs, n1, n2, d)
        """
        assert len(x.shape) == 4 and x.shape[-1] == self.dim, DEBUG(f"Input shape should be (bs, n1, n2, {self.dim}), but got {x.shape}")
        bs, n1, n2, d = x.shape
        
        x_norm = self.ln1(x)
        x1 = x_norm.view(bs, n1*n2, d)
        x2 = x_norm.permute(0, 2, 1, 3).contiguous().view(bs, n1*n2, d)
        xm = torch.concat([x1, x2], dim=0)
        if STATE['train'] and self.use_chk:
            xm = checkpoint(self.mamba, xm)
        else:
            xm = self.mamba(xm)
        
        x1, x2 = self.ln2(xm).split(bs, dim=0)
        x1 = x1.view(bs, n1, n2, d)
        x2 = x2.view(bs, n2, n1, d).permute(0, 2, 1, 3).contiguous()
        
        x = x + x1 + x2

        return x
    
class MatrixBlock(nn.Module):
    def __init__(self,
        block_dim=64, 
        block_num=4,
        d_state=64, 
        d_conv=4, 
        expand=2,
        ln="11",
        use_chk=False
    ):
        super().__init__()
        self.block_dim = block_dim
        
        self.layer = nn.ModuleList()
        for i in range(block_num):
            self.layer.append(MambaBlock(
                dim=block_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                ln=ln,
                use_chk=use_chk
            ))
        
    def forward(self, x):
        """
        Input: x: (bs, n1, n2, d)
        Output: x: (bs, n1, n2, d)
        """
        assert len(x.shape) == 4, DEBUG(f"Input shape should be (bs, n1, n2, {self.block_dim}), but got {x.shape}")
        bs, n1, n2, d = x.shape
        
        proj_x = x
        for layer in self.layer:
            proj_x = layer(proj_x)
        x = proj_x

        return x
    

class MatrixModule(nn.Module):
    def __init__(self,
        block_dim=4,
        block_nums=[4, 4, 4, 4],
        d_state=64,
        d_conv=4,
        expand=2,
        ln="11",
        use_chk=False
    ): 
        super().__init__()
        
        self.proj_in = nn.Linear(1, block_dim)
        self.proj_out = nn.Linear(block_dim, 1)
        
        self.layers = nn.ModuleList()
        for i in range(len(block_nums)):
            self.layers.append(MatrixBlock(
                block_dim=block_dim,
                block_num=block_nums[i],
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                ln=ln,
                use_chk=use_chk
            ))
            
    def forward(self, x):
        """
        Input: x: (bs, n1, n2)
        Output: x: (bs, n1, n2)
        """
        assert len(x.shape) == 3, DEBUG(f"Input shape should be (bs, n1, n2), but got {x.shape}")
        bs, n1, n2 = x.shape
        
        x_proj = self.proj_in(x.unsqueeze(-1)).view(bs, n1, n2, -1)
        for layer in self.layers:
            x_proj = layer(x_proj)
        x_proj = self.proj_out(x_proj).view(bs, n1, n2)
        x = x + x_proj
        
        return x


class Backbone(nn.Module):
    def __init__(self, model_setting_cfg):
        super().__init__()
        self._cfg = model_setting_cfg
        self.init_layers()
        
    def init_layers(self):
        block_nums = self._cfg['block_nums'] if 'block_nums' in self._cfg else [4, 4, 4, 4]
        block_dim = self._cfg['block_dim'] if 'block_dim' in self._cfg else 64
        d_state = self._cfg['d_state'] if 'd_state' in self._cfg else 64
        d_conv = self._cfg['d_conv'] if 'd_conv' in self._cfg else 4
        expand = self._cfg['expand'] if 'expand' in self._cfg else 2
        ln = self._cfg['ln'] if 'ln' in self._cfg else "11"
        use_chk = self._cfg['use_chk'] if 'use_chk' in self._cfg else False
        
        self.blocks = nn.ModuleDict()
        block_id = ['F1', 'F2', 'Kp']
        for id in block_id:
            self.blocks[id] = MatrixModule(
                block_nums=block_nums,
                block_dim=block_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                ln=ln,
                use_chk=use_chk
            )
    
    def forward(self, F1, F2, Kp, sbs):
        """
        Output: (bs, n1, n2)
        """
        bs, n1, n2 = Kp.shape
        
        F1 = self.blocks['F1'](F1)
        F2 = self.blocks['F2'](F2)
        Kp = self.blocks['Kp'](Kp)
        
        unsqueeze = partial(torch.unsqueeze, dim=1)
        repeat_interleave = partial(torch.repeat_interleave, repeats=sbs, dim=1)
        F1, F2, Kp = map(unsqueeze, [F1, F2, Kp])
        F1, F2, Kp = map(repeat_interleave, [F1, F2, Kp])
        
        X = torch.zeros(bs, sbs, n1, n2, device=Kp.device, requires_grad=True)
        
        return X, F1, F2, Kp

class BiQAP(GWBaseModel):
    def __init__(self, model_setting_cfg):
        super().__init__(model_setting_cfg)
        self.backbone = Backbone(model_setting_cfg['backbone'])
        
    def forward(self, data, **kwargs):
        F1, F2, Kp, sbs = data['F1'], data['F2'], data['Kp'], data['sbs']
        # sbs == 1
        bs, n1, n2 = Kp.shape
        
        F1, F2, Kp = KBQAP_norm(F1, F2, Kp)
        X, F1, F2, Kp = self.backbone(F1, F2, Kp, sbs)
        
        view_before = partial(torch.Tensor.view, size=(bs*sbs, n1, n2))
        F1 = F1.view(bs*sbs, n1, n1)
        F2 = F2.view(bs*sbs, n2, n2)
        X, Kp = map(view_before, [X, Kp])
        assign_slk = self.gromov(X, F1, F2, Kp)
        
        return assign_slk