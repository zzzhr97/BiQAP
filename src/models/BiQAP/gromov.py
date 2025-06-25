import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from src.utils.logger import INFO, DEBUG


def Linf_normalize(X):
    Xv_abs = torch.abs(X.view(X.shape[0], -1))
    norm = torch.max(Xv_abs, dim=1, keepdim=True)[0].unsqueeze(-1)
    return torch.div(X, norm)

def sample_gumbel(x, eps=1e-20):
    u = torch.empty_like(x, device=x.device, dtype=x.dtype).uniform_()
    return -torch.log(-torch.log(u + eps) + eps)

class PartialSinkhorn(nn.Module):
    def __init__(self, max_iters=20, tau=0.1):
        super(PartialSinkhorn, self).__init__()
        self.max_iters = max_iters
        self.tau = tau

    def forward(self, X, exp=True):
        assert len(X.shape) == 3, f'Error: x.shape = {X.shape}'
        if X.shape[1] > X.shape[2]: X = X.transpose(1, 2)
        
        bs, n1, n2 = X.shape
        # va = torch.ones(bs, n1, 1, device=X.device)
        vb = torch.ones(bs, n2, 1, device=X.device)
        rv = torch.ones(bs, n2, 1, device=X.device)
        
        if exp:
            X_exp = torch.exp( -X / self.tau )
        else:
            X_exp = X
        
        for i in range(self.max_iters):
            lu = torch.reciprocal(torch.bmm(X_exp, rv))
            rv = torch.reciprocal(torch.bmm(X_exp.transpose(1, 2), lu))
            rv = torch.min(rv, vb)
        
        lu_diag = torch.diag_embed(lu.squeeze(-1))
        rv_diag = torch.diag_embed(rv.squeeze(-1))
        X = torch.bmm(lu_diag, torch.bmm(X_exp, rv_diag))
        return X
    
    def _set_param(self, max_iters, tau):
        self.max_iters = max_iters
        self.tau = tau
        
class Gumbel(nn.Module):
    def __init__(self):
        super(Gumbel, self).__init__()
        # self.psinkhorn = PartialSinkhorn(max_iters=max_iters, tau=tau)
        
    def forward(self, X, gumbel_size=1, gumbel_lambda=0.05):
        assert len(X.shape) == 3, f'Error: x.shape = {X.shape}'
        if X.shape[1] > X.shape[2]: X = X.transpose(1, 2)
        
        X_rep = torch.repeat_interleave(X, gumbel_size, dim=0)
        X_gum = X_rep + sample_gumbel(X_rep) * gumbel_lambda
        # X_gum = self.psinkhorn(X_gum, exp=exp)
        assert X_gum.shape[0] == X.shape[0] * gumbel_size
        return X_gum
    
class GromovWasserstein(nn.Module):
    NORM_LAYERS = {
        'Linf': Linf_normalize,
    }
    def __init__(self, 
        gromov_iters=15,
        psinkhorn_iters=15,
        psinkhorn_tau=0.1,
        norm='Linf',    # from ['Linf', 'L1', 'L2']
        gumbel=False,
        gumbel_size=1,
        gumbel_lambda=0.05,
    ):
        super(GromovWasserstein, self).__init__()
        self.psinkhorn = PartialSinkhorn(max_iters=psinkhorn_iters, tau=psinkhorn_tau)
        self.gumbel_layer = Gumbel()
        self._set_param(gromov_iters, psinkhorn_iters, psinkhorn_tau, norm, gumbel, gumbel_size, gumbel_lambda)
        
    def gromov_iteration(self, X, F1, F2, Kp, gumbel_size=0, gumbel_lambda=0.002, init=False):
        """General gromov iteration."""
        
        exp = True
        
        if self.gumbel and gumbel_size > 0:
            X = self.gumbel_layer(X, gumbel_size=gumbel_size, gumbel_lambda=gumbel_lambda)
            if gumbel_size > 1:
                repeat = partial(torch.repeat_interleave, repeats=gumbel_size, dim=0)
                Kp, F1, F2 = map(repeat, [Kp, F1, F2])
                
        C = Kp + torch.bmm(torch.bmm(F1, X), F2)
        C = C + torch.bmm(torch.bmm(F1.permute(0, 2, 1).contiguous(), X), F2.permute(0, 2, 1).contiguous())
        C = -C
        C = self.norm_layer(C)
            
        X = self.psinkhorn(C, exp=exp)
        return X, F1, F2, Kp

    def forward(self, X, F1, F2, Kp):
        assert len(X.shape) == len(F1.shape) == len(F2.shape) == len(Kp.shape) == 3, \
            f'Error: x.shape = {X.shape}, F1.shape = {F1.shape}, F2.shape = {F2.shape}, Kp.shape = {Kp.shape}. They should be 3D.'
        bs, n1, n2 = X.shape
        assert F1.shape == (bs, n1, n1), f'Error: F1.shape = {F1.shape}, expected {(bs, n1, n1)}'
        assert F2.shape == (bs, n2, n2), f'Error: F2.shape = {F2.shape}, expected {(bs, n2, n2)}'
        assert Kp.shape == (bs, n1, n2), f'Error: Kp.shape = {Kp.shape}, expected {(bs, n1, n2)}'

        for i in range(self.giters):
            X, F1, F2, Kp = self.gromov_iteration(X, F1, F2, Kp, gumbel_size=self.gumbel_size[i], gumbel_lambda=self.gumbel_lambda[i])
 
        return X
    
    def _set_param(self, gromov_iters, psinkhorn_iters, psinkhorn_tau, norm, gumbel, gumbel_size, gumbel_lambda, **kwargs):
        self.giters = gromov_iters
        assert norm in self.NORM_LAYERS, f'Error: norm = {norm}'
        self.norm_layer = self.NORM_LAYERS[norm]
        self.gumbel = gumbel
        self.gumbel_size = gumbel_size
        self.gumbel_lambda = gumbel_lambda
        
        DEBUG(f"Gromov: giters {gromov_iters} | piters {psinkhorn_iters} | ptau {psinkhorn_tau} | norm {norm} | gumbel {gumbel} | gsize {gumbel_size} | glambda {gumbel_lambda}")
        
        if type(gumbel_size) != list:
            self.gumbel_size = [gumbel_size] * gromov_iters
        if type(gumbel_lambda) != list:
            self.gumbel_lambda = [gumbel_lambda] * gromov_iters
        
        self.psinkhorn._set_param(psinkhorn_iters, psinkhorn_tau)