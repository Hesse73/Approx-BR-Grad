from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Literal

import math
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

# For DEBUG
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: "Tensor shape={}, grad_fn={}\n{}".format(tuple(self.shape), self.grad_fn, original_repr(self))


def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class KDE:

    def __init__(self, markets:torch.Tensor, kernel='gaussian'):
        self.markets = markets.transpose(-1,-2)  # N,Sj,1 -> N,1,Sj
        self.kernel = kernel
    
    @staticmethod
    def _get_func_exp(mode='pdf', h=0.05):
        if mode == 'pdf':
            func = lambda x: 0.5/h*torch.exp(-torch.abs(x)/h)
        elif mode == 'grad':
            func = lambda x: -torch.sign(x)*0.5/(h**2)*torch.exp(-torch.abs(x)/h)
        else:
            func = lambda x: torch.where(x>=0,1.0,0.0) #-torch.sign(x)*0.5*torch.exp(-torch.abs(x)/h)
        return func

    @staticmethod
    def _get_func_gaussian(mode='pdf', h=0.05):
        factor_ = 1/(math.sqrt(2*math.pi)*h)
        scale_ = -0.5/(h**2)
        if mode == 'pdf':
            func = lambda x: factor_ * torch.exp(scale_ * x**2)
        elif mode == 'cdf':
            func = lambda x: 0.5*(1 + torch.erf(x/(math.sqrt(2)*h)))
            # func = lambda x: torch.where(x>=0,1.0,0.0)
        else:
            func = lambda x: 2*factor_*scale_*x * torch.exp(scale_ * x**2)
        return func
        
    def get_pred(self, bids:torch.Tensor, mode='pdf', h=0.05) -> torch.Tensor:
        # bids: N,Si,1
        handle = self._get_func_gaussian if self.kernel == 'gaussian' else self._get_func_exp
        func = handle(mode, h)
        x = bids - self.markets  # N,Si,Sj
        pred = func(x)  # N,Si,Sj
        pred = pred.mean(dim=-1, keepdim=True)  # pdf/grad: N,Si,1
        return pred

    def plot(self, filename=None, pdf_h=0.05, grad_h=0.05):
        x = torch.linspace(0,1,1000,device=self.markets.device).unsqueeze(-1)
        n = self.markets.shape[0]
        pdf = self.get_pred(x.unsqueeze(0).repeat(n,1,1), h=pdf_h)
        pdf_grad = self.get_pred(x.unsqueeze(0).repeat(n,1,1), h=grad_h, mode='grad')
        fig, axs = plt.subplots(n,1,figsize=(4, 3*n))
        for idx, ax in enumerate(axs):
            ax.hist(self.markets[idx,0].cpu(), bins=100, color='tomato')
            ax = ax.twinx()
            ax.plot(x.cpu(), pdf[idx,:,0].cpu(), color='skyblue', label='pdf')
            ax.plot(x.cpu(), pdf_grad[idx,:,0].cpu(), color='royalblue', label='grad')
            ax.legend()
        plt.tight_layout()
        if filename is None:
            return fig
        else:
            plt.savefig(filename)
            plt.close()

