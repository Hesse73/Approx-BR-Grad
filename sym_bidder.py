from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Literal

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torchvision.ops import MLP

import utils
from sym_prior import SymmetricPriors, SymmetricUniformPriors

class MLPBidNetwork(nn.Module):
    """MLP bid network with non-negative output constraint"""
    def __init__(self, hidden:List[int], activation:str):
        super().__init__()
        self.model = MLP(1, hidden + [1])
        if activation == 'abs':
            self.act = lambda x: torch.abs(x)
        elif activation == 'relu':
            self.act = lambda x: torch.relu(x)
        elif activation == 'exp':
            self.act = lambda x: torch.exp(x)
        else:
            raise NotImplementedError(f"Unknown Activation {activation}")
    def forward(self, x):
        return self.act(self.model(x))


class SymBidders:
    # default_reserve = 0.1
    default_aversion = 0.5

    def __init__(self, num_types:int, bidders_per_type: List[int], hidden:List[int], activation:str, device:torch.device) -> None:
        self.num_types = num_types
        self.bidders_per_types = bidders_per_type
        self.hidden = hidden
        self.device = device
        
        self.models = [MLPBidNetwork(hidden, activation).to(device) for _ in range(num_types)]

        self.N = sum(bidders_per_type)
        self.self_mask_nn = (torch.arange(self.N).unsqueeze(-1) == torch.arange(self.N)).to(device)


    def forward(self, vs:List[torch.Tensor]) -> torch.Tensor:
        """
        Given v~F [N,S,1], output b=(b_1, ..., b_n) [N,S,1]
        Args:
            vs (List[torch.Tensor]): value samples (Nt,S,1) for each type t
        Returns:
            torch.Tensor: concatenated bids (N,S,1)
        """
        bids: List[torch.Tensor] = [model(v) for model,v in zip(self.models, vs)]
        concated = torch.concat(bids, dim=0)
        return concated
    

    def market_bids_ipv(self, vs:List[torch.Tensor]) -> torch.Tensor:
        """Given value~F [N,S,1], output market price [N,S,1].
        Args:
            vs (List[torch.Tensor]): value samples (Nt,S,1) for each type t
        Returns:
            torch.Tensor: market prices (N,S,1)
        """
        bids = self.forward(vs)  # (Nj,S,1)
        N,S,_ = bids.shape
        # IPV model, just repeat each one's bids as condtioned distribution
        # repeat N times for each bidder Ni, then mask out self bid
        bids = bids.unsqueeze(0).repeat(N,1,1,1)  # (Ni,Nj,S,1)
        self_mask = self.self_mask_nn.view(N,N,1,1).repeat(1,1,S,1)  # (Ni,Nj,S,1)
        bids[self_mask] = -1
        market, _ = bids.max(dim=1)  # max over others' bids (Nj, dim=1)
        return market
    
    def smooth_grad(self, prior:SymmetricPriors, mechanism='fp', temperature=0.01,
                    interim_sample=10000, ante_sample=256, abr=False):
        """
        Implement the SM gradient estimation, allocation function is replaced with softmax
        (ICML'2023 Enabling First-Order Gradient-Based Learning for Equilibrium Computation in Markets)
        """
        # sample others bids
        with torch.no_grad():
            sampled_values = prior.sample(size=interim_sample)  # Tx[Nt,Sj,1]
            bids = self.forward(sampled_values)  # [Nj,Sj,1]
            other_bids = bids.unsqueeze(0).repeat(self.N,1,1,1)  # [Ni,Nj,Sj,1]
        # sample self values & bids
        sampled_values = prior.sample(size=ante_sample)  # Tx[Nt,Si,1]
        self_values:torch.Tensor = torch.concat(sampled_values, dim=0)  # Ni,Si,1
        self_bids = self.forward(sampled_values)  # Ni,Si,1
        # prepare
        full_self_mask = self.self_mask_nn.view(self.N,1,self.N,1,1).repeat(1,ante_sample,1,interim_sample,1)
        full_self_bids = self_bids.repeat(1,1,interim_sample)  # Ni,Si,Sj
        full_self_values = self_values.repeat(1,1,interim_sample)  # Ni,Si,Sj
        # create full bids sample, size: [Ni,Si,Nj,Sj,1]
        full_bids = other_bids.unsqueeze(1).repeat(1,ante_sample,1,1,1)  
        market_bids = full_bids.clone()
        full_bids[full_self_mask] = full_self_bids.flatten()  # set self bids values (requires_grad)
        market_bids[full_self_mask] = -1  # set market's self bids to -1 (mask out)
        market_price = market_bids.max(dim=2)[0].squeeze(-1)  # market price: Ni,Si,Sj
        # apply mechanism and get ex-post utility
        allocation = torch.softmax(full_bids/temperature, dim=2)  # soft-max on Nj dimension
        allocation = allocation[full_self_mask].view(self.N,ante_sample,interim_sample)  # Ni,Si,Sj
        if mechanism == 'fp':
            utility = allocation * (full_self_values - full_self_bids)
        elif mechanism == 'sp':
            utility = allocation * (full_self_values - market_price)
        # elif mechanism == 'allpay':
        #     utility = allocation * full_self_values - full_self_bids
        # elif mechanism == 'fp_res':
        #     utility = (full_self_bids >= self.default_reserve) * allocation * (full_self_values - full_self_bids)
        # elif mechanism == 'sp_res':
        #     utility = (full_self_bids >= self.default_reserve) * allocation * (full_self_values - market_price)

        # NOTE: SM cannot be directly applied for risk-aversion setting, since the ex-post utility is still not continuous
        # even after the soft-max allocation, so we adopt a small modification here.
        elif mechanism == 'fp_averse':
            payoff = full_self_values - full_self_bids
            masked_payoff = payoff.clone()
            masked_payoff[payoff<=0] = 1
            utility = torch.where(full_self_values>=full_self_bids, masked_payoff.pow(self.default_aversion), payoff)
            utility = allocation * utility
        elif mechanism == 'sp_averse':
            payoff = full_self_values - market_price
            masked_payoff = payoff.clone()
            masked_payoff[payoff<=0] = 1
            utility = torch.where(full_self_values>=full_self_bids, masked_payoff.pow(self.default_aversion), payoff)
            utility = allocation * utility
        else:
            raise NotImplementedError(f"Unknown mechanism {mechanism}")
        # ex-interim utility: avergaed ex-post utility
        interim_utility = utility.mean(dim=-1, keepdims=True)  # Ni,Si,1
        # calculate interim gradients
        # NOTE: autograd is slow !!! donnot compute second order unless required
        if abr:
            grad = torch.autograd.grad(interim_utility.sum(), inputs=self_bids, create_graph=True)[0]
            second_grad = torch.autograd.grad(grad.sum(), inputs=self_bids)[0]
            grad = grad.detach()
            second_grad = second_grad.detach()
        else:
            grad = torch.autograd.grad(interim_utility.sum(), inputs=self_bids)[0]
            second_grad = torch.ones_like(grad) * -1
        assert not grad.isnan().any()
        # print(self_values[0,:5,0], self_bids[0,:5,0], grad[0,:5,0])
        return self_bids, grad, second_grad

    
    def analytic_grad(self, prior:SymmetricPriors, mechanism='fp', kernel='gaussian', kde_h=0.05,
                      interim_sample=10000, ante_sample=256, abr=False):
        """
        Implement the proposed closed-form gradient estimation, pdf is estimated via KDE
        """
        # sample market distribution
        with torch.no_grad():
            sampled_values = prior.sample(size=interim_sample)  # Tx[Nt,Sj,1]
            market_bids = self.market_bids_ipv(sampled_values)  # Nj,Sj,1
        market_kde = utils.KDE(markets=market_bids, kernel=kernel)
        # sample self values
        sampled_values = prior.sample(size=ante_sample)  # Tx[Nt,Si,1]
        self_values:torch.Tensor = torch.concat(sampled_values, dim=0)  # Ni,Si,1
        self_bids = self.forward(sampled_values)  # Ni,Si,1
        # calculate gradients
        with torch.no_grad():
            # distributions: Ni,Si,1
            cdf = market_kde.get_pred(self_bids, mode='cdf', h=kde_h)
            pdf = market_kde.get_pred(self_bids, mode='pdf', h=kde_h)
            if abr:
                pdf_grad = market_kde.get_pred(self_bids, mode='grad', h=kde_h)
            # gradient for each mechanism
            if mechanism == 'fp':
                grad = (self_values-self_bids)*pdf - cdf
                if abr:
                    second_grad = (self_values-self_bids)*pdf_grad - 2*pdf
                else:
                    second_grad = torch.ones_like(grad) * -1
            elif mechanism == 'sp':
                grad = (self_values-self_bids)*pdf
                if abr:
                    second_grad = (self_values-self_bids)*pdf_grad - pdf
                else:
                    second_grad = torch.ones_like(grad) * -1
            # elif mechanism == 'allpay':
            #     grad = self_values*pdf - 1
            #     second_grad = self_values*pdf_grad
            # elif mechanism == 'fp_res':
            #     grad = torch.where(self_bids>=self.default_reserve, (self_values-self_bids)*pdf - cdf, 0.0)
            #     second_grad = torch.where(self_bids>=self.default_reserve, (self_values-self_bids)*pdf_grad - 2*pdf, 0.0)
            # elif mechanism == 'sp_res':
            #     grad = torch.where(self_bids>=self.default_reserve, (self_values-self_bids)*pdf, 0.0)
            #     second_grad = torch.where(self_bids>=self.default_reserve, (self_values-self_bids)*pdf_grad - pdf, 0.0)
            elif mechanism == 'fp_averse': 
                rho = self.default_aversion
                payoff = self_values - self_bids
                positive_mask = self_values > self_bids
                neg_grad = payoff*pdf - cdf
                pos_grad = payoff.pow(rho)*pdf - rho*payoff.pow(rho-1)*cdf
                grad = torch.where(positive_mask, pos_grad, neg_grad) 
                if abr:
                    neg_second_grad = payoff*pdf_grad - 2*pdf
                    pos_second_grad = payoff.pow(rho)*pdf_grad - 2*rho*payoff.pow(rho-1)*pdf + rho*(rho-1)*payoff.pow(rho-2)*cdf
                    second_grad = torch.where(positive_mask, pos_second_grad, neg_second_grad)
                else:
                    second_grad = torch.ones_like(grad) * -1
            elif mechanism == 'sp_averse': 
                rho = self.default_aversion
                payoff = self_values - self_bids
                positive_mask = self_values > self_bids
                neg_grad = payoff*pdf
                pos_grad = payoff.pow(rho)*pdf
                grad = torch.where(positive_mask, pos_grad, neg_grad) 
                if abr:
                    neg_second_grad = payoff*pdf_grad - pdf
                    pos_second_grad = payoff.pow(rho)*pdf_grad - rho*payoff.pow(rho-1)*pdf
                    second_grad = torch.where(positive_mask, pos_second_grad, neg_second_grad)
                else:
                    second_grad = torch.ones_like(grad) * -1
            else:
                raise NotImplementedError(f"Unknown mechanism {mechanism}")
        # print(self_values[0,:5,0], self_bids[0,:5,0], grad[0,:5,0])
        return self_bids, grad, second_grad


    def apply_grads(self, prior:SymmetricPriors, abr=False, analytic=True, mechanism='fp', 
                    interim_sample=10000, ante_sample=256, kernel='gaussian', kde_h=0.05, 
                    second_cond=1e-9, first_cond=1e-3, temperature=0.01):
        """Estimate gradient-based loss for bidding network"""
        if analytic:
            bids, grad, second_grad = self.analytic_grad(prior, mechanism, kernel, kde_h, interim_sample, ante_sample, abr)
        else:
            bids, grad, second_grad = self.smooth_grad(prior, mechanism, temperature, interim_sample, ante_sample, abr)
        # clear grads
        for model in self.models: model.zero_grad()
        # backward
        abr_condition = abr & (second_grad < -second_cond) & (grad.abs() < first_cond)
        bids_grad = torch.where(abr_condition, grad/second_grad, -grad)
        bids_grad *= 1/bids.numel()  # empirical gradient
        bids.backward(bids_grad)

    @torch.no_grad()
    def plot_special(self, prior:SymmetricPriors, filename=None):
        import matplotlib.pyplot as plt
        # plot the asymmetric BNE
        a,b = prior.ranges[0][1], prior.ranges[1][1]
        names = ['Weak Bidder', 'Strong Bidder']
        if a > b: names = [names[1], names[0]]
        x1 = torch.linspace(0,a,steps=1000)
        x2 = torch.linspace(0,b,steps=1000)
        k1 = 1/a**2-1/b**2
        k2 = -k1
        y1 = 1/(k1*x1)*(1-torch.sqrt(1-k1*x1**2))
        y2 = 1/(k2*x2)*(1-torch.sqrt(1-k2*x2**2))
        plt.plot(x1,y1,'--',label=f'BNE: {names[0]}')
        plt.plot(x2,y2,'--',label=f'BNE: {names[1]}')
        markers = ['v','^']
        steps = [20, 30]
        for idx, (lower,upper) in enumerate(prior.ranges):
            x = torch.linspace(lower,upper,steps=steps[idx],device=self.device).unsqueeze(-1)  # S,1
            y = self.models[idx](x)  # S,1
            plt.scatter(x.cpu(),y.cpu(),s=100,marker=markers[idx],label=f'{names[idx]}')
        plt.legend()
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xlabel(r'Values $v_i$')
        plt.ylabel(r'Bids $b_i$')
        plt.title('Learned Bidding Strategies of 2 Bidders with Asymmetric Prior')
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close()


    @torch.no_grad()
    def plot_bidders(self, prior:SymmetricPriors, mechanism='fp', filename=None):
        import matplotlib.pyplot as plt
        x = torch.linspace(0,1,steps=1000)
        if self.num_types == 1 or mechanism == 'sp':
            y = self.get_sym_bne(mechanism, self.N)(x)
            plt.plot(x,y,'--',label='gt')
        for idx, (lower,upper) in enumerate(prior.ranges):
            x = torch.linspace(lower,upper,steps=1000,device=self.device).unsqueeze(-1)  # S,1
            y = self.models[idx](x)  # S,1
            plt.plot(x.cpu(),y.cpu(),label=f'Type-{idx}')
        if mechanism == 'sp':
            plt.ylim(0,1)
            plt.xlim(0,1)
        plt.legend()
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close()

    @torch.no_grad()
    def evaluate_sym_l2(self, prior:SymmetricPriors, mechanism='fp', size=10000) -> Union[float, List[float]]:
        if self.num_types != 1 and mechanism != 'sp':
            return -1
        l2_per_type = []
        for idx in range(self.num_types):
            x = torch.linspace(*prior.ranges[idx],steps=size,device=self.device)  # S,
            y_gt:torch.Tensor = self.get_sym_bne(mechanism, self.N)(x)  # S,
            y_pred:torch.Tensor = self.models[idx](x.unsqueeze(-1)).flatten()  # S,
            l2 = (y_gt - y_pred).square().mean().sqrt().item()
            l2_per_type.append(l2)
        if self.num_types == 1:
            return l2_per_type[0]
        else:
            return l2_per_type
        # x = torch.linspace(0,1,steps=size,device=self.device)  # S,
        # y_gt:torch.Tensor = self.get_sym_bne(mechanism, self.N)(x)  # S,
        # y_pred:torch.Tensor = self.models[0](x.unsqueeze(-1)).flatten()  # S,
        # L2 = (y_gt - y_pred).square().mean().sqrt().item()
        # return L2
    
    # @torch.no_grad()
    # def approx_local_l2(self, prior:SymmetricPriors, mechanism='fp', gridsize=1000, interim_sample=10000) -> float:
    #     N,Si,Sj = self.N, gridsize, interim_sample
    #     # sample others market price
    #     sampled_values = prior.sample(size=interim_sample)  # Tx[Nt,Sj,1]
    #     market_price = self.market_bids_ipv(sampled_values)  # Ni,Sj,1
    #     bidder_index = 0
    #     l2_per_type = []
    #     for idx in range(self.num_types):
    #         # get market price
    #         cur_market_price = market_price[bidder_index].view(1,1,Sj,1)  # 1,1,Sj,1
    #         # get self v/bids
    #         vs = torch.linspace(*prior.ranges[idx],steps=Si).unsqueeze(-1).to(self.device)  # VS,1
    #         bs = vs.T  # 1,BS
    #         self_bids = self.models[idx](vs)  # VS,1
    #         # calculate best bids
    #         grid_vs = vs.repeat(1,Si).view(Si,Si,1,1)  # VS,BS,1,1
    #         grid_bids = bs.repeat(Si,1).view(Si,Si,1,1)  # VS,BS,1,1
    #         allocation = grid_bids > cur_market_price  # VS,BS,Sj,1
    #         utility = allocation * (grid_vs - grid_bids)  # VS,BS,Sj,1
    #         interim_utility = utility.mean(dim=-2)  # VS,BS,1
    #         best_bids_index = interim_utility.argmax(dim=1)  # VS,1
    #         # VS,BS at VS,1 -> VS,1
    #         best_bids = torch.gather(grid_bids.squeeze(),dim=1,index=best_bids_index)
    #         # calculate l2
    #         l2 = (self_bids - best_bids).square().mean().sqrt().item()
    #         l2_per_type.append(l2)
    #         bidder_index += self.bidders_per_types[idx]
    #     return l2_per_type


    @classmethod
    def get_sym_bne(self, mechanism='fp', n=2):
        if mechanism == 'fp':
            return lambda x: (n-1)/n*x
        elif mechanism == 'sp' or mechanism == 'sp_averse':
            return lambda x: x
        # elif mechanism == 'allpay':
        #     return lambda x: (n-1)/n*(x**n)
        # elif mechanism == 'fp_res':
        #     def fp_reserve(x:torch.Tensor):
        #         r = self.default_reserve
        #         bid = (1-1/n)*x + (x/n)*(r/x)**n
        #         return torch.where(x>=r, bid, r)
        #     return fp_reserve
        # elif mechanism == 'sp_res':
        #     return lambda x: torch.where(x>=self.default_reserve,x,self.default_reserve)
        elif mechanism == 'fp_averse':
            return lambda x: (1-1/(2*n-1))*x
        else:
            raise NotImplementedError(f"Unknown mechanism {mechanism}")


# if __name__ == '__main__':
#     from tqdm import tqdm
#     import argparse
#     import os
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--init_steps', type=int, default=200)
#     parser.add_argument('--abr', action='store_true')
#     parser.add_argument('--analytic', action='store_true')
#     parser.add_argument('--lr', type=float, default=0.02)
#     args = parser.parse_args()
#     using_abr = '_abr' if args.abr else ''
#     grad = 'analytic' if args.analytic else 'sm'
#     dir_name = f'figs/test_{grad}{using_abr}/'
#     if not os.path.exists(dir_name): os.makedirs(dir_name)
#     num_types, bidder_per_type, ranges = 2, [5,5], [(0,1), (0, 0.5)]
#     max_steps = 1000
#     utils.set_seed(42)
#     device = device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     prior = SymmetricUniformPriors(num_types,bidder_per_type,ranges, device)
#     bidders = SymBidders(num_types,bidder_per_type, [32,32], device=device)
#     optims = [SGD(model.parameters(), lr=args.lr) for model in bidders.models]
#     titer = tqdm(range(1,max_steps+1))
#     best_l2 = 100
#     for step in titer:
#         # compute grad
#         abr = True if args.abr and step >= args.init_steps else False
#         bidders.apply_grads(prior, analytic=args.analytic, abr=abr, kde_h=0.05)
#         # update
#         for optim in optims:
#             optim.step()
#         if step % 100 == 0:
#             bidders.plot_bidders(prior, filename=os.path.join(dir_name, f'test-{step}.png'))
#         l2 = bidders.evaluate_sym_l2(mechanism='fp')
#         if l2 < best_l2:
#             best_l2 = l2
#         titer.set_postfix_str(f"L2={best_l2:.4e}")