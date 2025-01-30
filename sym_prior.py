from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Literal

import torch

class SymmetricPriors:
    num_types: int
    bidders_per_type: List[int]
    ranges: List[Tuple[float, float]]
    def sample(self, size:int) -> torch.Tensor: pass


class SymmetricUniformPriors(SymmetricPriors):

    def __init__(self, num_types:int, bidders_per_type: List[int], ranges:List[Tuple[float,float]], device:torch.device) -> None:
        self.num_types = num_types
        self.bidders_per_type = bidders_per_type
        self.ranges = ranges
        self.device = device
        # build the distributions
        self.dists = []
        for num_bidder, range_ in zip(self.bidders_per_type, self.ranges):
            vec_range = [torch.tensor([value]*num_bidder, device=device).float() for value in range_]
            self.dists.append(torch.distributions.Uniform(*vec_range))

    def sample(self, size:int) -> List[torch.Tensor]:
        """
        Args:
            size (int): sample size S
        Returns:
            List[torch.Tensor]: sampled values for each distribution: [Nt,S,1] for each type t
        """
        value_samples = [dist.sample([size]).T.unsqueeze(-1) for dist in self.dists]
        return value_samples
    

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    prior = SymmetricUniformPriors(2, [3,1], ranges=[(0,1), (-2,-1)], device=device)
    samples = prior.sample(5)
    print(torch.concat(samples, dim=0).squeeze(-1))