import os
import argparse
import torch
import json
import time
import numpy as np
from tqdm import tqdm
from torch.optim import SGD
from collections import defaultdict

import utils
from sym_bidder import SymBidders
from sym_prior import SymmetricUniformPriors



parser = argparse.ArgumentParser()
# game setting
parser.add_argument('--n', type=int, default=2)
parser.add_argument('--mechanism', type=str, default='fp', choices=['fp', 'sp', 'fp_averse', 'sp_averse'])
# learning setting
parser.add_argument('--activation', type=str, default='abs', choices=['abs', 'relu', 'exp'], help='Activation to ensure non-negative bids')
parser.add_argument('--hidden', type=int, nargs='+', default=[32,32])
parser.add_argument('--max_steps', type=int, default=2000)
parser.add_argument('--record_freq', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--interim_sample', type=int, default=10000)
parser.add_argument('--ante_sample', type=int, default=256)
# method setting
parser.add_argument('--abr', action='store_true', help='To enable the approx-BR gradient')
parser.add_argument('--analytic', action='store_true', help='To enable the analytic gradient estimation')
parser.add_argument('--temperature', type=float, default=0.05, help='Smooth temperature, default 0.01 according to SM paper')
parser.add_argument('--kde_h', type=float, default=0.05, help='KDE h parameter')
parser.add_argument('--kernel', type=str, default='gaussian', choices=['gaussian', 'exponential'], help='KDE h parameter')
parser.add_argument('--init_steps', type=int, default=200, help='Initial steps w/o aBR')
parser.add_argument('--first_cond', type=float, default=0.01, help='First-order condition to apply aBR')
parser.add_argument('--second_cond', type=float, default=1e-8, help='Second-order condition to apply aBR')

random_init = 5

if __name__ == '__main__':
    args = parser.parse_args()
    device = device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    grad_method = 'Analytic' if args.analytic else 'SM'
    opt_method = 'aBR' if args.abr else 'UG'
    dir_name = f'figs/{args.mechanism}_{args.n}/{grad_method}-{opt_method}/'
    if not os.path.exists(dir_name): os.makedirs(dir_name)
    candidates = [0.01,0.02,0.04,0.05,0.1]  # [0.01,0.02,0.04,0.08,0.1]
    best_avg_l2 = 1e6
    for hyper_param in candidates:
        if args.analytic: args.kde_h = hyper_param
        else: args.temperature = hyper_param
        l2_records = defaultdict(list)
        sum_time = 0.0 
        for seed in range(random_init):
            # define the game
            utils.set_seed(seed)
            num_types, bidders_per_type= 1, [args.n]
            prior = SymmetricUniformPriors(num_types, bidders_per_type, ranges=[(0,1)], device=device)
            bidders = SymBidders(num_types, bidders_per_type, args.hidden, args.activation, device)
            # set optimizer for each type (only 1 type here actually)
            optims = [SGD(model.parameters(), lr=args.lr) for model in bidders.models]
            titer = tqdm(range(1, args.max_steps+1), desc=f'Seed {seed}')
            for step in titer:
                # apply gradients
                abr = True if args.abr and step > args.init_steps else False
                start = time.time()
                bidders.apply_grads(prior=prior, abr=abr, analytic=args.analytic, mechanism=args.mechanism, 
                                    interim_sample=args.interim_sample, ante_sample=args.ante_sample, kernel=args.kernel, kde_h=args.kde_h,
                                    second_cond=args.second_cond, first_cond=args.first_cond, temperature=args.temperature)
                # update parameters
                for optimizer in optims: optimizer.step()
                end = time.time()
                # evaluate L2 & record
                l2 = bidders.evaluate_sym_l2(prior=prior, mechanism=args.mechanism)
                titer.set_postfix_str(f"L2={l2:.4e}")
                if step % args.record_freq == 0:
                    l2_records[step].append(l2)
                sum_time += end-start
            l2_records['last_l2'].append(l2)
        avg_time = sum_time / (random_init * args.max_steps)
        # get average l2 for each random initialization
        for key in l2_records: l2_records[key] = np.array(l2_records[key]).mean()
        avg_last_l2 = l2_records['last_l2']
        if avg_last_l2 < best_avg_l2:
            # save best figure & dump records
            best_avg_l2 = avg_last_l2
            for key in l2_records: l2_records[key] = '%.3e' % l2_records[key]
            l2_records['time/iter'] = avg_time
            json.dump(l2_records, open(os.path.join(dir_name, 'l2_records.json'), 'w'), indent=2)
            bidders.plot_bidders(prior, mechanism=args.mechanism, filename=os.path.join(dir_name, f'last_strategy.pdf'))
