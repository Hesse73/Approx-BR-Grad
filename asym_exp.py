from sym_exp import *
import seaborn as sns
sns.set_style('whitegrid')

random_init = 5
if __name__ == '__main__':
    args = parser.parse_args()
    device = device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    grad_method = 'Analytic' if args.analytic else 'SM'
    opt_method = 'aBR' if args.abr else 'UG'
    dir_name = f'figs/asym_{args.mechanism}_{args.n}/{grad_method}-{opt_method}/'
    if not os.path.exists(dir_name): os.makedirs(dir_name)
    ranges = [(0,0.5), (0,1)]
    num_types, bidders_per_type= 2, [args.n,args.n]
    sum_l2 = [0.0] * num_types
    for seed in range(random_init):
        utils.set_seed(seed)
        prior = SymmetricUniformPriors(num_types, bidders_per_type, ranges, device=device)
        bidders = SymBidders(num_types, bidders_per_type, args.hidden, args.activation, device)
        # set optimizer for each type (only 1 type here actually)
        optims = [SGD(model.parameters(), lr=args.lr) for model in bidders.models]
        titer = tqdm(range(1, args.max_steps+1))
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
            # evaluate l2
            l2_per_type = bidders.evaluate_sym_l2(prior, mechanism=args.mechanism)
            titer.set_postfix_str(f'l2:{l2_per_type[0]:.4e}, {l2_per_type[1]:.4e}')
            if step % args.record_freq == 0:
                bidders.plot_bidders(prior, mechanism=args.mechanism, filename=os.path.join(dir_name, f'{step}.png'))
        sum_l2 = [a+b for a,b in zip(sum_l2, l2_per_type)]
    avg_l2 = ['%.3e'% (x/random_init) for x in sum_l2]
    json.dump(avg_l2, open(os.path.join(dir_name, 'l2_records.json'), 'w'))
        