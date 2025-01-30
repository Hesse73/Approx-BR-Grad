from sym_exp import *
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.set_style('whitegrid')

random_init = 5
jump = 5

if __name__ == '__main__':
    # set default args
    args = parser.parse_args('')
    args.mechanism = 'fp'
    args.analytic = False
    args.max_steps = 1000
    args.temperature = 0.02
    args.first_cond = 0.01
    args.lr = 0.04
    print(args.lr, args.temperature)
    device = device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for n in [10]:
        args.n = n
        dir_name = f'figs/SM_augment/{args.mechanism}_{args.n}'
        if not os.path.exists(dir_name): os.makedirs(dir_name)
        l2_records = []
        # run SM/SM+aBR with 5 initialization, record the L2
        for name in ['SM+aBR', 'SM']:
            args.abr = False if name == 'SM' else True
            best_l2 = 100
            for seed in range(random_init):
                utils.set_seed(seed)
                # define the game
                num_types, bidders_per_type= 1, [args.n]
                prior = SymmetricUniformPriors(num_types, bidders_per_type, ranges=[(0,1)], device=device)
                bidders = SymBidders(num_types, bidders_per_type, args.hidden, args.activation, device)
                # set optimizer for each type (only 1 type here actually)
                optims = [SGD(model.parameters(), lr=args.lr) for model in bidders.models]
                titer = tqdm(range(1, args.max_steps+1), desc=f'Seed {seed}')
                for step in titer:
                    # apply gradients
                    abr = True if args.abr and step > args.init_steps else False
                    bidders.apply_grads(prior=prior, abr=abr, analytic=args.analytic, mechanism=args.mechanism, 
                                        interim_sample=args.interim_sample, ante_sample=args.ante_sample, kernel=args.kernel, kde_h=args.kde_h,
                                        second_cond=args.second_cond, first_cond=args.first_cond, temperature=args.temperature)
                    # update parameters
                    for optimizer in optims: optimizer.step()
                    # evaluate L2 & record
                    l2 = bidders.evaluate_sym_l2(prior=prior, mechanism=args.mechanism)
                    # record Method, Run, Step, L2
                    if step % jump == 0:
                        l2_records.append([name, seed, step, l2])
                    titer.set_postfix_str(f"L2:{l2:.3e}")
                if l2 < best_l2:
                    best_l2 = l2
                    torch.save(bidders.models[0].cpu().state_dict(), os.path.join(dir_name, f'{name}.ckpt'))
        df = pd.DataFrame(l2_records, columns=['Method', 'Run', 'Step', 'L2'])
        df.to_csv(os.path.join(dir_name, 'l2_data.csv'))
    from sym_bidder import MLPBidNetwork
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')
    x = torch.linspace(0,1,1000).unsqueeze(-1)
    plt.plot(x,(1-1/n)*x,'--',label='BNE')
    x = x.to(device)
    for name in ['SM+aBR', 'SM']:
        model = MLPBidNetwork(args.hidden, args.activation)
        model.load_state_dict(torch.load(os.path.join(os.path.join(dir_name, f'{name}.ckpt'))))
        model = model.to(device)
        with torch.no_grad(): y = model(x)
        plt.plot(x.cpu(),y.cpu(),label=f'{name}')
    plt.legend()
    plt.xlabel(r'Value $v_i$')
    plt.ylabel(r'Bid $b_i$')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title(r'Comparison of Learned Strategy')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, 'strategies.pdf'))
    # # plot fig
    # graph = sns.lineplot(df, x='Step', y='L2', hue='Method')
    # graph.set(yscale='log')
    # plt.savefig(os.path.join(dir_name, 'SM_plot.pdf'))