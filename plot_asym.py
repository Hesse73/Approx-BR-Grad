from sym_exp import *
import seaborn as sns
sns.set_style('whitegrid')


if __name__ == '__main__':
    args = parser.parse_args()
    args.mechanism = 'fp'
    args.hidden = [128,128]  # for better representaion ability
    args.analytic = True
    args.abr = True
    device = device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ranges = [(0,0.5), (0,1)]
    num_types, bidders_per_type= 2, [1,1]
    sum_l2 = [0.0] * num_types
    utils.set_seed(42)
    args.kde_h, args.first_cond, args.lr = 0.01, 0.01, 0.05
    dir_name = f'figs/plot_asym'
    if not os.path.exists(dir_name): os.makedirs(dir_name)

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
        # # evaluate l2
        # l2_per_type = bidders.evaluate_sym_l2(prior, mechanism=args.mechanism)
        # titer.set_postfix_str(f'l2:{l2_per_type[0]:.4e}, {l2_per_type[1]:.4e}')
    # save model
    bidders.plot_special(prior, filename=os.path.join(dir_name, f'AsymStrategy.pdf'))
    torch.save([m.cpu().state_dict() for m in bidders.models], os.path.join(dir_name, f'asym_2types.ckpt'))
        