export CUDA_VISIBLE_DEVICES=2
python asym_exp.py --mechanism sp --analytic --n 3 --lr 0.2 --kde_h 0.1
python asym_exp.py --mechanism sp --analytic --n 4 --lr 0.2 --kde_h 0.1
python asym_exp.py --mechanism sp --analytic --n 5 --lr 0.2 --kde_h 0.1
python asym_exp.py --mechanism sp --analytic --abr --n  3 --lr 0.2 --first_cond 0.05 --kde_h 0.1
python asym_exp.py --mechanism sp --analytic --abr --n  4 --lr 0.2 --first_cond 0.05 --kde_h 0.1
python asym_exp.py --mechanism sp --analytic --abr --n  5 --lr 0.2 --first_cond 0.05 --kde_h 0.1