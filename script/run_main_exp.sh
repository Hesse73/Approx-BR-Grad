export CUDA_VISIBLE_DEVICES=7
# UG, n=2,5,10, fp or sp
python sym_exp.py --n 2 --mechanism fp --analytic;
python sym_exp.py --n 5 --mechanism fp --analytic;
python sym_exp.py --n 10 --mechanism fp --analytic;
python sym_exp.py --n 2 --mechanism sp --analytic;
python sym_exp.py --n 5 --mechanism sp --analytic;
python sym_exp.py --n 10 --mechanism sp --analytic;
# aBR, n=2,5,10, fp or sp
python sym_exp.py --n 2 --mechanism fp --analytic --abr;
python sym_exp.py --n 5 --mechanism fp --analytic --abr;
python sym_exp.py --n 10 --mechanism fp --analytic --abr;
python sym_exp.py --n 2 --mechanism sp --analytic --abr;
python sym_exp.py --n 5 --mechanism sp --analytic --abr;
python sym_exp.py --n 10 --mechanism sp --analytic --abr;