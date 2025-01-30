export CUDA_VISIBLE_DEVICES=5
# UG, n=2,5,10, fp_averse or sp_averse
python sym_exp.py --n 6 --mechanism fp_averse --analytic;
python sym_exp.py --n 8 --mechanism fp_averse --analytic;
python sym_exp.py --n 10 --mechanism fp_averse --analytic;
python sym_exp.py --n 6 --mechanism sp_averse --analytic;
python sym_exp.py --n 8 --mechanism sp_averse --analytic;
python sym_exp.py --n 10 --mechanism sp_averse --analytic;
# aBR, n=2,5,10, fp_averse or sp_averse
python sym_exp.py --n 6 --mechanism fp_averse --analytic --abr --first_cond 0.05;
python sym_exp.py --n 8 --mechanism fp_averse --analytic --abr --first_cond 0.05;
python sym_exp.py --n 10 --mechanism fp_averse --analytic --abr --first_cond 0.05;
python sym_exp.py --n 6 --mechanism sp_averse --analytic --abr --first_cond 0.05;
python sym_exp.py --n 8 --mechanism sp_averse --analytic --abr --first_cond 0.05;
python sym_exp.py --n 10 --mechanism sp_averse --analytic --abr --first_cond 0.05;