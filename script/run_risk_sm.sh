export CUDA_VISIBLE_DEVICES=4
# (basline) SM, n=2,5,10, fp_averse or sp_averse
python sym_exp.py --n 6 --mechanism fp_averse;
python sym_exp.py --n 8 --mechanism fp_averse;
python sym_exp.py --n 10 --mechanism fp_averse;
python sym_exp.py --n 6 --mechanism sp_averse;
python sym_exp.py --n 8 --mechanism sp_averse;
python sym_exp.py --n 10 --mechanism sp_averse;