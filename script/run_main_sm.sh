export CUDA_VISIBLE_DEVICES=6
# (baseline) SM, n=2,5,10, fp or sp
python sym_exp.py --n 2 --mechanism fp;
python sym_exp.py --n 5 --mechanism fp;
python sym_exp.py --n 10 --mechanism fp;
python sym_exp.py --n 2 --mechanism sp;
python sym_exp.py --n 5 --mechanism sp;
python sym_exp.py --n 10 --mechanism sp;