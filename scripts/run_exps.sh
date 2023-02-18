#!/usr/bin/bash

# This script runs all experiments that use betting CS and compare different sampling strategies and control variate use.
SEED=322
if [ ! -d "figures" ]; then
    mkdir -p figures
fi

python src/Experiment1.py --mode cs --N 200 --method_suite betting --post_process logical --legend_flag none --out_path figures/exp1_inv_2_cs.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment1.py --mode cs --N 200 --method_suite betting --post_process logical --legend_flag none --out_path figures/exp1_inv_8_cs.tex --seed $SEED --small_prop 0.8 --f_method inv
python src/Experiment1.py --mode cs --N 200 --method_suite betting --post_process logical --out_path figures/exp1_prop_2_cs.tex --seed $SEED --small_prop 0.2 --f_method prop
python src/Experiment1.py --mode cs --N 200 --method_suite betting --post_process logical --legend_flag none --out_path figures/exp1_prop_8_cs.tex --seed $SEED --small_prop 0.8 --f_method prop
python src/Experiment1.py --mode hist --N 200 --method_suite betting --epsilon 0.2 --post_process logical --legend_flag none --out_path figures/exp1_inv_2_hist.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment1.py --mode hist --N 200 --method_suite betting --epsilon 0.2 --post_process logical --legend_flag none --out_path figures/exp1_inv_8_hist.tex --seed $SEED --small_prop 0.8 --f_method inv
python src/Experiment1.py --mode hist --N 200 --method_suite betting --epsilon 0.2 --post_process logical --out_path figures/exp1_prop_2_hist.tex --seed $SEED --small_prop 0.2 --f_method prop
python src/Experiment1.py --mode hist --N 200 --method_suite betting --epsilon 0.2 --post_process logical --legend_flag none --out_path figures/exp1_prop_8_hist.tex --seed $SEED --small_prop 0.8 --f_method prop

python src/Experiment2.py --mode cs --out_path figures/exp2_inv_2_cs.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment2.py --mode cs --out_path figures/exp2_inv_8_cs.tex --seed $SEED --small_prop 0.8 --f_method inv
python src/Experiment2.py --mode cs --out_path figures/exp2_prop_2_cs.tex --seed $SEED --small_prop 0.2 --f_method prop
python src/Experiment2.py --mode cs --out_path figures/exp2_prop_8_cs.tex --seed $SEED --small_prop 0.8 --f_method prop
python src/Experiment2.py --mode hist --out_path figures/exp2_inv_2_hist.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment2.py --mode hist --out_path figures/exp2_inv_8_hist.tex --seed $SEED --small_prop 0.8 --f_method inv
python src/Experiment2.py --mode hist --out_path figures/exp2_prop_2_hist.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment2.py --mode hist --out_path figures/exp2_prop_8_hist.tex --seed $SEED --small_prop 0.8 --f_method inv

python src/Experiment3.py --mode cs --out_path figures/exp3_inv_2_cs.tex --seed $SEED --small_prop 0.2 --f_method inv --save_fig
python src/Experiment3.py --mode cs --out_path figures/exp3_inv_5_cs.tex --seed $SEED --small_prop 0.5 --f_method inv --save_fig
python src/Experiment3.py --mode cs --out_path figures/exp3_inv_8_cs.tex --seed $SEED --small_prop 0.8 --f_method inv --save_fig
python src/Experiment3.py --mode cs --out_path figures/exp3_prop_2_cs.tex --seed $SEED --small_prop 0.2 --f_method prop --save_fig
python src/Experiment3.py --mode cs --out_path figures/exp3_prop_5_cs.tex --seed $SEED --small_prop 0.5 --f_method prop --save_fig
python src/Experiment3.py --mode cs --out_path figures/exp3_prop_8_cs.tex --seed $SEED --small_prop 0.8 --f_method prop --save_fig
python src/Experiment3.py --mode hist --out_path figures/exp3_inv_2_hist.tex --seed $SEED --small_prop 0.2 --f_method inv --save_fig
python src/Experiment3.py --mode hist --out_path figures/exp3_inv_5_hist.tex --seed $SEED --small_prop 0.5 --f_method inv --save_fig
python src/Experiment3.py --mode hist --out_path figures/exp3_inv_8_hist.tex --seed $SEED --small_prop 0.8 --f_method inv --save_fig
python src/Experiment3.py --mode hist --out_path figures/exp3_prop_2_hist.tex --seed $SEED --small_prop 0.2 --f_method inv --save_fig
python src/Experiment3.py --mode hist --out_path figures/exp3_prop_5_hist.tex --seed $SEED --small_prop 0.5 --f_method inv --save_fig
python src/Experiment3.py --mode hist --out_path figures/exp3_prop_8_hist.tex --seed $SEED --small_prop 0.8 --f_method inv --save_fig
