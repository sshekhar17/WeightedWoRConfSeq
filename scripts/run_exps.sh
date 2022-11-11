#!/usr/bin/bash
SEED=322

python src/Experiment1.py --mode cs --out_path figures/exp1_inv_2_cs.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment1.py --mode cs --out_path figures/exp1_inv_8_cs.tex --seed $SEED --small_prop 0.8 --f_method inv
python src/Experiment1.py --mode cs --out_path figures/exp1_prop_2_cs.tex --seed $SEED --small_prop 0.2 --f_method prop
python src/Experiment1.py --mode cs --out_path figures/exp1_prop_8_cs.tex --seed $SEED --small_prop 0.8 --f_method prop
python src/Experiment1.py --mode hist --out_path figures/exp1_inv_2_hist.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment1.py --mode hist --out_path figures/exp1_inv_8_hist.tex --seed $SEED --small_prop 0.8 --f_method inv
python src/Experiment1.py --mode hist --out_path figures/exp1_prop_2_hist.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment1.py --mode hist --out_path figures/exp1_prop_8_hist.tex --seed $SEED --small_prop 0.8 --f_method inv

python src/Experiment2.py --mode cs --out_path figures/exp2_inv_2_cs.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment2.py --mode cs --out_path figures/exp2_inv_8_cs.tex --seed $SEED --small_prop 0.8 --f_method inv
python src/Experiment2.py --mode cs --out_path figures/exp2_prop_2_cs.tex --seed $SEED --small_prop 0.2 --f_method prop
python src/Experiment2.py --mode cs --out_path figures/exp2_prop_8_cs.tex --seed $SEED --small_prop 0.8 --f_method prop
python src/Experiment2.py --mode hist --out_path figures/exp2_inv_2_hist.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment2.py --mode hist --out_path figures/exp2_inv_8_hist.tex --seed $SEED --small_prop 0.8 --f_method inv
python src/Experiment2.py --mode hist --out_path figures/exp2_prop_2_hist.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment2.py --mode hist --out_path figures/exp2_prop_8_hist.tex --seed $SEED --small_prop 0.8 --f_method inv

python src/Experiment3.py --mode cs --out_path figures/exp3_inv_2_cs.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment3.py --mode cs --out_path figures/exp3_inv_5_cs.tex --seed $SEED --small_prop 0.5 --f_method inv
python src/Experiment3.py --mode cs --out_path figures/exp3_inv_8_cs.tex --seed $SEED --small_prop 0.8 --f_method inv
python src/Experiment3.py --mode cs --out_path figures/exp3_prop_2_cs.tex --seed $SEED --small_prop 0.2 --f_method prop
python src/Experiment3.py --mode cs --out_path figures/exp3_prop_5_cs.tex --seed $SEED --small_prop 0.5 --f_method prop
python src/Experiment3.py --mode cs --out_path figures/exp3_prop_8_cs.tex --seed $SEED --small_prop 0.8 --f_method prop
python src/Experiment3.py --mode hist --out_path figures/exp3_inv_2_hist.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment3.py --mode hist --out_path figures/exp3_inv_5_hist.tex --seed $SEED --small_prop 0.5 --f_method inv
python src/Experiment3.py --mode hist --out_path figures/exp3_inv_8_hist.tex --seed $SEED --small_prop 0.8 --f_method inv
python src/Experiment3.py --mode hist --out_path figures/exp3_prop_2_hist.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment3.py --mode hist --out_path figures/exp3_prop_5_hist.tex --seed $SEED --small_prop 0.5 --f_method inv
python src/Experiment3.py --mode hist --out_path figures/exp3_prop_8_hist.tex --seed $SEED --small_prop 0.8 --f_method inv
