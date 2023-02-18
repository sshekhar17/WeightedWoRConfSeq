#!/usr/bin/bash

# This script runs all experiments to compare Hoeffding, empirical-Bernstein, and betting CSes.
#
if [ ! -d "figures/comp_cs" ]; then
    mkdir -p figures/comp_cs
fi
SEED=322

python src/Experiment1.py --method_suite comp_cs --N 200 --mode coverage --out_path figures/comp_cs/exp1_inv_2_coverage.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment1.py --method_suite comp_cs --N 200 --mode coverage --out_path figures/comp_cs/exp1_inv_8_coverage.tex --seed $SEED --small_prop 0.8 --f_method inv
python src/Experiment1.py --method_suite comp_cs --N 200 --mode coverage --out_path figures/comp_cs/exp1_prop_2_coverage.tex --seed $SEED --small_prop 0.2 --f_method prop
python src/Experiment1.py --method_suite comp_cs --N 200 --mode coverage --out_path figures/comp_cs/exp1_prop_8_coverage.tex --seed $SEED --small_prop 0.8 --f_method prop

python src/Experiment1.py --method_suite comp_cs --N 200 --mode hist --out_path figures/comp_cs/exp1_inv_2_hist.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment1.py --method_suite comp_cs --N 200 --mode hist --out_path figures/comp_cs/exp1_inv_8_hist.tex --seed $SEED --small_prop 0.8 --f_method inv
python src/Experiment1.py --method_suite comp_cs --N 200 --mode hist --out_path figures/comp_cs/exp1_prop_2_hist.tex --seed $SEED --small_prop 0.2 --f_method inv
python src/Experiment1.py --method_suite comp_cs --N 200 --mode hist --out_path figures/comp_cs/exp1_prop_8_hist.tex --seed $SEED --small_prop 0.8 --f_method inv
