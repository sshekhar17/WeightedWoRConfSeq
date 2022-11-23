#!/usr/bin/bash

# This script runs all experiments to compare the CSes
SEED=322

python src/Experiment1.py --mode cs --out_path figures/comp_cs/exp1_inv_2_cs.tex --seed $SEED --small_prop 0.2 --f_method inv --method_suite comp_cs
