# Risk-limiting Financial Audits via Weighted Sampling without Replacement

## Installation

Requires Python 3.8.3. Install dependencies with `pip install -r requirements.txt`

## Running experiments

Figures generated from running experiments will be in the `figures/` directory.

- `bash scripts/run_exps.sh` runs all the experiments in the main body of the paper.
- `bash scripts/run_cs_comp.sh` runs all the experiments in Appendix D that compares different types of CSes.

These figures can be rendered in a LaTeX --- below is an example template.

```
\documentclass[10pt]{article}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=newest}
\pgfplotsset{scaled y ticks=false}
\usepgfplotslibrary{groupplots}
\usepgfplotslibrary{dateplot}
\tikzstyle{every node}=[font=\small]
\pgfplotsset{
    yticklabel style={/pgf/number format/fixed},
}
% For fixing legend of histograms
\pgfplotsset{compat=1.11,
 /pgfplots/ybar legend/.style={
 /pgfplots/legend image code/.code={
 \draw[##1,/tikz/.cd,yshift=-0.25em]
 (0cm,0cm) rectangle (3pt,0.8em);},
 },
}

\begin{document}
\input{<path to tex figure>}
\end{document}
```


## Code structure

All of the code is the `src/` directory.

The notebook `src/example.ipynb` explains the basic steps in setting up an 
experiment, and constructing confidence sequences. 

Confidence sequence code is in `hoeffding.py`, `bernstein.py`, (for Hoeffding and empirical-Bernstein) and the betting CS code is in `weightedCSsequential.py`, along with the helper functions.

`utils.py` contains code for generating the transaction/misstatement values, as well as other useful functions.


### Reproducing the figures 
`ExperimentBase.py` and `Experiment{1,2,3}.py` contain the code for running experiments and simulation setups that occur. `Experiment4.py` contains the code 
for comparing the performance of three methods (propM, propM+CV, and uniform) on 
a 'semi-real-world' [dataset of house prices](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction). 

