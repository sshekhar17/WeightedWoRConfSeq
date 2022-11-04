"""
    Old Implementation
"""
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

from weightedCSsequential import run_one_expt
from utils import generate_MFS, predictive_correction1
from utils import first_threshold_crossing

from ExperimentBase import *

from tqdm import tqdm 
#
def main1(M, f, S,  nG=100, savefig=False, figname=None,
            title=None, return_vals=False, plot_results=True):
    """
    plot the CS for uniform and propM strategies; with and without logicalCS 
    """
    N = len(M)
    Pi = M / M.sum()
    grid, _, LowerCS1, UpperCS1, Idx1, _ = run_one_expt(M, f, S, nG=nG,
                                                    method_name='uniform', 
                                                    lambda_max=1, 
                                                    use_CV=False)

    LowerCS1_, UpperCS1_ = predictive_correction1(LowerCS1, UpperCS1, 
                                            Idx=Idx1, Pi=Pi, f=f, 
                                            logical=True, intersect=True)

    _, _, LowerCS2, UpperCS2, Idx2, _ = run_one_expt(M, f, S, nG=nG,
                                                    method_name='propM', 
                                                    lambda_max=1, 
                                                    use_CV=False)

    LowerCS2_, UpperCS2_ = predictive_correction1(LowerCS2, UpperCS2, 
                                            Idx=Idx2, Pi=Pi, f=f, 
                                            logical=True, intersect=True)

    if plot_results:
        NN = np.arange(1, N+1) 
        palette = sns.color_palette(n_colors=10)
        plt.figure()

        plt.plot(NN, UpperCS1-LowerCS1,  label='uniform', color=palette[0])
        plt.plot(NN, UpperCS1_-LowerCS1_,  '--', label='uniform + logical', color=palette[0])

        plt.plot(NN, UpperCS2- LowerCS2,  label='propM', color=palette[1])
        plt.plot(NN, UpperCS2_-LowerCS2_,  '--', label='uniform + logical', color=palette[1])
        plt.legend(fontsize=13)
        plt.xlabel('Sample Size (n)', fontsize=14)
        plt.ylabel('Width of CS', fontsize=14)
        title = 'CS Width vs Sample Size' if title is None else title 
        plt.title(title, fontsize=15)

        if savefig:
            if figname is None:
                figname = './data/NoSideInfoCS.png' 
            plt.savefig(figname, dpi=450)
        else:
            plt.show()
    
    if return_vals:
        return LowerCS1, UpperCS1, Idx1, LowerCS2, UpperCS2, Idx2


def main2(M, f, S, epsilon=0.05):
    """
    calculate the stopping time for uniform and propM strategies 
    """
    Pi = M/M.sum()
    # get the lower and upper CS for uniform and propM strategies 
    LowerCS1, UpperCS1, Idx1, LowerCS2, UpperCS2, Idx2 = main1(M, f, S,  nG=100, 
    savefig=False, figname=None, title=None, return_vals=True, plot_results=False)
    # get the improved versions of these CSes 
    LowerCS1_, UpperCS1_ = predictive_correction1(LowerCS1, UpperCS1, 
                                            Idx=Idx1, Pi=Pi, f=f, 
                                            logical=True, intersect=True)
    LowerCS2_, UpperCS2_ = predictive_correction1(LowerCS2, UpperCS2, 
                                            Idx=Idx2, Pi=Pi, f=f, 
                                            logical=True, intersect=True)

    W1 = UpperCS1-LowerCS1
    W2 = UpperCS2-LowerCS2
    W1_ = UpperCS1_-LowerCS1_
    W2_ = UpperCS2_-LowerCS2_
    N = len(M)
    # get the stopping times 
    ST1 = first_threshold_crossing(W1, th=epsilon, max_time=N, upward=False)
    ST2 = first_threshold_crossing(W2, th=epsilon, max_time=N, upward=False)
    ST1_ = first_threshold_crossing(W1_, th=epsilon, max_time=N, upward=False)
    ST2_ = first_threshold_crossing(W2_, th=epsilon, max_time=N, upward=False)
    return ST1, ST2, ST1_, ST2_


def main3(N, N1, M_ranges = [ [1e5, 1e6], [1e2, 1*1e3]], 
                f_ranges = [[1e-3, 2*1e-3], [0.4, 0.5]],  
                num_trials = 200, epsilon=0.05, plot_results =True, 
                return_vals=False, title=None, figname=None, 
                savefig=False
):
    """
    plot the stopping time distributions 
    """
    A = 0.1 # any arbitrary value if fine, since we are not using S in this expt
    N2 = N-N1
    ST1 = np.zeros((num_trials, ))
    ST2 = np.zeros((num_trials, ))
    ST1_ = np.zeros((num_trials, ))
    ST2_ = np.zeros((num_trials, ))
    for trial in tqdm(range(num_trials)):
        M, f, S =  generate_MFS(N_vals=(N1, N2), 
                    N=N, 
                    M_ranges = M_ranges, 
                    f_ranges = f_ranges, 
                    a = A)
        st1, st2, st1_, st2_ = main2(M, f, S, epsilon=epsilon)
        ST1[trial] = st1
        ST2[trial] = st2
        ST1_[trial] = st1_
        ST2_[trial] = st2_
    
    if plot_results: 
        palette = sns.color_palette(n_colors=10)
        plt.figure()
        # plt.hist(ST1, label='uniform', color=palette[0], alpha=0.8, density=True)
        plt.hist(ST1_, label='uniform+logical', color=palette[0], alpha=0.5, density=True)
        plt.hist(ST2, label='propM', color=palette[1], alpha=0.8, density=True)
        plt.hist(ST2_, label='propM+logical', color=palette[1],
                    alpha=0.5, density=True, edgecolor='black')
        plt.legend(fontsize=13)
        if title is None:
            title = r'Distribution of Stopping Times ($epsilon$='+f'{epsilon})'
        plt.title(title, fontsize=15)
        plt.xlabel(r'Stopping Time ($\tau$)', fontsize=14)
        plt.xlabel('Density', fontsize=14)
        if savefig: 
            if figname is None:
                figname = '../data/histogram.png'
            plt.savefig(figname, dpi=450)
        else: 
            plt.show()
        if return_vals:
            return ST1, ST2, ST1_, ST2_