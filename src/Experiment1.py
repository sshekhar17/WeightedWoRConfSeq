"""Weighted CS without replacement and without side-information."""
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from weightedCSsequential import run_one_expt
from utils import generate_MFS, predictive_correction1
from utils import first_threshold_crossing, get_arg_values

from ExperimentBase import *

from tqdm import tqdm
# Two experiments


def get_methods_dict(lambda_max=2):
    """Generate the methods dictionary for this experiment."""
    methods_dict = {}
    methods_dict['propM'] = {
        'method_name': 'propM',
        'use_CV': False,
        'lambda_max': lambda_max
    }
    # methods_dict['propM+logical'] =  {'method_name':'propM', 'use_CV':False, 'lambda_max':lambda_max, 'intersect':True,
    #                                         'logical_CS':True}
    methods_dict['uniform'] = {
        'method_name': 'uniform',
        'use_CV': False,
        'lambda_max': lambda_max
    }
    # methods_dict['uniform+logical'] =  {'method_name':'uniform', 'use_CV':False, 'lambda_max':lambda_max, 'intersect':True,
    #                                         'logical_CS':True}
    return methods_dict


def CSexperiment(M_ranges,
                 f_ranges,
                 N=200,
                 N1=60,
                 lambda_max=2,
                 a=0.5,
                 inv_prop=True,
                 nG=100,
                 save_fig=False,
                 plot_CS_width=True):
    N2 = N - N1
    if inv_prop:
        title = f'{N1/N:.0%} large ' + r'$\pi$ values,    $f \propto 1/\pi$'
        figname = f'../data/NoSideInfoCSf_inv_propto_M_large_{N1}'
        f_ranges = [f_ranges[1], f_ranges[0]]
    else:
        title = f'{N1/N:.0%} large ' + r'$\pi$ values,    $f \propto \pi$'
        figname = f'../data/NoSideInfoCSf_propto_M_large_{N1}'
    # generate the problem instance
    M, f, S = generate_MFS(N_vals=(N1, N2),
                           N=N,
                           M_ranges=M_ranges,
                           f_ranges=f_ranges,
                           a=a)
    # create the methods dict
    methods_dict = get_methods_dict(lambda_max=lambda_max)
    # create the figure information dictionary
    xlabel = r'Sample Size ($n$) '
    if plot_CS_width:
        ylabel = 'Width of CS'
    else:
        ylabel = ' '
    fig_info_dict = {
        'title': title,
        'figname': figname,
        'xlabel': xlabel,
        'ylabel': ylabel
    }
    # run one trial of the experiment
    CompareMethodsOneTrial(M,
                           f,
                           S,
                           methods_dict,
                           nG=nG,
                           save_fig=save_fig,
                           fig_info_dict=fig_info_dict,
                           return_vals=False,
                           plot_results=True,
                           plot_CS_width=plot_CS_width,
                           post_process=True,
                           return_post_processed_separately=True)


def HistExperiment1(M_ranges,
                    f_ranges,
                    N=200,
                    N1=100,
                    epsilon=0.05,
                    inv_prop=True,
                    verbose=False,
                    plot_results=False,
                    save_fig=False,
                    a=0.5,
                    opacity=0.7,
                    num_trials=20):
    if inv_prop:
        title = f'Stopping Times Distribution \n' + f'{N1/N:.0%} large ' + r'$\pi$ values,    $f \propto 1/\pi$'
        figname = f'../data/NoSideInfoHist_f_inv_propto_M_large_{N1}'
        f_ranges = [f_ranges[1], f_ranges[0]]
    else:
        title = f'Stopping Times Distribution \n' + f'{N1/N:.0%} large ' + r'$\pi$ values,    $f \propto \pi$'
        figname = f'../data/NoSideInfoHist_f_propto_M_large_{N1}'
        # assume the original f values are proportional to M
        # so we don't need any modification here
    # generate the dictionary with methods information
    methods_dict = get_methods_dict()
    # generate the dictionary with histogram plotting information
    StoppingTimesDict = getStoppingTimesDistribution(
        methods_dict,
        N,
        N1,
        a=a,
        M_ranges=M_ranges,
        f_ranges=f_ranges,
        num_trials=num_trials,
        save_fig=False,
        post_process=True,
        epsilon=epsilon,
        return_post_processed_separately=True)

    if verbose:
        for key in StoppingTimesDict:
            st = StoppingTimesDict[key].mean()
            print(f'Method={key}, \t StoppingTimeAverage = {st:.2f}')

    if plot_results:
        xlabel = 'Stopping Times'
        ylabel = 'Density'
        hist_info_dict = {
            'title': title,
            'figname': figname,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'ymax': 1
        }
        plot_hists(N=N,
                   StoppingTimesDict=StoppingTimesDict,
                   save_fig=save_fig,
                   hist_info_dict=hist_info_dict,
                   opacity=opacity)


if __name__ == '__main__':
    args = get_arg_values()
    #### Set experiment options
    CSExpt = args.mode == 'cs'
    HistExpt = not CSExpt
    N = 200
    N1 = int(np.floor(args.small_prop * N))
    inv_prop = args.f_method == 'inv'

    #### Select the parameters of the experiments
    a = 0.1
    nG = 100
    N2 = N - N1
    lambda_max = 2
    M_ranges = [[1e5, 1e6], [1e2, 1 * 1e3]]
    f_ranges = [[0.4, 0.5], [1e-3, 2 * 1e-3]]
    epsilon = 0.05
    num_trials = 500

    #### Run the experiments
    if CSExpt:
        CSexperiment(M_ranges,
                     f_ranges,
                     N=N,
                     N1=N1,
                     lambda_max=lambda_max,
                     a=a,
                     inv_prop=inv_prop,
                     nG=nG,
                     save_fig=args.out_path,
                     plot_CS_width=True)

    if HistExpt:
        HistExperiment1(M_ranges,
                        f_ranges,
                        N=N,
                        N1=N1,
                        epsilon=epsilon,
                        inv_prop=inv_prop,
                        verbose=True,
                        plot_results=True,
                        save_fig=args.out_path,
                        a=a,
                        num_trials=num_trials)
