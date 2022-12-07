"""Weighted CS without replacement and with accurate side-information."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from weightedCSsequential import run_one_expt
from utils import generate_MFS, predictive_correction1
from utils import first_threshold_crossing, get_arg_values
from ExperimentBase import *

from tqdm import tqdm


# Two experiments
def get_methods_dict(lambda_max=2,
                     f_over_S_range=None,
                     cs='Bet',
                     method_suite='default'):
    """Generate the methods dictionary for this experiment."""
    methods_dict = {}
    if method_suite == 'default':
        methods_dict['propM'] = {
            'method_name': 'propM',
            'use_CV': False,
            'lambda_max': lambda_max
        }
        methods_dict['propMS'] = {
            'method_name': 'propMS',
            'use_CV': False,
            'lambda_max': lambda_max,
            'f_over_S_range': f_over_S_range
        }
        if cs != 'Bet':
            for key in methods_dict:
                methods_dict[key]['cs'] = cs
    else:
        cses = ['Hoef.', 'Emp. Bern.'
                ] + ([] if method_suite == 'comp_cs' else ['Bet'])
        for cs in cses:
            methods_dict[cs] = {
                'method_name': 'propMS',
                'cs': cs,
                'use_CV': False,
                'lambda_max': lambda_max,
                'f_over_S_range': f_over_S_range
            }
    return methods_dict


def CSexperiment2(M_ranges,
                  f_ranges,
                  method_suite,
                  N=200,
                  N1=60,
                  lambda_max=2,
                  a=0.5,
                  inv_prop=True,
                  nG=100,
                  save_fig=False,
                  plot_CS_width=True,
                  post_process='logical',
                  f_over_S_range=None):
    N2 = N - N1
    if inv_prop:
        title = f'{N1/N:.0%} large ' + r'$\pi$ values,    $f \propto 1/\pi$'
        figname = f'/AccurateSideInfoCSf_inv_propto_M_large_{N1}_A_{a}'.replace(
            '.', '_')
        figname = '../data' + figname
        f_ranges = [f_ranges[1], f_ranges[0]]
    else:
        title = f'{N1/N:.0%} large ' + r'$\pi$ values,    $f \propto \pi$'
        figname = f'/AccurateSideInfoCSf_propto_M_large_{N1}_A_{a}'.replace(
            '.', '_')
        figname = '../data' + figname
    # generate the problem instance
    M, f, S = generate_MFS(N_vals=(N1, N2),
                           N=N,
                           M_ranges=M_ranges,
                           f_ranges=f_ranges,
                           a=a)
    # create the methods dict
    methods_dict = get_methods_dict(lambda_max=lambda_max,
                                    f_over_S_range=f_over_S_range,
                                    method_suite=method_suite)
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
                           post_process=post_process
                           in ['separate', 'logical'],
                           return_post_processed_separately=post_process
                           in ['separate'])


def HistExperiment2(M_ranges,
                    f_ranges,
                    N=200,
                    N1=100,
                    epsilon=0.05,
                    inv_prop=True,
                    verbose=False,
                    plot_results=False,
                    save_fig=False,
                    a=0.5,
                    f_over_S_range=None,
                    lambda_max=2.5,
                    num_trials=20):
    if inv_prop:
        title = f'Stopping Times Distribution \n' + f'{N1/N:.0%} large ' + r'$\pi$ values,    $f \propto 1/\pi$'
        figname = f'/AccurateSideInfoHist_f_inv_propto_M_large_{N1}_A_{a}'.replace(
            '.', '_')
        figname = '../data' + figname
        f_ranges = [f_ranges[1], f_ranges[0]]
    else:
        title = f'Stopping Times Distribution \n' + f'{N1/N:.0%} large ' + r'$\pi$ values,    $f \propto \pi$'
        figname = f'/AccurateSideInfoHist_f_propto_M_large_{N1}_A_{a}'.replace(
            '.', '_')
        figname = '../data' + figname
        # assume the original f values are proportional to M
        # so we don't need any modification here
    # generate the dictionary with methods information
    methods_dict = get_methods_dict(lambda_max=lambda_max,
                                    f_over_S_range=f_over_S_range,
                                    method_suite=method_suite)
    # generate the dictionary with histogram plotting information
    StoppingTimesDict = getStoppingTimesDistribution(methods_dict,
                                                     N,
                                                     N1,
                                                     a=a,
                                                     M_ranges=M_ranges,
                                                     f_ranges=f_ranges,
                                                     num_trials=num_trials,
                                                     save_fig=False,
                                                     post_process=False,
                                                     epsilon=epsilon)

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
            'ylabel': ylabel
        }
        plot_hists(N=N,
                   StoppingTimesDict=StoppingTimesDict,
                   save_fig=save_fig,
                   hist_info_dict=hist_info_dict,
                   opacity=0.5)


if __name__ == '__main__':
    args = get_arg_values()
    CSExpt = args.mode == 'cs'
    HistExpt = not CSExpt
    save_fig = args.out_path
    N = 200
    N1 = int(np.floor(args.small_prop * N))
    inv_prop = args.f_method == 'inv'

    a = 0.1  # ensures that (f/s)-values lie in [1-a, 1+a]
    N2 = N - N1
    epsilon = 0.05
    # f_over_S_range = [1-a, 1+a]
    f_over_S_range = [1 / (1 + a), 1 / (1 - a)]
    # M_ranges = [ [1e5, 1e6], [1e5, 1e6]]
    # M_ranges = [ [1e5, 5e5], [6e5, 1e6]]
    M_ranges = [[1e5, 5e5], [1e4, 1e5]]
    f_ranges = [[0.4, 0.5], [1e-3, 2 * 1e-3]]
    lambda_max = 2.0
    nG = 100

    if CSExpt:
        CSexperiment2(M_ranges,
                      f_ranges,
                      args.method_suite,
                      N=N,
                      N1=N1,
                      lambda_max=lambda_max,
                      a=a,
                      inv_prop=inv_prop,
                      nG=nG,
                      save_fig=save_fig,
                      plot_CS_width=args.cs_metric == 'width',
                      post_process=args.post_process,
                      f_over_S_range=f_over_S_range)

    if HistExpt:
        epsilon = 0.05
        num_trials = 250
        HistExperiment2(M_ranges,
                        f_ranges,
                        args.method_suite,
                        N=N,
                        N1=N1,
                        lambda_max=lambda_max,
                        epsilon=epsilon,
                        inv_prop=inv_prop,
                        verbose=True,
                        plot_results=True,
                        save_fig=save_fig,
                        a=a,
                        f_over_S_range=f_over_S_range,
                        lambda_max=lambda_max,
                        num_trials=num_trials)
