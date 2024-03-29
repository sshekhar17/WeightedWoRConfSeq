"""Weighted CS without replacement and with general side-information:

Comparison of propM with propM+control-variates.
"""
import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib as tpl

from weightedCSsequential import run_one_expt
from utils import generate_MFS, predictive_correction1
from utils import first_threshold_crossing, get_arg_values
from ExperimentBase import *

from tqdm import tqdm

#########################################################
#### get the absolute path of the directory to store data
module_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(module_dir)
FIG_DIR = parent_dir + '/data'
print(FIG_DIR)
####
#########################################################

def get_methods_dict(lambda_max=2, beta_max=0.5, method_suite='default'):
    """Generate the methods dictionary for this experiment."""
    methods_dict = {}
    if method_suite == 'default':
        methods_dict['propM'] = {
            'method_name': 'propM',
            'use_CV': False,
            'lambda_max': lambda_max
        }
        methods_dict['propM+CV'] = {
            'method_name': 'propM',
            'use_CV': True,
            'lambda_max': lambda_max,
            'beta_max': beta_max
        }
    else:
        cses = ['Hoef.', 'Emp. Bern.'
                ] + ([] if method_suite == 'comp_cs' else ['Bet'])
        for cs in cses:
            methods_dict[cs] = {
                'method_name': 'propM',
                'cs': cs,
                'use_CV': True,
                'lambda_max': lambda_max,
                'beta_max': beta_max
            }
    return methods_dict


def CSexperiment3(M_ranges,
                  f_ranges,
                  method_suite,
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
        figname = f'{FIG_DIR}/CV_CS_inv_propto_M_large_{N1}'
        f_ranges = [f_ranges[1], f_ranges[0]]
    else:
        title = f'{N1/N:.0%} large ' + r'$\pi$ values,    $f \propto \pi$'
        figname = f'{FIG_DIR}/CV_CS_propto_M_large_{N1}'
    # generate the problem instance
    M, f, S = generate_MFS(N_vals=(N1, N2),
                           N=N,
                           M_ranges=M_ranges,
                           f_ranges=f_ranges,
                           a=a)
    # create the methods dict
    methods_dict = get_methods_dict(lambda_max=lambda_max,
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
                           post_process=True)


def HistExperiment3(M_ranges,
                    f_ranges,
                    method_suite,
                    N=200,
                    N1=100,
                    epsilon=0.05,
                    inv_prop=True,
                    verbose=False,
                    plot_results=False,
                    save_fig=False,
                    a=0.5,
                    lambda_max=2.5,
                    beta_max=0.5,
                    num_trials=20,
                    return_vals=False,
                    progress_bar=True, 
                    c=0.1):
    if inv_prop:
        title = f'Stopping Times Distribution \n' + f'{N1/N:.0%} large ' + r'$\pi$ values,    $f \propto 1/\pi$'
        figname = f'{FIG_DIR}/CVHist_f_inv_propto_M_large_{N1}'
        f_ranges = [f_ranges[1], f_ranges[0]]
    else:
        title = f'Stopping Times Distribution \n' + f'{N1/N:.0%} large ' + r'$\pi$ values,    $f \propto \pi$'
        figname = f'{FIG_DIR}/CVHist_f_propto_M_large_{N1}'
        # assume the original f values are proportional to M
        # so we don't need any modification here
    # generate the dictionary with methods information
    methods_dict = get_methods_dict(lambda_max=lambda_max,
                                    beta_max=beta_max,
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
                                                     post_process=True,
                                                     epsilon=epsilon,
                                                     progress_bar=progress_bar, 
                                                     use_CV=True, 
                                                     c=c)

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
    if return_vals:
        ST, STcv = StoppingTimesDict['propM'], StoppingTimesDict['propM+CV']
        return ST, STcv


def GainExperiment(Corr,
                   M_ranges,
                   f_ranges,
                   N,
                   N1,
                   epsilon=0.025,
                   inv_prop=True,
                   lambda_max=2.5,
                   beta_max=0.5,
                   num_trials=50,
                   plot_results=False,
                   save_fig=False,
                   return_vals=False):
    GainMean = np.zeros((Corr.shape))
    GainStd = np.zeros((Corr.shape))
    for i, corr in tqdm(list(enumerate(Corr))):
        ST, STcv = HistExperiment3(M_ranges,
                                   f_ranges,
                                   'default',
                                   N,
                                   N1,
                                   epsilon=epsilon,
                                   inv_prop=inv_prop,
                                   verbose=False,
                                   save_fig=False,
                                   lambda_max=lambda_max,
                                   beta_max=beta_max,
                                   num_trials=num_trials,
                                   return_vals=True,
                                   progress_bar=False, 
                                   c=corr)
        Ratio = STcv / ST
        GainMean[i], GainStd[i] = Ratio.mean(), Ratio.std()

    if inv_prop:
        figname = f'/CV_Gain_f_inv_propto_M_large_{N1}_eps_{epsilon}'.replace(
            '.', '_')
        figname = f'{FIG_DIR}' + figname
    else:
        figname = f'/CV_Gain_f_propto_M_large_{N1}_eps_{epsilon}'.replace(
            '.', '_')
        figname = f'{FIG_DIR}' + figname
    # plot the results
    if plot_results:
        palette = sns.color_palette(n_colors=10)
        plt.figure()
        plt.plot(Corr, GainMean, color=palette[0])
        plt.fill_between(Corr,
                         GainMean - GainStd,
                         GainMean + GainStd,
                         alpha=0.3,
                         color=palette[0])
        plt.plot(Corr, np.ones(Corr.shape), 'k--', alpha=0.6)
        plt.title('Reduction in Sample-Size by using Control Variates',
                  fontsize=15)
        plt.xlabel(r'Correlation between $S$ and $f$ ', fontsize=13)
        plt.ylabel('Ratio of stopping times with and without CV', fontsize=13)

        if save_fig:
            figname_tex = figname + '.tex'
            figname_png = figname + '.png'
            plt.savefig(figname_png, dpi=450)
            tpl.save(figname_tex,
                     axis_width=r'\figwidth',
                     axis_height=r'\figheight')
            # plt.savefig(figname, dpi=450)
        else:
            plt.show()
    if return_vals:
        return GainMean, GainStd


if __name__ == '__main__':
    args = get_arg_values()
    CSExpt = args.mode == 'cs'
    HistExpt = args.mode == 'hist'
    GainExpt = args.mode == 'gain'
    save_fig = args.save_fig
    N = 200
    N1 = int(np.floor(args.small_prop * N))
    inv_prop = args.f_method == 'inv'

    a = 0.1
    corr = 0.7 # controls the correlation b/w f & S
    N2 = N - N1
    f_over_S_range = [1 - a, 1 + a]
    M_ranges = [[5e5, 9e5], [1e6, 1e6]]
    f_ranges = [[0.4, 0.5], [1e-3, 2 * 1e-3]]
    lambda_max = 2.0 # 
    nG = 100 #100
    beta_max = 0.8
    num_trials = 5

    # M_ranges = [ [1e5, 1e6], [1e2, 1*1e3]],
    if CSExpt:
        CSexperiment3(M_ranges,
                      f_ranges,
                      args.method_suite,
                      N,
                      N1,
                      lambda_max,
                      a=a,
                      inv_prop=inv_prop,
                      nG=nG,
                      save_fig=save_fig,
                      plot_CS_width=True)

    if HistExpt:
        epsilon = 0.05
        ST, STcv= HistExperiment3(M_ranges,
                        f_ranges,
                        args.method_suite,
                        N,
                        N1,
                        epsilon=epsilon,
                        inv_prop=inv_prop,
                        verbose=True,
                        save_fig=save_fig,
                        a=a,
                        lambda_max=lambda_max,
                        beta_max=beta_max,
                        num_trials=num_trials,
                        return_vals=True,
                        plot_results=True, 
                        c=corr)
        print(f'Average gain with a={a} is {(ST/STcv).mean():.3f}')
    if GainExpt:
        Corr = np.linspace(0.1, 0.9, 9)
        Gain = np.zeros(Corr.shape)
        # smaller values of epsilon shows better gains 
        # in this experiment
        epsilon = 0.025 
        GainExperiment(Corr,
                       M_ranges,
                       f_ranges,
                       N,
                       N1,
                       epsilon=epsilon,
                       inv_prop=inv_prop,
                       lambda_max=lambda_max,
                       beta_max=beta_max,
                       num_trials=num_trials,
                       plot_results=True,
                       save_fig=save_fig,
                       return_vals=False)
