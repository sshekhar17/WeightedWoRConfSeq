from typing import Optional

import os

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tpl
import seaborn as sns
from tqdm import tqdm

from weightedCSsequential import run_one_expt
from utils import predictive_correction1, generate_MFS, first_threshold_crossing
from constants import ColorsDict, LineStyleDict


# def plot_results1
def plot_CS(N,
            M,
            f,
            Results_Dict,
            plot_CS_width=False,
            palette=None,
            save_fig=False,
            fig_info_dict={},
            diagnostics=None):
    # default color palette
    # if palette is None:
    #     palette = sns.color_palette(palette='tab10', n_colors=len(Results_Dict)+1)
    #plot the Confidence sequences
    m_star = np.sum(M * f) / np.sum(M)
    assert (N == len(M))
    NN = np.arange(1, N + 1)
    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots()
    for i, key in enumerate(Results_Dict):
        color = ColorsDict[key]
        linestyle = LineStyleDict[key]
        results = Results_Dict[key]
        # we implicitly follow the assumption that the
        # values in the Results_Dict dictionary is iterable,
        # and the first two elements are the Lower and Upper CS
        # respectively.
        L, U = results[0], results[1]
        if plot_CS_width:
            ax.plot(NN, U - L, label=key, color=color, linestyle=linestyle)

        else:
            ax.plot(NN, L, label=key, color=color, linestyle=linestyle)
            ax.plot(NN, U, color=color, linestyle=linestyle)

    if not plot_CS_width:
        ax.plot(NN,
                m_star * np.ones(NN.shape),
                '--',
                label=f'$m^*$={m_star}',
                color='k')
    if len(fig_info_dict) == 0:
        title = f'Confidence Sequence for $m^*$={m_star}'
        xlabel = 'Number of queries'
        ylabel = ''
        figname = '../data/tempCS'
    else:
        title = fig_info_dict['title']
        xlabel = fig_info_dict['xlabel']
        ylabel = fig_info_dict['ylabel']
        figname = fig_info_dict['figname']
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.legend(fontsize=13)
    fig.tight_layout()
    # save the figure
    if diagnostics is not None:
        name, lower_d, upper_d = diagnostics
        diag_figs = []
        for diag in [lower_d, upper_d]:
            diag_names = [
                '$\\lambda_t$', '$V_t$', '$c_t$', 'Avg. $V_t$',
                'Avg. $V_tc_t$', 'Pred. avg. $V_tc_t$', 'Pred. $\sum 1/c_t^2$',
                'Ind margin t.', 'Total margin t.', '$\widehat{\mu}_t$'
            ]
            figtmp, axes = plt.subplots(len(diag_names),
                                        1,
                                        sharex=True,
                                        figsize=(10, len(diag_names) * 2),
                                        squeeze=False)
            for idx, (diag_name, arr) in enumerate(zip(diag_names, diag)):
                t = np.arange(1, arr.shape[0] + 1)
                axes[idx, 0].plot(t, arr, label=diag_name)
                axes[idx, 0].set_title(diag_name)
            diag_figs.append(figtmp)

    if save_fig:
        figname_ = figname + '.tex' if save_fig is True else save_fig
        tpl.save(figname_, axis_width=r'\figwidth', axis_height=r'\figheight')
        basename = os.path.splitext(figname_)[0]
        picname_ = basename + '.png'
        fig.savefig(picname_, dpi=300)
        if diagnostics is not None:
            diag_figs[0].savefig(basename + '_lower_diagnostics.png', dpi=300)
            diag_figs[1].savefig(basename + '_upper_diagnostics.png', dpi=300)


def plot_hists(N,
               StoppingTimesDict,
               palette=None,
               plot_mean_val=False,
               save_fig=False,
               hist_info_dict={},
               opacity=0.5,
               ymax=None):
    # default color palette
    if palette is None:
        palette = sns.color_palette(palette='tab10',
                                    n_colors=len(StoppingTimesDict) + 1)
    #plot the Confidence sequences
    NN = np.arange(1, N + 1)
    fig = plt.figure()
    for i, key in enumerate(StoppingTimesDict):
        color = ColorsDict[key]

        ST = StoppingTimesDict[key]
        st = ST.mean()
        plt.hist(ST, label=key, alpha=opacity, color=color, density=True)
        if plot_mean_val:
            plt.axvline(x=st, linestyle='--', color=color, linewidth=1.5)

    if len(hist_info_dict) == 0:
        title = r'Stopping Times Distribution'
        xlabel = 'Stopping Times'
        ylabel = 'Density'
        figname = '../data/histCS'
        ymax = None
    else:
        title = hist_info_dict['title']
        xlabel = hist_info_dict['xlabel']
        ylabel = hist_info_dict['ylabel']
        figname = hist_info_dict['figname']
        if 'ymax' in hist_info_dict:
            ymax = hist_info_dict['ymax']
        else:
            ymax = None
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.legend(fontsize=13)
    if ymax is not None:
        plt.ylim([0, ymax])
    # save the figure
    if save_fig:
        figname_ = figname + '.tex' if save_fig is True else save_fig
        tpl.save(figname_, axis_width=r'\figwidth', axis_height=r'\figheight')
        # plt.savefig(figname, dpi=450)
        picname_ = os.path.splitext(figname_)[0] + '.png'
        fig.savefig(picname_, dpi=300)


def CompareMethodsOneTrial(M,
                           f,
                           S,
                           methods_dict,
                           nG=100,
                           save_fig=False,
                           fig_info_dict={},
                           return_vals=False,
                           plot_results=True,
                           plot_CS_width=True,
                           post_process=True,
                           return_post_processed_separately=False,
                           seed: Optional[int] = None):
    """plot the CS for uniform and propM strategies; with and without
    logicalCS.

    methods_dict        :dict keys are strings denoting method names,
    and values are dicts that represent the additional keyword arguments
    to be sent into the function 'run_one_expt' fig_info_dict
    :dict either empty or contains the following keys 'figname',
    'title', 'xlabel', 'ylabel'
    """
    N = len(M)
    Pi = M / M.sum()
    grid = np.linspace(0, 1, nG)

    if len(methods_dict) == 0:
        raise Exception("No methods given!!!")
    #####################
    Results_Dict = {}
    for key in methods_dict:
        kwargs = methods_dict[key]
        _, _, LowerCS, UpperCS, Idx, _, diagnostics = run_one_expt(M,
                                                                   f,
                                                                   S,
                                                                   nG=nG,
                                                                   **kwargs)
        if post_process:
            LowerCS_, UpperCS_ = predictive_correction1(LowerCS,
                                                        UpperCS,
                                                        Idx=Idx,
                                                        Pi=Pi,
                                                        f=f,
                                                        logical=True,
                                                        intersect=True)
            if not return_post_processed_separately:
                LowerCS, UpperCS = LowerCS_, UpperCS_
        Results_Dict[key] = (LowerCS, UpperCS, Idx)
        if post_process and return_post_processed_separately:
            key_ = key + '+logical'
            Results_Dict[key_] = (LowerCS_, UpperCS_, Idx)

    if plot_results:
        plot_CS(N,
                M,
                f,
                diagnostics=diagnostics,
                Results_Dict=Results_Dict,
                plot_CS_width=plot_CS_width,
                save_fig=save_fig,
                fig_info_dict=fig_info_dict)
    if return_vals:
        return Results_Dict


def getStoppingTimesDistribution(methods_dict,
                                 N,
                                 N1,
                                 a=0.5,
                                 M_ranges=[[1e5, 1e6], [1e2, 1 * 1e3]],
                                 f_ranges=[[1e-3, 2 * 1e-3], [0.4, 0.5]],
                                 num_trials=200,
                                 nG=100,
                                 save_fig=False,
                                 post_process=True,
                                 epsilon=0.05,
                                 progress_bar=True,
                                 return_post_processed_separately=False,
                                 seed: Optional[float] = None):

    # initialize the dictionary to store the stopping times
    StoppingTimesDict = {}
    # Now run the main loop
    range_ = range(num_trials)
    if progress_bar:
        range_ = tqdm(range_)

    seed_seq = np.random.SeedSequence(seed) if seed is not None else [
        None for _ in range(num_trials)
    ]

    for trial in range_:

        M, f, S = generate_MFS(N_vals=(N1, N - N1),
                               N=N,
                               M_ranges=M_ranges,
                               f_ranges=f_ranges,
                               a=a,
                               seed=seed_seq[trial])
        # run one trial of the experiment
        Results_Dict = CompareMethodsOneTrial(
            M,
            f,
            S,
            methods_dict=methods_dict,
            nG=nG,
            save_fig=False,
            return_vals=True,
            plot_results=False,
            post_process=post_process,
            return_post_processed_separately=return_post_processed_separately)
        # calculate and record the new stopping times
        for key in Results_Dict:
            if key not in StoppingTimesDict:
                StoppingTimesDict[key] = np.zeros((num_trials, ))
            result = Results_Dict[key]
            L, U = result[0], result[1]
            W = U - L
            st = first_threshold_crossing(W,
                                          th=epsilon,
                                          max_time=N,
                                          upward=False)
            StoppingTimesDict[key][trial] = st
    return StoppingTimesDict
