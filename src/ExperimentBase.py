from typing import Any, Dict, List, Optional

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib as tpl
import seaborn as sns
from tqdm import tqdm

from weightedCSsequential import run_one_expt
from utils import predictive_correction1, generate_MFS, first_threshold_crossing
from constants import ColorsDict, LineStyleDict


# def plot_results1
def plot_CS(N: int,
            M: np.ndarray,
            f: np.ndarray,
            Results_Dict: Dict[str, Any],
            plot_CS_width: bool = False,
            palette: Optional[matplotlib.colors.Colormap] = None,
            save_fig: bool = False,
            fig_info_dict: Dict[str, Any] = {},
            diagnostics: Optional[Dict[str, np.ndarray]] = None,
            legend_flag: bool = False) -> None:
    """Plot (and save to disk) confidence sequence from experiment results
        Arguments
            N:
                Number of transactions
            M:
                Values of each transaction
            f:
                True misstated fraction for each transaction
            Results_Dict:
                Results (mapping between method name and result CS)
            plot_CS_width:
                If true plots the width of the CS, but plots the CS boundaries explicitly otherwise
            palette:
                Matplotlib palette to use for plotting
            save_fig:
                Whether to save the figure or not
            fig_info_dict:
                Additional keywords to pass to the plotting function
            diagnostics:
                Potential diagnostics of the experiment to plot
            legend_flag:
                Plot the legend iff it is true.
    """
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
        title = 'Confidence Sequence for $m^*$'
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
    if legend_flag:
        ax.legend(fontsize=13)
    fig.tight_layout()
    # save the figure
    diag_figs = []
    for method_key, diag in diagnostics.items():
        if diag is not None:
            name, lower_d, upper_d = diag
            if name == 'eb':
                diag_names = [
                    '$\\lambda_t$', '$V_t$', '$c_t$', 'Avg. $V_t$',
                    'Avg. $V_tc_t$', 'Pred. avg. $V_tc_t$',
                    'Pred. $\sum 1/c_t^2$', 'Ind margin t.', 'Total margin t.',
                    '$\widehat{\mu}_t(\\lambda_t)$', '$\widehat{\mu}_t$'
                ]
                ds = [('lower', lower_d), ('upper', upper_d)]
            else:
                diag_names = [
                    '$c_t$', '$\\lambda_t$', 'Margin', '$\\widehat{m}_t$',
                    '$\\widehat{\mu}_t(\\lambda_t)$', '$\\widehat{\mu}_t$'
                ]
                ds = [('hoef', lower_d)]
            for name, diag in ds:
                figtmp, axes = plt.subplots(len(diag_names),
                                            1,
                                            sharex=True,
                                            figsize=(10, len(diag_names) * 2),
                                            squeeze=False)
                for idx, (diag_name, arr) in enumerate(zip(diag_names, diag)):
                    t = np.arange(1, arr.shape[0] + 1)
                    axes[idx, 0].plot(t, arr, label=diag_name)
                    axes[idx, 0].set_title(diag_name)
                diag_figs.append((name, figtmp))

    if save_fig:
        figname_ = figname + '.tex' if save_fig is True else save_fig
        tpl.save(figname_,
                 figure=fig,
                 axis_width=r'\figwidth',
                 axis_height=r'\figheight')
        basename = os.path.splitext(figname_)[0]
        picname_ = basename + '.png'
        fig.savefig(picname_, dpi=300)
        for name, fig in diag_figs:
            fig.savefig(basename + f'_{name}_diagnostics.png', dpi=300)


def plot_hists(N: int,
               StoppingTimesDict: Dict[str, Any],
               palette: Optional[matplotlib.colors.Colormap] = None,
               plot_mean_val: bool = False,
               save_fig: bool = False,
               hist_info_dict: Dict[str, Any] = {},
               opacity: float = 0.5,
               ymax: Optional[float] = None,
               legend_flag: bool = False) -> None:
    """Plot (and save to disk) histogram of stoppimg times from experiment results
        Arguments
            N:
                Number of transactions
            StoppingTimesDict:
                Results (mapping between method name and distribution of stopping time result)
            palette:
                Matplotlib palette to use for plotting
            plot_mean_val:
                Plots the mean stopping time iff this is true
            save_fig:
                Whether to save the figure or not
            hist_info_dict:
                Additional keywords to pass to the plotting function
            opacity:
                Opacity of plot
            ymax:
                maximum y value in plot
            legend_flag:
                Plot the legend iff it is true.

    """
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
    if legend_flag:
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
    else:
        plt.show()


def plot_error_rate(N: int,
                    alpha: float,
                    StoppingTimesDict: Dict[str, Any],
                    palette: Optional[matplotlib.colors.Colormap] = None,
                    plot_mean_val: bool = False,
                    save_fig: bool = False,
                    hist_info_dict: Dict[str, Any] = {},
                    opacity: float = 0.5,
                    ymax: Optional[float] = None,
                    legend_flag: bool = False):
    """Plot (and save to disk) proportion of CSes don't cover the true parameter (m^*) from experiment results over time
        Arguments
            N:
                Number of transactions
            StoppingTimesDict:
                Results (mapping between method name and distribution of first time CS doesn't cover 0)
            palette:
                Matplotlib palette to use for plotting
            plot_mean_val:
                Plots the mean stopping time iff this is true
            save_fig:
                Whether to save the figure or not
            hist_info_dict:
                Additional keywords to pass to the plotting function
            opacity:
                Opacity of plot
            ymax:
                maximum y value in plot
            legend_flag:
                Plot the legend iff it is true.
    """
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

        error_rates = np.array([np.sum(ST <= i)
                                for i in range(N)]) / ST.shape[0]
        plt.plot(NN,
                 error_rates,
                 label=f'{key} (error {error_rates[-1]:.3f})',
                 alpha=opacity,
                 color=color)

    if len(hist_info_dict) == 0:
        title = r'CS error rates'
        xlabel = '# of samples'
        ylabel = 'Prop. of trials not covered in CS'
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
    if legend_flag:
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


def CompareMethodsOneTrial(
        M: np.ndarray,
        f: np.ndarray,
        S: np.ndarray,
        methods_dict: Dict[str, Any],
        nG: int = 100,
        save_fig: bool = False,
        fig_info_dict: Dict[str, Any] = {},
        return_vals: bool = False,
        plot_results: bool = True,
        plot_CS_width: bool = True,
        post_process: bool = True,
        return_post_processed_separately: bool = False,
        seed: Optional[int] = None,
        legend_flag: bool = False) -> Optional[Dict[str, Any]]:
    """Compare multiple methods on a single trial
        Arguments
            M:
                Transaction values
            f:
                True misstatement fractions
            s:
                Scores for each transaction
            methods_dict:
                Dictonary mapping method name to additional keyword arguments
            nG:
                number of grid items to be used for betting CS in experiment
            save_fig:
                save figure iff this is true
            fig_info_dict:
                Additional keywords to pass to the plotting function
            return_vals:
                Return results if true
            plot_results:
                Plot results if true
            plot_CS_width:
                Plot CS width (instead of CS) if true
            post_process, return_post_processed_separately:
                Post process CS by taking running intersection and intersecting with logical CS if true,
                and whether to return both post-processed and original CS

    """
    N = len(M)
    Pi = M / M.sum()
    grid = np.linspace(0, 1, nG)

    if len(methods_dict) == 0:
        raise Exception("No methods given!!!")
    #####################
    Results_Dict = {}
    diag_dict = {}
    for key in methods_dict:
        kwargs = methods_dict[key]
        _, _, LowerCS, UpperCS, Idx, _, diagnostics = run_one_expt(M,
                                                                   f,
                                                                   S,
                                                                   nG=nG,
                                                                   seed=seed,
                                                                   **kwargs)
        diag_dict[key] = diagnostics
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
        else:
            LowerCS_, UpperCS_ = predictive_correction1(LowerCS,
                                                        UpperCS,
                                                        Idx=Idx,
                                                        Pi=Pi,
                                                        f=f,
                                                        logical=False,
                                                        intersect=True)

        Results_Dict[key] = (LowerCS, UpperCS, Idx)
        if post_process and return_post_processed_separately:
            key_ = key + '+logical'
            Results_Dict[key_] = (LowerCS_, UpperCS_, Idx)

    if plot_results:
        plot_CS(N,
                M,
                f,
                diagnostics=diag_dict,
                Results_Dict=Results_Dict,
                plot_CS_width=plot_CS_width,
                save_fig=save_fig,
                fig_info_dict=fig_info_dict,
                legend_flag=legend_flag)
    if return_vals:
        return Results_Dict


def getStoppingTimesDistribution(
        methods_dict: Dict[str, Any],
        N: int,
        N1: int,
        a: float = 0.5,
        M_ranges: List[List[float]] = [[1e5, 1e6], [1e2, 1 * 1e3]],
        f_ranges: List[List[float]] = [[1e-3, 2 * 1e-3], [0.4, 0.5]],
        num_trials: int = 200,
        nG: int = 100,
        save_fig: bool = False,
        post_process: bool = True,
        epsilon: float = 0.05,
        progress_bar: bool = True,
        return_post_processed_separately: bool = False,
        seed: Optional[float] = None,
        use_CV:Optional[bool]=False,
        c :Optional[float]=0.1) -> Dict[str, np.ndarray]:
    """Run experiment comparing the stopping time distributions
        (first time CS width dips below epsilon) of different methods
        Arguments
            methods_dict:
                Dictonary mapping method name to additional keyword arguments
            N:
                Number of transactions
            N1:
                Number of 'large' transactions
            a:
                Parameter determining accuracy of scores generated (smaller is more accurate)
            M_ranges:
                List of 2 M ranges for 'large' and 'small' M values to be drawn from
            f_ranges:
                List of 2 f ranges for 'large' and 'small' f values to be drawn from
            num_trials:
                number of trials to run simulation for
            nG:
                number of grid items to be used for betting CS in experiment
            save_fig:
                save figure iff this is true
            post_process, return_post_processed_separately:
                Post process CS by taking running intersection and intersecting with logical CS if true,
                and whether to return both post-processed and original CS
            epsilon:
                Parameter determining CS threshold at which stopping time is determined
            progress_bar:
                Plot results if true
            seed:
                Random seed
            use_CV:
                flag indicating a control-variate experiment 
            c:
                float deciding the amount of correlation between f and S
        Return
            Dict mapping method name to array of stopping times for each trial
    """

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
        # modify the scores if using control variates 
        if use_CV:
            S = c*f + (1-c)*np.random.random(f.shape)
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


def getCoverage(methods_dict: Dict[str, Any],
                N: int,
                N1: int,
                a: float = 0.5,
                M_ranges: List[List[float]] = [[1e5, 1e6], [1e2, 1 * 1e3]],
                f_ranges: List[List[float]] = [[1e-3, 2 * 1e-3], [0.4, 0.5]],
                num_trials: int = 200,
                nG: int = 100,
                save_fig: bool = False,
                post_process: bool = True,
                epsilon: float = 0.05,
                progress_bar: bool = True,
                return_post_processed_separately: bool = False,
                seed: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Run experiment getting the empirical coverage rate of each CS
        (first time CS width dips below epsilon) of different methods
        Arguments
            methods_dict:
                Dictonary mapping method name to additional keyword arguments
            N:
                Number of transactions
            N1:
                Number of 'large' transactions
            a:
                Parameter determining accuracy of scores generated (smaller is more accurate)
            M_ranges:
                List of 2 M ranges for 'large' and 'small' M values to be drawn from
            f_ranges:
                List of 2 f ranges for 'large' and 'small' f values to be drawn from
            num_trials:
                number of trials to run simulation for
            nG:
                number of grid items to be used for betting CS in experiment
            save_fig:
                save figure iff this is true
            post_process, return_post_processed_separately:
                Post process CS by taking running intersection and intersecting with logical CS if true,
                and whether to return both post-processed and original CS
            epsilon:
                Parameter determining CS threshold at which stopping time is determined
            progress_bar:
                Plot results if true
            seed:
                Random seed
        Return
            Dict mapping method name to array of first time CS did not
            cover the true parameter for each trial (inf if the CS never miscovers the true parameter)
    """

    # initialize the dictionary to store the stopping times
    StoppingTimesDict = {}
    # Now run the main loop
    range_ = range(num_trials)
    if progress_bar:
        range_ = tqdm(range_)

    seed_seq = np.random.SeedSequence(seed) if seed is not None else [
        None for _ in range(num_trials)
    ]
    M, f, S = generate_MFS(N_vals=(N1, N - N1),
                           N=N,
                           M_ranges=M_ranges,
                           f_ranges=f_ranges,
                           a=a,
                           seed=seed_seq[0])

    m_star = np.sum(M * f / np.sum(M))
    for trial in range_:
        np.random.seed(seed_seq[trial])
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

            lb_mask = np.logical_or(L < m_star, np.isclose(L, m_star))
            if np.all(lb_mask):
                ub_mask = np.logical_or(U > m_star, np.isclose(U, m_star))
                if np.all(ub_mask):
                    res = np.inf
                else:
                    res = np.min(np.where(np.logical_not(ub_mask)))
            else:
                res = np.min(np.where(np.logical_not(lb_mask)))
            StoppingTimesDict[key][trial] = res
    return StoppingTimesDict
