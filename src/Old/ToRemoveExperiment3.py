"""
    Weighted CS without replacement and with general side-information: 
        Comparison of propM with propM+control-variates.
"""
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

from weightedCSsequential import run_one_expt
from utils import generate_MFS, predictive_correction1
from utils import first_threshold_crossing
from ExperimentBase import * 

from tqdm import tqdm 

def get_methods_dict(lambda_max=2, beta_max=0.5):
    """
    Generate the methods dictionary for this experiment 
    """
    methods_dict = {}
    methods_dict['propM'] =  {'method_name':'propM', 'use_CV':False, 'lambda_max':lambda_max}
    methods_dict['propM+CV'] =  {'method_name':'propM', 'use_CV':True, 'lambda_max':lambda_max,
                                'beta_max':beta_max}
    return methods_dict

def CSexperiment3(M_ranges, f_ranges, N=200, N1=60,lambda_max=2, a=0.5, inv_prop=True, 
                    nG=100, save_fig=False, plot_CS_width=True):
    N2 = N-N1
    if inv_prop:
        title = f'{N1/N:.0%} large ' +r'$\pi$ values,    $f \propto 1/\pi$'
        figname = f'../data/Control_Variates_inv_propto_M_large_{N1}.png'
        f_ranges = [f_ranges[1], f_ranges[0]]
    else:
        title = f'{N1/N:.0%} large ' +r'$\pi$ values,    $f \propto \pi$'
        figname = f'../data/Control_Variates_propto_M_large_{N1}.png'
    # generate the problem instance
    M, f, S =  generate_MFS(N_vals=(N1, N2), 
                N=N, 
                M_ranges = M_ranges, 
                f_ranges = f_ranges, 
                a = a)
    # create the methods dict 
    methods_dict = get_methods_dict(lambda_max=lambda_max)
    # create the figure information dictionary 
    xlabel = r'Sample Size ($n$) '
    if plot_CS_width:
        ylabel = 'Width of CS'
    else:
        ylabel = ' '
    fig_info_dict = {'title':title, 'figname':figname, 'xlabel':xlabel,
                        'ylabel':ylabel}
    # run one trial of the experiment 
    CompareMethodsOneTrial(M, f, S, methods_dict,  nG=nG, save_fig=save_fig,
                            fig_info_dict=fig_info_dict, return_vals=False,
                            plot_results=True, plot_CS_width=plot_CS_width,
                            post_process=True)


def HistExperiment3(M_ranges, f_ranges, N=200, N1=100, epsilon=0.05, inv_prop=True, 
                    verbose=False, plot_results=False, save_fig=False, 
                    a=0.5, lambda_max=2.5, beta_max=0.5, 
                    num_trials=20, return_vals=False):
    if inv_prop:
        title = f'Stopping Times Distribution \n' +f'{N1/N:.0%} large ' +r'$\pi$ values,    $f \propto 1/\pi$'
        figname = f'../data/AccurateSideInfoHist_f_inv_propto_M_large_{N1}.png'
        f_ranges = [f_ranges[1], f_ranges[0]]
    else:
        title = f'Stopping Times Distribution \n' +f'{N1/N:.0%} large ' +r'$\pi$ values,    $f \propto \pi$'
        figname = f'../data/AccurateSideInfoHist_f_propto_M_large_{N1}.png'
        # assume the original f values are proportional to M 
        # so we don't need any modification here
    # generate the dictionary with methods information 
    methods_dict = get_methods_dict(lambda_max=lambda_max, beta_max=beta_max)
    # generate the dictionary with histogram plotting information 
    StoppingTimesDict = getStoppingTimesDistribution(methods_dict, N, N1, 
                            a=a, M_ranges=M_ranges, f_ranges=f_ranges,
                            num_trials=num_trials, save_fig=False, post_process=True, 
                            epsilon=epsilon)

    if verbose: 
        for key in StoppingTimesDict:
            st = StoppingTimesDict[key].mean()
            print(f'Method={key}, \t StoppingTimeAverage = {st:.2f}')
    
    if plot_results:
        xlabel='Stopping Times'
        ylabel='Density'
        hist_info_dict = {
            'title':title, 'figname':figname, 'xlabel':xlabel, 
            'ylabel':ylabel
        }
        plot_hists(N=N, StoppingTimesDict=StoppingTimesDict,
                    save_fig=save_fig, hist_info_dict=hist_info_dict, 
                    opacity=0.5) 
    if return_vals:
        return StoppingTimesDict

def main1(M, f, S,  nG=100, savefig=False, figname=None,
            title=None, return_vals=False, plot_results=True, f_over_S_range=None, 
            lambda_max=2, return_error_flag=False, intersect=True, beta_max=0.5):
    """
    plot the CS for uniform and propM strategies; with and without logicalCS 
    """
    N = len(M)
    Pi = M / M.sum()
    grid, _, LowerCS1, UpperCS1, Idx1, E1 = run_one_expt(M, f, S, nG=nG,
                                                    method_name='propM', 
                                                    lambda_max=lambda_max, 
                                                    use_CV=False, 
                                                    f_over_S_range=f_over_S_range)

    if intersect:
        LowerCS1, UpperCS1 = predictive_correction1(LowerCS1, UpperCS1, 
                                                Idx=Idx1, Pi=Pi, f=f, 
                                                logical=True, intersect=True)

    _, _, LowerCS2, UpperCS2, Idx2, E2 = run_one_expt(M, f, S, nG=nG,
                                                    method_name='propM', 
                                                    lambda_max=lambda_max, 
                                                    use_CV=True, 
                                                    beta_max=beta_max)

    if intersect:
        LowerCS2, UpperCS2 = predictive_correction1(LowerCS2, UpperCS2, 
                                                Idx=Idx2, Pi=Pi, f=f, 
                                                logical=True, intersect=True)

    if plot_results:
        NN = np.arange(1, N+1) 
        palette = sns.color_palette(n_colors=10)
        plt.figure()

        plt.plot(NN, UpperCS1-LowerCS1,  label='propM', color=palette[2])
        plt.plot(NN, UpperCS2- LowerCS2,  label='propM+CV', color=palette[1])
        plt.legend(fontsize=13)
        plt.xlabel('Sample Size (n)', fontsize=14)
        plt.ylabel('Width of CS', fontsize=14)
        title = 'CS Width vs Sample Size' if title is None else title 
        plt.title(title, fontsize=15)

        if savefig:
            if figname is None:
                figname = './data/ControlVariatesCS.png' 
            plt.savefig(figname, dpi=450)
        else:
            plt.show()
    
    if return_vals:
        if return_error_flag:
            E = E1 or E2
            return LowerCS1, UpperCS1, Idx1, LowerCS2, UpperCS2, Idx2, E
        else:
            return LowerCS1, UpperCS1, Idx1, LowerCS2, UpperCS2, Idx2


def main2(M, f, S, epsilon=0.05, f_over_S_range=None, lambda_max=2, 
            max_counts=20, intersect=True, beta_max=0.5):
    """
    calculate the stopping time for uniform and propM strategies 
    """
    Pi = M/M.sum()
    # get the lower and upper CS for uniform and propM strategies 
    Error = True 
    count = 0 
    while Error and count < max_counts:
        LowerCS1, UpperCS1, Idx1, LowerCS2, UpperCS2, Idx2, Error = main1(M, f, S,  nG=100, 
        savefig=False, figname=None, title=None, return_vals=True, plot_results=False,
        f_over_S_range=f_over_S_range, lambda_max=lambda_max, return_error_flag=True,
        intersect=intersect, beta_max=beta_max)
        count +=1 
    if count==max_counts:
        raise Exception(f'All the {max_counts} attempts to compute a valid CS failed!!!!')
    
    # get the improved versions of these CSes 
    if intersect:
        LowerCS1, UpperCS1 = predictive_correction1(LowerCS1, UpperCS1, 
                                                Idx=Idx1, Pi=Pi, f=f, 
                                                logical=True, intersect=True)
        LowerCS2, UpperCS2 = predictive_correction1(LowerCS2, UpperCS2, 
                                                Idx=Idx2, Pi=Pi, f=f, 
                                                logical=True, intersect=True)
    W1 = UpperCS1-LowerCS1
    W2 = UpperCS2-LowerCS2

    N = len(M)
    # get the stopping times 
    ST1 = first_threshold_crossing(W1, th=epsilon, max_time=N, upward=False)
    ST2 = first_threshold_crossing(W2, th=epsilon, max_time=N, upward=False)
    return ST1, ST2


def main3(N, N1, M_ranges = [ [1e5, 1e6], [1e2, 1*1e3]], 
                f_ranges = [[1e-3, 2*1e-3], [0.4, 0.5]],  
                num_trials = 200, epsilon=0.05, plot_results =True, 
                return_vals=False, title=None, figname=None, 
                savefig=False, f_over_S_range=None, A=0.1, 
                lambda_max=2, progress_bar=True, beta_max = 0.5
):
    """
    plot the stopping time distributions 
    """
    N2 = N-N1
    ST1 = np.zeros((num_trials, ))
    ST2 = np.zeros((num_trials, ))
    ST1_ = np.zeros((num_trials, ))
    ST2_ = np.zeros((num_trials, ))
    range_ = range(num_trials) 
    if progress_bar:
        range_=tqdm(range_)
    for trial in range_:
        M, f, S =  generate_MFS(N_vals=(N1, N2), 
                    N=N, 
                    M_ranges = M_ranges, 
                    f_ranges = f_ranges, 
                    a = A)
        st1, st2 = main2(M, f, S, epsilon=epsilon, f_over_S_range=f_over_S_range,
                            lambda_max=lambda_max, beta_max=beta_max)
        ST1[trial] = st1
        ST2[trial] = st2
    
    if plot_results: 
        palette = sns.color_palette(n_colors=10)
        plt.figure()
        plt.hist(ST1, label='propM', color=palette[2], alpha=0.8, density=True)
        plt.hist(ST2, label='propM+CV', color=palette[1], alpha=0.8, density=True)
        plt.legend(fontsize=13)
        if title is None:
            title = r'Distribution of Stopping Times ($epsilon$='+f'{epsilon})'
        plt.title(title, fontsize=15)
        plt.xlabel(r'Stopping Time ($\tau$)', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        if savefig: 
            if figname is None:
                figname = '../data/histogram.png'
            plt.savefig(figname, dpi=450)
        else: 
            plt.show()
    if return_vals:
        return ST1, ST2


if __name__=='__main__':

    CSExpt = False
    HistExpt = False
    GainExpt = True
    savefig=False

    A=0.1
    N = 200 
    N1 = 100
    N2 = N-N1
    epsilon=0.05
    f_over_S_range = [1-A, 1+A]
    inv_prop=False
    M_ranges = [ [1e5, 1e6], [1e5, 1*1e6]]
    lambda_max = 2.5

    # M_ranges = [ [1e5, 1e6], [1e2, 1*1e3]],  
    if CSExpt:
        
        if inv_prop:
            title = f'{N1/N:.0%} large ' +r'$\pi$ values,    $f \propto 1/\pi$'
            figname = f'../data/NoSideInfoCSf_inv_propto_M_large_{N1}.png'
            f_ranges = [[1e-3, 2*1e-3], [0.4, 0.5]]
        else:
            title = f'{N1/N:.0%} large ' +r'$\pi$ values,    $f \propto \pi$'
            figname = f'../data/NoSideInfoCSf_propto_M_large_{N1}.png'
            f_ranges = [[0.4, 0.5], [1e-3, 2*1e-3]]
 

        title = 'CS with accurate side-information'
        figname = f'../data/Presentation_fig14_CS.png'
        M, f, S =  generate_MFS(N_vals=(N1, N2), 
                    N=N, # total number of transactions = sum(N_vals)
                    M_ranges = M_ranges,  
                    f_ranges = f_ranges, 
                    a = A)
        main1(M, f, S,  nG=100, savefig=savefig, figname=figname, title=title, 
                f_over_S_range=f_over_S_range, lambda_max=lambda_max)



    if HistExpt:
        A = 0.99
        f_over_S_range = [1-A, 1+A]
        if inv_prop:
            title = f'Stopping Times Distribution \n' +f'{N1/N:.0%} large ' +r'$\pi$ values,    $f \propto 1/\pi$'
            figname = f'../data/AccurateSideInfoHist_f_inv_propto_M_large_{N1}.png'
            f_ranges = [[1e-3, 2*1e-3], [0.4, 0.5]]
        else:
            title = f'Stopping Times Distribution \n' +f'{N1/N:.0%} large ' +r'$\pi$ values,    $f \propto \pi$'
            figname = f'../data/AccurateSideInfoHist_f_propto_M_large_{N1}.png'
            f_ranges = [[0.4, 0.5], [1e-3, 2*1e-3]]
 
        title = f'Stopping Times Distribution'
        figname = '../data/Presentation_fig_14_Hist.png'
        ST, STcv = main3(N, N1, M_ranges = M_ranges, 
                        f_ranges = f_ranges,  
                        num_trials = 20, epsilon=epsilon, plot_results =True, 
                        return_vals=True, title=title, figname=figname, 
                        savefig=savefig, A=A, f_over_S_range=f_over_S_range, 
                        lambda_max=lambda_max, beta_max=0.0)


    if GainExpt:
        AA = np.linspace(0.1, 0.9, 9)
        Gain = np.zeros(AA.shape)
        num_trials = 50 
        M_ranges = [ [1e2, 1e3], [6e2, 9e2]]
        epsilon = 0.02
        ###############################
        for i, A_ in tqdm(list(enumerate(AA))):
            f_over_S_range_ = [1-A_, 1+A_]
            if inv_prop:
                title = f'Stopping Times Distribution \n' +f'{N1/N:.0%} large ' +r'$\pi$ values,    $f \propto 1/\pi$'
                figname = f'../data/AccurateSideInfoHist_f_inv_propto_M_large_{N1}.png'
                f_ranges = [[1e-3, 2*1e-3], [0.4, 0.5]]
            else:
                title = f'Stopping Times Distribution \n' +f'{N1/N:.0%} large ' +r'$\pi$ values,    $f \propto \pi$'
                figname = f'../data/AccurateSideInfoHist_f_propto_M_large_{N1}.png'
                f_ranges = [[0.4, 0.5], [1e-3, 2*1e-3]]
    
            title = f'Stopping Times Distribution'
            figname = '../data/Presentation_fig_14_Hist.png'
            ST, STcv = main3(N, N1, M_ranges = M_ranges, 
                            f_ranges = f_ranges,  num_trials = num_trials,
                            epsilon=epsilon, plot_results =False, 
                            return_vals=True, savefig=False, A=A_, f_over_S_range=None, 
                            lambda_max=lambda_max, progress_bar=False)
            
            gain = (STcv/ST).mean()
            Gain[i] = gain
            print(f'Gain with A={A_:.1f} is {gain:.2f}')
        plt.figure()
        plt.plot(1-AA, Gain)
        plt.plot(AA, np.ones(AA.shape), 'k--', alpha=0.6)
        plt.title('Relative Reduction in Sample-Size by using Control Variates')
        plt.xlabel('A (a measure of correlation)')
        plt.ylabel('Average of the ratio of stopping times')

