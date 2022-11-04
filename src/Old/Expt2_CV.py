from weightedCS import generate_MFS, get_sampling_weights, get_conf_seq, sample_indices 
from weightedCS import control_variate_values, control_variate_weights, get_bets
from weightedCS import get_wealth_process, sample_indices
from tqdm import tqdm 
from utils import plot_results1
import numpy as np 
import matplotlib.pyplot as plt 



def first_threshold_crossing(arr, th, max_time=100, upcross=True):
    if not upcross: 
        arr *= -1 
        th *= - 1
    if np.any(arr>th):
        return np.argmax(arr>th)+1
    else:
        return max_time



def get_StoppingTime_ratio(num_trials, l, delta_m=0.01, progress_bar=False, 
                            random_seed=None, return_stopping_times=False, 
                            alpha=0.05, lambda_max=0.5, beta_max=0.5):
    if random_seed is not None:
        np.random.seed(random_seed)
    print(f'{num_trials} trials,  l={l:.2f}, N={N}')
    StoppingTimes, StoppingTimesCV = np.zeros((num_trials)), np.zeros((num_trials))

    range_ = range(num_trials)
    range_ = tqdm(range_) if progress_bar else range_
    # for trial in tqdm(range(num_trials)):
    for trial in range_:
        M, f, S = generate_MFS(N_vals=N_vals, N=N, M_ranges=M_ranges, f_ranges=f_ranges, a=a)
        Pi = M/M.sum() 
        m_star = np.sum(Pi*f) 
        m = min(max(1e-5, m_star + delta_m), 1-1e-5)

        method = lambda M, S: M #propM strategy
        use_propMS = False
        S = f*l + (1-l)*np.random.random((N,))
        SW = get_sampling_weights(M=M, S=S, method=method)
        max_f_S_ratio = max(np.abs(f/S))
        
        Idx = sample_indices(N, SamplingWeights=SW) 


        W, _, _, _ = get_wealth_process(m, f, Pi, Idx, SamplingWeights=SW, S=S,  
                                use_propMS=use_propMS, max_lambda_val=lambda_max, max_f_S_ratio=max_f_S_ratio,
                                use_CV=False, beta_max=beta_max)
        StoppingTimes[trial] = first_threshold_crossing(W, th=1/alpha, max_time=N)
        
        Wcv, _, _, _ = get_wealth_process(m, f, Pi, Idx, SamplingWeights=SW, S=S,  
                                use_propMS=use_propMS, max_lambda_val=lambda_max, max_f_S_ratio=max_f_S_ratio,
                                use_CV=True, beta_max=beta_max)
        
        StoppingTimesCV[trial] = first_threshold_crossing(Wcv, th=1/alpha, max_time=N)


    # ratio = StoppingTimes.mean() / StoppingTimesCV.mean()
    ratio = (StoppingTimesCV / StoppingTimes).mean()
    if return_stopping_times:
        return ratio, StoppingTimes, StoppingTimesCV
    else:
        return ratio
    

if __name__ == '__main__':

    # First generate the data 
    N1, N2  = 150, 150
    N = N1+N2
    N_vals = [N1, N2]
    M_ranges = [[1e4, 2e4], [1e5, 2*1e6]] 
    f_ranges = [[0.4, 0.5], [1e-2, 2*1e-2]] 
    a=0.1

    lambda_max = 8
    beta_max = 2.5

    num_trials = 20
    ll = np.linspace(0.1, 0.80, 8) 
    Ratio = np.zeros(ll.shape)
    delta_m = 0.05
    # random_seed = np.random.randint(low=0, high=10000)
    # random_seed = 7753
    random_seed = 2954
    StoppingTimesDict, StoppingTimesDictCV = {}, {}

    for i, l in enumerate(ll):
        # print(f'({i+1}/{len(ll)})') 
        Ratio[i], stop, stopcv = get_StoppingTime_ratio(num_trials=num_trials,
                                        l=l, delta_m=delta_m, 
                                        random_seed=random_seed, 
                                        return_stopping_times=True, 
                                        lambda_max=lambda_max, 
                                        beta_max=beta_max)
    
        StoppingTimesDict[l] = stop
        StoppingTimesDictCV[l] = stopcv
    
################################################################################
    save_fig=False
################################################################################
    plt.plot(ll, np.ones(ll.shape), '--', label='propM (normalized)')
    plt.plot(ll, Ratio, label='propM + control variates')
    title = 'Reduction in number of queries by using \n Control Variates '+f'for accuracy={delta_m}'
    plt.title(title, fontsize=15)
    plt.ylabel('Average Ratio of # of queries ', fontsize=14)
    plt.xlabel('Correlation between $S$ and $f$', fontsize=14)
    plt.legend(loc="lower left", fontsize=13)
    if save_fig:
        figname = f'Stopping_Time_ratio_(random_seed={random_seed})_.png'
        plt.savefig(figname, dpi=450)

    plt.figure()
    fig, axs = plt.subplots(nrows=2, ncols=1)
    l_ = ll[1]
    Ratio_dist = StoppingTimesDictCV[l_]/StoppingTimesDict[l_]
    axs[0].hist(Ratio_dist, bins=20,  alpha=0.4, density=True)
    axs[0].axvline(x=1.0, ls='--', color='k')
    axs[0].legend(fontsize=13)
    axs[0].set_xlabel('Ratio of stopping times', fontsize=13)
    axs[0].set_ylabel(f'corr={l_:.1f}', fontsize=13)
    axs[0].set_title(f'Distribution of the ratio of # of calls to auditor', fontsize=15)
    axs[0].set_xlim([0.4, 1.5])

    l_ = ll[5]
    Ratio_dist = StoppingTimesDictCV[l_]/StoppingTimesDict[l_]
    axs[1].hist(Ratio_dist, bins=50,  alpha=0.4, density=True)
    axs[1].axvline(x=1.0, ls='--', color='k')
    axs[1].legend(fontsize=13)
    axs[1].set_xlabel('Ratio of stopping times', fontsize=13)
    axs[1].set_ylabel(f'corr={l_:.1f}', fontsize=13)
    axs[1].set_xlim([0.4, 1.5])
    # axs[1].set_title(f'Distribution of Gains with CV', fontsize=15)
    if save_fig:
        figname = f'Stopping_Time_hist_(random_seed={random_seed})_.png'
        plt.savefig(figname, dpi=450)
        
