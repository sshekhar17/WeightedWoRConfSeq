from random import random, sample
import numpy as np 
import scipy.stats as stats 
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm 
plt.style.use('seaborn-white')
from utils import brute_force_CS_solver2, predictive_correction1



def generate_MFS(N_vals=(100, 100), 
                N=200, # total number of transactions = sum(N_vals)
                M_ranges = [ [1e3, 1e4], [1e5, 2*1e5]],  
                f_ranges = [[0.4, 0.5], [1e-3, 2*1e-3]], 
                a = 0.1):
    """
    Generate synthetic M, f, S values 

    Parameters
       N_vals   :       number of transactions in different components 
       N        :       total number of transactions = sum(N_vals)
       M_ranges :       range of M-values in different components
       f_ranges :       range of f-values in different components
       a        :       relative error in generating S from f
                            a \leq f/S \leq 2 - a. 
    Returns 
        M, f, S :       (N, ) numpy arrays 
    """
    N_ = 0
    for n in N_vals:
        N_ += n
    if N_!=N:
        print('N doesn"t match sum of N_vals!!')
        N = N_
    M = np.empty((N,))   
    f = np.empty((N,))
    num_components = len(N_vals)
    assert num_components == len(M_ranges)
    assert num_components == len(f_ranges)
    n_=0
    for i in range(num_components):
        n = N_vals[i]
        M_lower, M_upper = M_ranges[i][0], M_ranges[i][1]
        del_M = M_upper-M_lower
        f_lower, f_upper = f_ranges[i][0], f_ranges[i][1]
        del_f = f_upper-f_lower

        M[n_:n_+n] = np.random.random((n,))*del_M + M_lower
        f[n_:n_+n] = np.random.random((n,))*del_f + f_lower

        n_ += n 
    S = (a + (2*(1-a))*np.random.random((N,)) ) * f 
    S[S>=1] = 1.0
    return M, f, S



def get_sampling_weights(M, S=None, method='propM'):
    """
    Return the weights to drive the sampling without replacement strategy 

    Parameters 
        M       :(N,) numpy array 
                    Monetary value of transactions
        S       :(N,) numpy array                   
                    Side information
        method  :either string or a function-handle                          
                    * allowed values 'propM', 'propMS'
                    * if function-handle, it takes in as inputs M and S 
                        and returns an (N,) array  SamplingWeights
    Returns 
        SamplingWeights     :(N,) numpy array 
                                ith value specifies the weight to be given 
                                to index i in the sampling without replacement 
                                procedure.
    """
    if method=='propM':
        SamplingWeights = M 

    elif method=='propMS':
        if S is None:
            print('No side information to use propMS: switching to propM')
            S = np.ones(M.shape)
        SamplingWeights=M*S 
    elif method=='uniform':
        N = len(M)
        SamplingWeights = np.ones((N,))/N
    elif callable(method):
        SamplingWeights=method(M=M, S=S)

    return SamplingWeights



def sample_indices(N, SamplingWeights=None):
    """
    Return a sequence of indices queried according to the SampleingWeights
    """
    if SamplingWeights is None:
        # default is uniform sampling 
        SamplingWeights = np.ones((N,))
    
    Idx = np.empty((N,))
    remaining = [i for i in range(N)]

    for t in range(N-1):
        # the sampling distribution at time t
        Q_t_ = np.array([SamplingWeights[j] for j in remaining])
        assert(np.sum(Q_t_)>0)
        Q_t = Q_t_/np.sum(Q_t_)
        # draw the ith sample from the remaining indices
        idx = np.random.choice(a=remaining, p=Q_t)
        i_ = remaining.index(idx)
        # update the index set 
        Idx[t] = idx
        # update the remaining set 
        remaining.remove(idx)
    # the last index to be queried
    Idx[-1] = remaining[-1]
    # change the Idx data type to int (needed for indexing)
    Idx = Idx.astype(int)
    return Idx


def get_sampling_probability(Idx, SamplingWeights, return_min=False):
    """
    Return the probabilities assigned by the sampling distributions 
    to the sequence of indices observed (i.e., Idx)

    Parameters
        Idx             :(N,) index of transactions queried
        SamplingWeights :(N,) sampling weights assigned to transactions
        return_min      :bool if True, return the running minimum 
                            of probability vectors 

    Return: 
        Qt  :(N,) array of probabilities assigned to elements of Idx 
    """
    N=len(Idx)
    assert N==len(SamplingWeights) 
    #re-order the sampling weights according to the indices drawn
    Idx = Idx.astype(int)
    # permute the Sampling weights according to Idx
    SW = SamplingWeights[Idx] 

    # initialize the Qt and Qt_min arrays
    Qt = np.ones((N,))
    if return_min:
        Qt_min = np.ones((N,))

    # for i in Idx[:-1]:
    for i in range(N-1):
        qt = SW[i:]
        assert(qt.sum()>0)
        qt /= qt.sum()
        # store the probability assigned to next index 
        Qt[i] = qt[0]
        # store the minimax probability assigned by 
        # qt to any index not yet queried
        if return_min:
            Qt_min[i] = min(qt)
    if return_min:
        return Qt, Qt_min
    else:
        return Qt


def get_sampling_distribution(t, Idx, SamplingWeights, return_rearranged=True):
    """
    Return the probability vector used for sampling the index at time t 


    Parameters
        t                   :time 
        Idx                 :(N,) array of indices queried
        SamplingWeights     :(N,) array of sampling weights assigned to transactions
        return_rearranged   :bool, if True, the vector q_t is 
                                rearranged according to the array Idx
    Return 
        qt      :(N-t+1,) size if return_rearranged, otherwise 
                 (N,) size, with zeros at index Idx[0], ..., Idx[t-1]
    """
    if return_rearranged:
        # rearrange the sampling weights 
        Idx = Idx.astype(int)
        SamplingWeights = SamplingWeights[Idx]
        qt_ = SamplingWeights[t:]
    else:
        qt_ = np.zeros(Idx.shape)
        for idx in Idx[t:]:
            qt_[idx]=SamplingWeights[idx]
    # normalize the probability distribution
    assert(qt_.sum()>0)
    qt = qt_/qt_.sum()
    return qt


def get_bets(m, Idx, f, SamplingWeights, Pi=None, tol=1e-10):
    """
    Compute the bets for the given value of m 

    Parameters 
        m                   :float, betting against this value of m^*
        Idx                 :(N,) array of sequence of indices sampled
        f                   :(N,) array of f values associated with transactions
        SamplingWeights     :(N,) array of sampling weights given to each transaction
        Pi                  :(N,) pmf vector 
    
    Return 
        bets        :(N,) array of centered bets 
        Z           :(N,) the un-centered betting function 
                            Z_t = \pi(I_t) f(I_t) / q_t(I_t) 
        cond_mean   :(N,) array of the conditional mean under the 
                        null hypothesis that $m^*=m$. 
                            \mu_t = m - \sum_{s=1}^{t-1} \pi(I_s) f(I_s)
    """
    N = len(Idx)
    assert (N==len(f)) and (N==len(SamplingWeights)) 
    # set default weighting to uniform
    if Pi is None:
        Pi = np.ones((N,))/N
    # get the probabilities assigned to Idx
    Qt = get_sampling_probability(Idx, SamplingWeights=SamplingWeights, 
                                    return_min=False)
    # define the bet Z, obtained via importance sampling
    Idx = Idx.astype(int)
    f_, Pi_ = f[Idx], Pi[Idx]
    Z = f_*Pi_/(Qt+tol)
    # obtain the conditional mean of the bets (assuming m=m^*)
    Mu_t = np.zeros((N,))
    Mu_t[1:] = np.cumsum(Pi_[:-1]*f_[:-1])
    cond_mean = m - Mu_t
    # return the centered bets 
    bets = Z - cond_mean 
    return bets, Z, cond_mean 



def get_lambda1(Bets, method='kelly', tol=1e-10):
    """
    Return the array of bet values 

    Parameters
        Bets        :(N,) array of centered bets
        method      :string, allowed options are 'kelly' and 'ONS'
    
    Return 
        Lambda      :(N,) array of bets
                        Note that these values of Lambda may not 
                        satisfy the constraints to ensure the 
                        non-negativity of the wealth process. 
                        We will have to clip the values of lambda by 
                        calling 'get_lambda_bound1' for that. 
    """
    N = len(Bets)
    if method=='kelly':
        B_ = np.cumsum(Bets[1:])
        B2_ = np.cumsum(Bets[1:]*Bets[1:])
        Lambda = np.zeros((N,)) 
        Lambda[1:] = B_/(B2_+tol)
    else:
        raise Exception('Only implemented Kelly betting')
    return Lambda 


def get_lambda_bounds1(Idx, SamplingWeights, cond_mean,
                        Pi, tol=1e-10, max_lambda_val=2, 
                        use_propMS=False, max_f_S_ratio=1.0, 
                        use_propM=False, S=None, use_CV=False, 
                        Beta=None):
    N = len(Idx)
    # sanity check
    assert(len(SamplingWeights)==N) and (len(cond_mean)==N)
    if use_CV: 
        if Beta is None:
            Beta = 0.5*np.ones(Pi_.shape())
        elif isinstance(Beta, float):
            Beta = Beta*np.ones(Pi_.shape())
        else:
            assert isinstance(Beta, np.ndarray)
    # initialize the vectors of lower and upper bounds on lambda
    lower, upper = np.zeros((N,)), np.zeros((N,))
    # rearrange Pi according to Idx
    Idx = Idx.astype(int)
    Pi_ = Pi[Idx]
    for t in range(N):
        # get the term max_{i} {\pi(i)/q_t(i)}
        qt = get_sampling_distribution(t, Idx, SamplingWeights,
                                         return_rearranged=True)
        if use_propMS: #TODO: remove heuristic
            if S is None:
                S = np.ones((N,))
            at = max_f_S_ratio * np.sum(Pi[t:] * S[t:])
        elif use_propM:
            at = 1 * np.sum(Pi[t:])
        else:
            at = np.max(Pi_[t:]/(qt+tol)) 
        # get the bounds 
        mu_t = cond_mean[t] 
        if not use_CV:
            if mu_t<=0: # 0-mu_t > 0 
                lower[t] = max(-1/(at-mu_t), -max_lambda_val)
                upper[t] = max_lambda_val
            elif at-mu_t>0:
                lower[t] = max(-1/(abs(at-mu_t)+tol), -max_lambda_val)
                upper[t] = min(1/(mu_t+tol), max_lambda_val)
            else:
                lower[t] = max(-max_lambda_val, -max_lambda_val)
                upper[t] = min(1/(mu_t+tol), max_lambda_val)
        else: # use Control Variates 
            max_val_ = at - mu_t + Beta[t]
            min_val_ =  0 - mu_t - Beta[t] 
            if max_val_ < 0:
                lower[t] = -max_lambda_val 
                upper[t] = min(1/(tol + abs(min_val_)), max_lambda_val)
            elif max_val_>0 and min_val_<0:
                lower[t] = max(-max_lambda_val, -1/(max_val_ + tol))
                upper[t]  = min(max_lambda_val, 1/(abs(min_val_)+tol))
            elif min_val_ > 0:
                lower[t] = max(-max_lambda_val, -1/(max_val_ + tol))
                upper[t] = max_lambda_val 
            
    return lower, upper 

def get_lambda_bounds_CV(m, Idx, cond_mean, Pi, tol=1e-10, max_lambda_val=2, 
                        S=None, Beta=None):
    Idx = Idx.astype(int) 
    S_, Pi_ = S[Idx], Pi[Idx]
    N = len(Idx)
    meanS = np.zeros((N,))
    for i in range(N):
        meanS[i] = np.sum(Pi_[i:] * S_[i:]) / np.sum(Pi_[i:])
    # get the max and min values 
    minB, maxB = -cond_mean, (1-m)*np.ones((N,))
    minS = np.minimum(Beta * (1-meanS), -Beta * meanS)
    maxS = np.maximum(Beta * (1-meanS), -Beta * meanS)
    minCV = minB - maxS
    maxCV = maxB - minS
   # get the lower and uppwer values 
    lower = np.ones((N,))*(-max_lambda_val)
    upper = np.ones((N,))*max_lambda_val
    for t in range(N):
        if minCV[t]> 0: 
            lower[t] = (-1/(maxCV[t]+tol))
            upper[t] = max_lambda_val
        elif maxCV[t]>0>minCV[t]:
            lower[t] = (-1/(maxCV[t]+tol))
            upper[t] = (-1/(minCV[t]+tol))
        else:
            lower[t] = -max_lambda_val 
            upper[t] = (-1/(minCV[t] + tol))
    lower = np.maximum(-max_lambda_val, lower)
    upper = np.minimum(max_lambda_val, upper)
    return lower, upper


def control_variate_values(S, Idx, SamplingWeights):
    N = len(Idx) 
    assert len(SamplingWeights)==len(S)==N 
    Idx = Idx.astype(int)
    S_ = S[Idx]
    cv_vals = np.zeros((N,)) 
    for t in range(N-1):
        q_t = get_sampling_distribution(t, Idx, SamplingWeights, return_rearranged=True)
        s_mean_t = (q_t * S_[t:]).sum() 
        cv_vals[t] = S_[t] - s_mean_t 
    return cv_vals 


def control_variate_weights(Bets, cv_vals, beta_max=0.5):
    N = len(Bets)
    assert len(cv_vals)==N

    Beta = np.zeros((N,)) 
    for i in range(1, N):
        C = cv_vals[:i] 
        B = Bets[:i] 

        numerator = (C * B).sum()
        denominator = (C * C).sum() 
        Beta[i] = -(numerator / denominator)
    Beta = np.clip(Beta, a_min=-beta_max, a_max=beta_max)
    return Beta 

def get_wealth_process(m, f, Pi, Idx, SamplingWeights, S=None,  betting_method='kelly', 
                        use_propMS=False, max_lambda_val=2.0, max_f_S_ratio=1.0,
                        use_CV=False, beta_max=0.5, return_bets=False):
    N = len(f) 
    # step 1: get the bets 
    SW = SamplingWeights
    Bets, Z, cond_mean = get_bets(m=m, Idx=Idx, f=f, SamplingWeights=SW, Pi=Pi) 
    # step 1.5: get the control variates 
    Beta = None
    if use_CV: 
        assert (S is not None)
        cv_vals = control_variate_values(S, Idx, SamplingWeights)
        Beta = control_variate_weights(Bets, cv_vals, beta_max=beta_max)
        Bets = Bets - (Beta * cv_vals)
    # step 2: get the lambda values 
    Lambda = get_lambda1(Bets, method=betting_method) 
    # step 3: get the bounds on lambda values 
    # if use_CV:
    #     lower, upper = get_lambda_bounds_CV(m, Idx, cond_mean, Pi, 
    #                                         max_lambda_val=max_lambda_val,
    #                                         S=S, Beta=Beta)
    # elif not use_propMS:
    #     # for propms
    #     lower, upper = get_lambda_bounds_CV(m, Idx, cond_mean, Pi, 
    #                                         max_lambda_val=max_lambda_val,
    #                                         S=S, Beta=np.zeros((N,)))

    # else:
    lower, upper = get_lambda_bounds1(Idx=Idx, SamplingWeights=SW,
                                        cond_mean=cond_mean, Pi=Pi,
                                        use_propMS=use_propMS, 
                                        max_lambda_val=max_lambda_val, 
                                        max_f_S_ratio=max_f_S_ratio, 
                                        S=S, use_CV=use_CV, Beta=Beta)
    Lambda = np.clip(a=Lambda, a_min=lower, a_max=upper) 
    # step 4: get the wealth process 
    Wealth = np.cumprod(1 + Lambda*Bets) 
    # return the computed wealth 
    if return_bets:
        return Wealth, Lambda, lower, upper, Bets
    else:
        return Wealth, Lambda, lower, upper

def get_conf_seq(f, Pi, S, Idx, grid, SW, use_propMS=False, 
                max_lambda_val=2.0, betting_method='kelly', 
                max_f_S_ratio=1.0, threshold=40, intersect=True, 
                use_CV=False, beta_max=0.5):
    N, nG = len(f), len(grid)
    WW = np.zeros((nG, N))
    for i, m in enumerate(grid):
        # lower, upper == lower and upper bounds on lambda
        Wi, Lambda, lower, upper = get_wealth_process(m=m, f=f, Pi=Pi, S=S, Idx=Idx,
                                            SamplingWeights=SW,
                                            betting_method=betting_method, 
                                            use_propMS=use_propMS, 
                                            max_lambda_val=max_lambda_val, 
                                            max_f_S_ratio=max_f_S_ratio, 
                                            use_CV=use_CV, beta_max=beta_max)
        WW[i] = Wi
    # solve for the CS 
    LowerCS_, UpperCS_ = brute_force_CS_solver2(WW, grid, threshold=threshold) 
    ## do the predictive correction 
    if intersect:
        LowerCS_, UpperCS_ = predictive_correction1(LowerCS_, UpperCS_, Idx, Pi, f, 
                                                    intersect=True, logical=False) 
        LowerCS, UpperCS = predictive_correction1(LowerCS_, UpperCS_, Idx, Pi, f,
                                                    intersect=True, logical=True) 

    return LowerCS, UpperCS, Lambda, lower, upper


def main(N, M, f, Pi, S, sampling_methods, method_names=None,
            threshold=40, accuracy=0.1, save_fig=False, figname=None, 
            max_lambda_val=2.0, max_f_S_ratio=1.0, random_seed=None,
            plot_figs=True): 
    # set the random seed 
    if random_seed is not None:
        np.random.seed(random_seed)

    results = {}
    if method_names is None:
        method_names = np.arange(len(sampling_methods))
    m_star = np.sum(f*Pi)

    # plotting colors 
    # palette = sns.color_palette(palette='colorblind', n_colors=10)
    if plot_figs:
        palette = sns.color_palette(n_colors=10)
        plt.figure()
        # fig, axs = plt.subplots(1, 2)
        fig, axs = plt.subplots()
    for j, samp_method in enumerate(sampling_methods):
        SW = get_sampling_weights(M, S, method=samp_method)
        Idx = sample_indices(N, SamplingWeights=SW) 
        grid = np.linspace(0, 1, 101)
        nG = len(grid)
        use_propMS = (method_names[j]=='propMS') #TODO: remove 
        # ###====================================================
        # ###====================================================
        #### The next line does the same job as the commented part above. 
        LowerCS, UpperCS, Lambda, lower, upper = get_conf_seq(f=f, Pi=Pi, S=S, Idx=Idx, 
                grid=grid, SW=SW, use_propMS=use_propMS, max_lambda_val=max_lambda_val, 
                betting_method='kelly', max_f_S_ratio=max_f_S_ratio, threshold=threshold, 
                intersect=True)

        # find the first time the width of the CS is within the required 
        # multiplicative accuracy 
        Width = UpperCS - LowerCS 
        stoppingtime = 1 + np.argmax(Width<accuracy)
        if stoppingtime==1: # never stopped 
            stoppingtime = N
        # store the results for method_names[j]
        result = (LowerCS, UpperCS, stoppingtime, Lambda, lower, upper, Idx) 
        results[method_names[j]] = result
        # plot the results for this method 
        NN = np.arange(1, N+1)
        if plot_figs:
            color = palette[j]
            # axs.plot(NN, LowerCS, label=method_names[j] + ' + logical', color=color)
            axs.plot(NN, LowerCS, label=method_names[j], color=color)
            axs.plot(NN, UpperCS,  color=color)

            # ==UNCOMMENT THIS TO PLOT THE PROBABILISTIC CS AS WELL==
            # axs.plot(NN, LowerCS_, '--',  color=color, label=method_names[j])
            # axs.plot(NN, UpperCS_, '--',  color=color)

        # axs[1].plot(NN, UpperCS-LowerCS, label=method_names[j], color=color)
    if plot_figs:
        axs.plot(NN, m_star*np.ones(NN.shape), label='$m^*$', color=palette[j+1])
        axs.legend()
        axs.set_title('Confidence Sequence for $m^*$')
    # axs[1].set_title('CS-width')
    # axs[1].legend()
    if save_fig and plot_figs: 
        if figname is None:
            figname ='../data/temp.png'
        plt.savefig(figname, dpi=450)
    return results 


if __name__ == '__main__':
    # np.random.seed(2)

    N = 100
    NN = np.arange(1, N+1)
    # M-uniform f-bimodal 
    a = 0.9
    max_f_S_ratio = 1/a
    n = int(0.2*N)
    N_vals = [N-n, n]
    M_ranges = [ [1e1, 1e3], [1e5, 2*1e5]] 
    f_ranges = [[0.4, 0.5], [1e-7, 2*1e-7]] 

   
        # max_f_S_ratio = 1
    # S = np.random.random(f.shape)

    get_stopping_times = True
    num_trials=100
    save_fig=False
    basefigname = f'/PropMS_vs_PropM_betting_a_{a}_N_{N}'.replace('.','_')
    basefigname = '../data' + basefigname 
    

    
    if False:
        SW1 = get_sampling_weights(M, S, method='propM')
        Idx1 = sample_indices(N, SamplingWeights=SW1) 
        SW2 = get_sampling_weights(M, S, method='propMS')
        Idx2 = sample_indices(N, SamplingWeights=SW2) 

        m = m_star + 0.05 
        W1, Lambda1, lower1, upper1 = get_wealth_process(m=m, f=f, Pi=Pi, S=S, Idx=Idx1,
                                            SamplingWeights=SW1,
                                            betting_method='kelly', 
                                            use_propMS=False, 
                                            max_f_S_ratio=max_f_S_ratio)
        W2, Lambda2, lower2, upper2 = get_wealth_process(m=m, f=f, Pi=Pi, S=S, Idx=Idx2,
                                            SamplingWeights=SW2,
                                            betting_method='kelly', 
                                            use_propMS=True, 
                                            max_f_S_ratio=max_f_S_ratio)
        NN = np.arange(1, N+1)
        plt.plot(NN, W1, label='propM' )
        plt.plot(NN, W2, label='propMS' )
        plt.plot(NN, 40*np.ones((N,)), '--', alpha=0.5)
        plt.legend()
        plt.yscale('log')
    if True:

        # define the different sampling methods 
        # a sampling method must take M and S as inputs 
        # and return the (unnormalized) weights assigned 
        # to each transaction 
        uniform_method = lambda M, S: np.ones(M.shape)
        propM_method = lambda M, S: M 
        propMS_method = lambda M, S: M*S 
        propM2S_method = lambda M, S: M*M*S 
        propMinv_method = lambda M, S: M**(-1) 
        # Sampling_methods = (propM_method, propMS_method, propM2S_method, propMinv_method)
        # Method_names = ('propM', 'propMS', 'propM^2S', 'propM^{-1}')

        Sampling_methods = (propM_method, propMS_method, uniform_method)
        Method_names = ('propM', 'propMS', 'uniform')


        if get_stopping_times:
            StoppingTimesMS = np.zeros((num_trials,))
            StoppingTimesM = np.zeros((num_trials,))
            StoppingTimesU = np.zeros((num_trials,))

            for trial in tqdm(range(num_trials)):
                M, f, S = generate_MFS(N_vals=N_vals, N=N,
                                M_ranges = M_ranges, 
                                f_ranges = f_ranges, 
                                a = a)
                Pi = M/M.sum()
                m_star = np.sum(f*Pi) 

                results = main(N=N, M=M, f=f, Pi=Pi, S=S, sampling_methods=Sampling_methods, 
                                method_names=Method_names, max_f_S_ratio=max_f_S_ratio, 
                                accuracy=0.05, save_fig=False, figname=None, plot_figs=False)
                
                StoppingTimesMS[trial], StoppingTimesM[trial] = results['propMS'][2], results['propM'][2]
                StoppingTimesU[trial]= results['uniform'][2]

            plt.figure()
            plt.hist(x=StoppingTimesM, label='propM', alpha=0.75)
            plt.hist(x=StoppingTimesMS, label='propMS', alpha=0.75)
            plt.hist(x=StoppingTimesU, label='unioform', alpha=0.75)
            plt.xlabel('Stopping Time') 
            plt.ylabel('Number of trials')
            plt.title(f'Distribution of Stopping Times (N={N}, a={a}, {num_trials} trials)')
            plt.legend()
            if save_fig:
                figname4 = basefigname + '_StoppingTimes_.png'
                plt.savefig(figname4, dpi=450)
        else:
            M, f, S = generate_MFS(N_vals=N_vals, N=N,
                            M_ranges = M_ranges, 
                            f_ranges = f_ranges, 
                            a = a)
            Pi = M/M.sum()
            m_star = np.sum(f*Pi) 

            # plot the results 
            plt.figure()
            plt.bar(NN, M, label='M', width=0.8, alpha=0.7)
            plt.bar(NN, M*f, label='M*f', width=0.8)
            plt.bar(NN, M*S, label='M*S', alpha=0.8)
            plt.xlabel('Transaction Index')
            plt.ylabel('Value')
            plt.title('Monetary Values, True Misstated Amounts and \n Predicted Misstated Amounts')
            plt.legend()
            plt.yscale('log')
            plt.ylim(bottom=0, top=max(M)*1.3)

            if save_fig:
                figname0 = basefigname + '_data_.png'
                plt.savefig(figname0, dpi=450)




            figname1 = basefigname + '_CS_.png'        
            results = main(N=N, M=M, f=f, Pi=Pi, S=S, sampling_methods=Sampling_methods, 
                                method_names=Method_names, max_f_S_ratio=max_f_S_ratio, 
                                accuracy=0.05, save_fig=save_fig, figname=figname1, plot_figs=True)


            ##########################################################
            ## Uncomment from here 
            IdxM = results['propM'][-1] 
            IdxMS = results['propMS'][-1]

            plt.figure()
            fig, (ax0, ax1) = plt.subplots(2,1)
            ax0.bar(NN, M[IdxM], label='M', alpha=0.7)
            ax0.bar(NN, M[IdxM]*f[IdxM], label='M*f', alpha=0.8)
            ax0.set_title('Sequence of queries by prop-M strategy')
            ax0.axes.get_xaxis().set_visible(False)
            ax0.legend()

            ax1.bar(NN, M[IdxMS], label='M', alpha=0.7)
            ax1.bar(NN, M[IdxMS]*f[IdxMS], label='M*f', alpha=0.8)
            ax1.set_title('Sequence of queries by prop-MS strategy')
            ax1.legend()

            if save_fig:
                figname2 = basefigname + '_queries_.png'
                plt.savefig(figname2, dpi=450)

            
            ## Uncomment up to here
            ##########################################################
