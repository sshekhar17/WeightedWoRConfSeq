from random import random, sample
import numpy as np 
import scipy.stats as stats 
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm 
plt.style.use('seaborn-white')
from utils import brute_force_CS_solver2, predictive_correction1
from weightedCS import generate_MFS, get_sampling_weights, sample_indices
from weightedCS import control_variate_weights, get_lambda1


def get_min_max_values_of_payoffs(m, Idx, Pi, S, f, sampling_method='propM', 
        f_S_ratio_min=0, f_S_ratio_max=10, use_CS=False, Beta=None, S_mean=None):
    """
    Obtain the range of values the payoffs can lie. This 
    will then be used to calculate allowed values the 
    bets can take.  
    """
    # some preprocessing 
    Idx = Idx.astype(int)
    S_, f_, Pi_ = S[Idx], f[Idx], Pi[Idx]
    N = len(Idx)
    Pi_f_sum = np.zeros((N,)) 
    Pi_f_sum[1:] = np.cumsum(Pi_[:-1] * f_[:-1])
    mu_m = m - Pi_f_sum
    Pi_sum = np.zeros((N,))
    Pi_sum[1:] = np.cumsum(Pi_[:-1])
    if sampling_method=='propM':
        max_ = 1 - Pi_sum - mu_m
        min_ = -mu_m
    elif sampling_method=='propMS':
        Pi_S_total = np.sum(Pi_*S_)
        Pi_S_sum = np.zeros((N,))
        Pi_S_sum[1:] = np.cumsum(Pi_[:-1]*S_[:-1])
        max_ = f_S_ratio_max*(Pi_S_total-Pi_S_sum) - mu_m
        min_ = f_S_ratio_min*(Pi_S_total-Pi_S_sum) - mu_m
        pass 
    elif sampling_method=='uniform':
        Pi_max = np.array([
            np.max(Pi_[t:]) for t in range(N)
        ])
        max_ = np.arange(N, 0, -1) * Pi_max - mu_m 
        min_ = -mu_m 
    else:
        raise Exception('Sampling method must be "propM",  \
                "propMS", or "uniform"')
    if use_CS:
        min_CV, max_CV = get_min_max_CV(N, Beta=Beta, S_mean=S_mean)
        max_ = max_ - min_CV 
        min_ = min_ - max_CV
    return min_, max_

def get_min_max_CV(N, Beta=None, S_mean=None):
    Beta = np.ones((N,))*0.5 if Beta is None else Beta 
    if S_mean is None: 
        A, B = Beta, -Beta 
    else:
        A, B = Beta*(1-S_mean), -Beta*S_mean 
    min_CV = np.minimum(A, B)
    max_CV = np.maximum(A, B)
    return min_CV, max_CV

def get_S_mean(Idx, S, SW):
    N = len(Idx)
    Idx = Idx.astype(int)
    S_, SW_ = S[Idx], SW[Idx]
    SW_sum = np.cumsum(SW_[::-1])[::-1]

    temp = S_ * SW_
    S_mean = (np.cumsum(temp[::-1])[::-1]) / SW_sum 
    return S_mean 

def get_payoff_values(m, Idx, S, Pi, f, method='propM'):
    # some preprocessing 
    Idx = Idx.astype(int)
    S_, f_, Pi_ = S[Idx], f[Idx], Pi[Idx]
    N = len(Idx)
    Pi_f_sum = np.zeros((N,)) 
    Pi_f_sum[1:] = np.cumsum(Pi_[:-1] * f_[:-1])
    mu_m = m - Pi_f_sum
    Pi_sum = np.zeros((N,))
    Pi_sum[1:] = np.cumsum(Pi_[:-1])
    Pi_S_sum = np.zeros((N,))
    Pi_S_sum[1:] = np.cumsum(Pi_[:-1] * S_[:-1])
    if method=='propM':
        payoff =  f_*(1- Pi_sum) - mu_m
    elif method=='propMS':
        Pi_S_total = np.sum(Pi_*S_)
        payoff = f_ *(Pi_S_total- Pi_S_sum) / S_  - mu_m 
    elif method=='uniform':
        payoff = np.arange(N, 0, -1) * f_ * Pi_ - mu_m 
    else:
        raise Exception('Sampling method must be "propM",  \
                "propMS", or "uniform"')
    return payoff

def get_lambda_bounds_from_range(max_val, min_val, max_lambda_val=2.0, tol=1e-10):
    N = len(max_val)
    assert N==len(min_val)
    lower, upper = np.zeros((N,)), np.zeros((N,))
    for t in range(N):
        m0, m1 = min_val[t], max_val[t]
        if m0>=0: 
            lower[t] = -1/(m1+tol)
            upper[t] = max_lambda_val 
        elif m1>0>m0: 
            lower[t] = -1/(m1+tol)
            upper[t] = 1/(abs(m0)+tol)
        else: #m1<=0
            lower[t] = -max_lambda_val 
            upper[t] = 1/(abs(m0)+tol)
    lower = np.clip(a=lower, a_min=-max_lambda_val, a_max=max_lambda_val)
    upper = np.clip(a=upper, a_min=-max_lambda_val, a_max=max_lambda_val)
    return lower, upper

def get_lambda_values():
    pass 

def get_CV_mean_vector():
    pass 

def get_wealth_new():
    pass 

def get_F_S_from_dataset():
    pass 



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



if __name__ == '__main__':
    # np.random.seed(2)

    N = 1000
    NN = np.arange(1, N+1)
    # M-uniform f-bimodal 
    a = 0.9
    max_f_S_ratio = 1/a
    n = int(0.1*N)
    N_vals = [N-n, n]
    M_ranges = [ [1e5, 1e6], [1e5, 2*1e5]] 
    f_ranges = [[0.4, 0.5], [1e-2, 1*1e-1]] 
    M, f, S = generate_MFS(N_vals=N_vals, N=N,
                    M_ranges = M_ranges, 
                    f_ranges = f_ranges, 
                    a = a)
    Pi = M/M.sum()
    m_star = np.sum(f*Pi) 
    m = m_star + 0
    method='uniform'
    use_CV = True 

    SW = get_sampling_weights(M=M, S=S, method=method)
    Idx = sample_indices(N, SamplingWeights=SW) 

    min_, max_ = get_min_max_values_of_payoffs(m=m, 
        Idx=Idx, Pi=Pi, S=S, f=f, sampling_method=method, 
        f_S_ratio_max=1/a, f_S_ratio_min=1/(2-a)) 

    payoff = get_payoff_values(m=m, Idx=Idx, S=S, Pi=Pi, 
        f=f, method=method)

    # Check teh mean of S 
    S_mean = get_S_mean(Idx=Idx, S=S, SW=SW)
    S_ = S[Idx.astype(int)] 
    Beta = control_variate_weights(Bets=payoff, cv_vals=S_-S_mean,beta_max=0.25)

    min_CV, max_CV = get_min_max_CV(N=N, Beta=Beta, S_mean=S_mean) 

    # get the wealth process without control variates 
    payoff = payoff 
    Lambda = get_lambda1(payoff)
    lower, upper = get_lambda_bounds_from_range(max_val=max_, min_val=min_)
    Lambda = np.clip(a=Lambda, a_min=lower, a_max=upper) 
    W = np.cumprod(1 + Lambda*payoff) 
    # get the wealth process with control variates 
    payoffCV = payoff - Beta*(S_-S_mean)
    LambdaCV = get_lambda1(payoffCV)
    lowerCV, upperCV = get_lambda_bounds_from_range(max_val=max_-min_CV,
                                                min_val=min_-max_CV)
    LambdaCV = np.clip(a=LambdaCV, a_min=lowerCV, a_max=upperCV) 
    WCV = np.cumprod(1 + LambdaCV*payoffCV) 


    plt.plot(NN, W, label='no-CV')
    plt.plot(NN, WCV, label='CV')
    if m!=m_star:
        plt.yscale('log')
    plt.legend()
    plt.show()



