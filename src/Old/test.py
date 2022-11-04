import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from Expt2_CV import first_threshold_crossing 

from weightedCSsequential import run_one_expt
from utils import generate_MFS

from weightedCS import sample_indices

def tryCV(eps=0.5, N1=900, N2=100, method1='propM', method2='propM'): 
    A = 0.1
    N = N1 + N2

    # M_ranges = [[1e4, 2e4], [1e5, 2*1e6]] 
    M_ranges = [[1e4, 2e4], [1e5, 2*1e6]] 
    f_ranges = [[0.4, 0.5], [1e-2, 2*1e-2]] 

    M, f, S =  generate_MFS(N_vals=(N1, N2), 
                N=N, # total number of transactions = sum(N_vals)
                M_ranges = M_ranges,  
                f_ranges = f_ranges, 
                a = A)
    # eps = 0.9
    S = (1-eps+ 2*eps*np.random.random(f.shape))*f 
    Pi = M / M.sum()
    # get the indices 
    Idx = sample_indices(N=N, SamplingWeights=M) 
    
    # compute the control-variates 
    Pi_, S_, M_ = Pi[Idx], S[Idx], M[Idx]
    S_mean = np.flip( np.cumsum( np.flip(Pi_ * S_)  )  )
    Pi_sum = np.flip( np.cumsum( np.flip(Pi_)))
    S_mean /=  Pi_sum 
    CV_vals = S_ - S_mean 
    # get the original payoffs 
    Mu0 = np.cumsum(Pi_ * f[Idx])
    m_star = np.sum(Pi*f) 
    m = m_star + 0.005
    # print(f'm^* = {m_star:.2f}, \t m={m}')
    Z = f[Idx] * Pi_sum - (m - Mu0)
    # get the beta value 
    beta_star = np.sum(Z * CV_vals) / np.sum(CV_vals*CV_vals + 1e-20) 
    beta_vals = np.zeros(f.shape) 
    beta_vals[1:] = np.cumsum(Z[:-1]*CV_vals[:-1]) / (np.cumsum(CV_vals[:-1]**2) + 1e-20)
    beta_vals = np.clip(beta_vals, -2, 2)
    # print(f'The best CV-weight is {beta_star:.4f}')
    # compute the wealth with a fixed bet value 
    # Zcv = Z - beta_star * CV_vals 
    Zcv = Z - beta_vals * CV_vals 
    lambda_max = 2
    lambd = max(-lambda_max, min(np.sum(Z)/np.sum(Z**2), lambda_max))
    lambd_cv = max(-lambda_max, min(np.sum(Zcv)/np.sum(Zcv**2), lambda_max))
    # print(f'no-cv Lambda: {lambd:.2f}, \t cv Lambda: {lambd_cv:.2f}')
    W = np.cumprod(1 + lambd*Z) 
    Wcv = np.cumprod(1 + lambd_cv*Zcv)

    # NN = np.arange(1, len(W)+1)
    # plt.plot(NN, W, label='no-cv')
    # plt.plot(NN, Wcv, label='cv')
    # plt.legend()
    # plt.yscale('log')
    # plt.show()
    return W, Wcv

def main(plot_fig=True):
    A = 0.1
    N1, N2 = 900, 100
    N = N1 + N2
    M_ranges = [[1e4, 2e4], [1e5, 2*1e6]] 
    f_ranges = [[0.4, 0.5], [1e-2, 2*1e-2]] 

    M, f, S =  generate_MFS(N_vals=(N1, N2), 
                N=N, # total number of transactions = sum(N_vals)
                M_ranges = M_ranges,  
                f_ranges = f_ranges, 
                a = A)



    # M_ranges = [[1e4, 2e4], [1e5, 2*1e6]] 
    # f_ranges = [[0.4, 0.5], [1e-2, 2*1e-2]] 

    # M, f, S =  generate_MFS(N_vals=(N1, N2), 
    #             N=N, # total number of transactions = sum(N_vals)
    #             M_ranges = M_ranges,  
    #             f_ranges = f_ranges, 
    #             a = A)

    nG = 100

    eps = 0.5
    S = (1-eps+ 2*eps*np.random.random(f.shape))*f 
    lambda_max = 2.0
    beta_max = 0.5

    grid, WealthCV, LowerCS1, UpperCS1, seen1 = run_one_expt(M, f, S, nG=nG,
                                                    method_name='propM', 
                                                    lambda_max=lambda_max, 
                                                    use_CV=True, beta_max=beta_max)
    
    _, Wealth, LowerCS2, UpperCS2, seen2 = run_one_expt(M, f, S, nG=nG,
                                                    method_name='propM', 
                                                    lambda_max=lambda_max, 
                                                    use_CV=False)


    m_star = np.sum(M*f)/np.sum(M)
    m_idx = int( (m_star + 0.07)*len(grid)) -1 
    
    Wcv, W = WealthCV[m_idx], Wealth[m_idx]
    NN = np.arange(1, len(Wcv)+1)
    if plot_fig: 
        plt.plot(NN, Wcv, label='cv')
        plt.plot(NN, W, label='no-cv')
        plt.plot(NN, np.ones(NN.shape)*20, 'k--')
        plt.yscale('log')
        plt.legend()
        plt.show()
    
    Tcv = first_threshold_crossing(Wcv, 20, max_time=N+1, upcross=True)
    T = first_threshold_crossing(W, 20, max_time=N+1, upcross=True)
    
    return Tcv, T


if __name__=='__main__':
    num_trials = 30
    plot_hist = False
    plot_wealth = False
    if plot_hist: 
        TTcv = np.zeros((num_trials, ))
        TT = np.zeros((num_trials, ))

        for trial in tqdm(range(num_trials)): 
            TTcv[trial], TT[trial] = main(plot_fig=False)
        

        plt.figure() 
        plt.hist(TTcv, label='cv', alpha=0.6)
        plt.hist(TT, label='no-cv', alpha=0.6)
        plt.legend() 
        plt.show()

        print(f'CV: {TTcv.mean()}, \t no-CV: {TT.mean()}')
    elif plot_wealth: 
        eps = 0.5
        N1, N2 = 900, 100
        N, num_trials = N1+N2, 50
        NN = np.arange(1, N+1)
        palette = sns.color_palette(n_colors=10)
        plt.figure()
        ST, STcv = np.zeros((num_trials,)), np.zeros((num_trials,))
        W, Wcv = np.zeros((N,)), np.zeros((N,))
        for trial in tqdm(range(num_trials)):
            W_, Wcv_ = tryCV(eps=eps, N1=N1, N2=N2)
            W += W_
            Wcv += Wcv_
            # plt.plot(NN, W, label='no-cv', color=palette[0], alpha=0.3)
            # plt.plot(NN, Wcv, label='cv', color=palette[1], alpha=0.3)
        W /= num_trials 
        Wcv /= num_trials  
        plt.plot(NN, W, label='no-cv', color=palette[0], alpha=0.8)
        plt.plot(NN, Wcv, label='cv', color=palette[1], alpha=0.8)
        plt.legend()
        plt.yscale('log')
    else: 

        eps = 0.5
        N1, N2 = 900, 100
        N, num_trials = N1+N2, 100
        NN = np.arange(1, N+1)
        palette = sns.color_palette(n_colors=10)
        alpha = 0.02
        # plt.figure()
        ST, STcv = np.zeros((num_trials,)), np.zeros((num_trials,))
        for trial in tqdm(range(num_trials)):
            W,  Wcv = tryCV(eps=eps, N1=N1, N2=N2)
            ST[trial] = first_threshold_crossing(W, 1/alpha, max_time=N+1)
            STcv[trial] = first_threshold_crossing(Wcv, 1/alpha, max_time=N+1)

        plt.hist(ST, label='no-cv', color=palette[0], alpha=0.5)
        plt.axvline(x=ST.mean(), color='k', linestyle='--')
        plt.hist(STcv, label='cv', color=palette[1], alpha=0.5)
        plt.axvline(x=STcv.mean(), color='r', linestyle='--')
        plt.xlim([0, N+1])
        plt.title(f'Epsilon={eps}')
        plt.show()
    
    print(f'Reduction in sample-requirement: {STcv.mean()/ST.mean():.2f}')