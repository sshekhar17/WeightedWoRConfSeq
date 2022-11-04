from math import log
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-white')



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
                            1-a \leq f/S \leq 1 + a. 
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
    factor = (1-a) + 2*a*np.random.random((N,))
    S = factor * f 
    S[S>=1] = 1.0
    return M, f, S





def generate_random_problem_instance(N=100,
                                     M_min=100,
                                     M_max=1000,
                                     f_min=0.5,
                                     f_max=0.9,
                                     random_seed=None):
    """Generates a random problem instance.

    Arguments
        N : int
            number of transactions
        M_min : float
            minimum value of the reported transactions
        M_max : float
            maximum value of reported transactions
        f_min : float  \in [0, 1]
            minimum fraction of the fraudulent transactions
        f_max : float  \in [0, 1]
            maximum fraction of fraudulent transactions

    Returns
        M_vals : np.ndarray (N, )
            generated transaction values
        f_vals : np.ndarray (N, )
            generated (oracle) fractions of fraudulent transaction values
    """
    assert (M_max > M_min) and (f_max > f_min)
    if random_seed is not None:
        np.random.seed(random_seed)
    M_vals = M_min + (M_max - M_min) * np.random.random((N, ))
    f_vals = f_min + (f_max - f_min) * np.random.random((N, ))

    return M_vals, f_vals

def get_decreasing_CS(LowerCS, UpperCS, Idx, M, f):
    N = len(M) 
    assert (len(f)==N) and (len(Idx)==N)

    Idx = Idx.astype(int)
    M_ = M[Idx]
    Pi_ = M_/M_.sum() 
    f_ = f[Idx]

    m_star_t = np.zeros((N,)) 
    m_star_t[1:] = np.cumsum( Pi_[:-1] * f_[:-1] )
    LowerCS_ = np.maximum(LowerCS-m_star_t, 0) 
    UpperCS_ = np.maximum(UpperCS-m_star_t, 0)

    return LowerCS_, UpperCS_


def generate_bimodal_problem_instance(N=100,
                                      M_low=(10, 100),
                                      M_high=(950, 1000),
                                      f_min=0.5,
                                      f_max=0.9,
                                      random_seed=None):
    """Generates a random problem instance.

    Arguments
        N : int
            number of transactions
        M_low : Tuple[float, float]
            range for low values of the reported transactions
        M_high : Tuple[float, float]
            range for high values of the reported transactions
        f_min : float  \in [0, 1]
            minimum fraction of the fraudulent transactions
        f_max : float  \in [0, 1]
            maximum fraction of fraudulent transactions

    Returns
        M_vals : np.ndarray (N, )
            generated transaction values
        f_vals : np.ndarray (N, )
            generated (oracle) fractions of fraudulent transaction values
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    small_vals = M_low[0] + (M_low[1] - M_low[0]) * np.random.random(
        (N // 2, ))
    big_vals = M_high[0] + (M_high[1] - M_high[0]) * np.random.random(
        (N - (N // 2), ))
    f_vals = f_min + (f_max - f_min) * np.random.random((N // 2, ))
    f_vals_zeros = np.full((N - (N // 2)), 0.00001)

    return np.concatenate([small_vals, big_vals],
                          axis=0), np.concatenate([f_vals, f_vals_zeros])


def generate_score_functions(f_vals, alpha=0.8, beta=1.2, random_seed=None):
    """Generates score functions under the bounded deviations assumption.

    Arguments
        f_vals  : np.ndarray (N,)
            oracle fractions of fraudulent transactions
        alpha   : float \in [0, 1]
            lower bound of the relative error of score
        beta    : float \in [0, 1]
            upper bound of the relative error of score

    Returns
        S_vals : np.ndarray (N, )
            score values satisfying
                 alpha \leq S_vals[i]/f_vals[i] \leq beta for all i \in [N]
    """
    N = len(f_vals)
    mult_factors = alpha + (beta - alpha) * np.random.random((N, ))
    S_vals = np.minimum(1, mult_factors * f_vals)
    return S_vals


def phi_c(lambd, c, verbose=False):
    assert (c > 0)
    if c * lambd >= 1:
        if verbose:
            print(f'Warning: c*lambda >= 1: \t setting lambda to 1/(c+1e-5)')
        lambd = 1 / (c + 1e-5)
    return -(log(1 - c * lambd) + c * lambd) / (c * c)


def brute_force_CS_solver2(WW, grid, threshold=40):
    """
    Finds the CS by linear search over a fixed grid 

    Arguments
        WW          :(nG, N) ndarray 
                        the wealth process calculated at nG grid points 
        grid        :(nG,) ndarray
                        grid of points between 0 and  1
        threshold   : float

    Returns
        L           :(N,) ndarray
                        Lower CS 
        U           :(N,) ndarray 
                        Upper CS 

    """
    N = WW.shape[1]
    L, U = np.zeros((N,)), np.zeros((N,))
    for t in range(N):
        vals = WW[:, t] 
        sols = grid[np.where(vals<threshold)]
        if sols.size==0:
            l, u = 0, 1
        else:
            l, u = min(sols), max(sols)
        L[t], U[t] = l, u
    return L, U


def predictive_correction1(LowerCS, UpperCS, Idx, Pi, f, intersect=True,
                            logical=True):
    Idx = Idx.astype(int)

    one_minus_pi_sum = 1- np.cumsum(Pi[Idx])
    lower_logical = np.cumsum(f[Idx]*Pi[Idx])
    upper_logical = lower_logical + one_minus_pi_sum

    if logical:
        LowerCS = np.maximum(LowerCS, lower_logical) 
        UpperCS = np.minimum(UpperCS, upper_logical)

    if intersect:
        lmax, umin = 0, 1
        for i, (li, ui) in enumerate(zip(LowerCS, UpperCS)):
            lmax, umin = max(li, lmax), min(ui, umin)
            LowerCS[i] = lmax
            UpperCS[i] = umin

    return LowerCS, UpperCS

def first_threshold_crossing(arr, th, max_time=100, upward=True):
    if upward:
        if np.any(arr>th):
            return np.argmax(arr>th)+1
        else:
            return max_time
    else:
        if np.any(arr<th):
            return np.argmax(arr<th)+1
        else:
            return max_time




