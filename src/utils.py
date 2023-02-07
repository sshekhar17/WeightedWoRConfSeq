from typing import List, Optional, Tuple

import argparse
from math import log
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-white')


def generate_MFS(
        N_vals: Tuple[int, int] = (100, 100),
        N: int = 200,  # total number of transactions = sum(N_vals)
        M_ranges: List[List[float]] = [[1e3, 1e4], [1e5, 2 * 1e5]],
        f_ranges: List[List[float]] = [[0.4, 0.5], [1e-3, 2 * 1e-3]],
        a: float = 0.1,
        seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndrarry, np.ndarray]:
    """Generate synthetic M, f, S values.

    Parameters
       N_vals   :       number of transactions in different components
       N        :       total number of transactions = sum(N_vals)
       M_ranges :       range of M-values in different components
       f_ranges :       range of f-values in different components
       a        :       relative error in generating S from f
                            1-a \leq f/S \leq 1 + a.
       seed     :       random seed used to generate data. `None` for no seed.
    Returns
        M, f, S :       (N, ) numpy arrays
    """
    N_ = 0
    for n in N_vals:
        N_ += n
    if N_ != N:
        print('N doesn"t match sum of N_vals!!')
        N = N_
    M = np.empty((N, ))
    f = np.empty((N, ))
    num_components = len(N_vals)
    assert num_components == len(M_ranges)
    assert num_components == len(f_ranges)
    n_ = 0
    if seed is not None:
        np.random.seed(seed)
    for i in range(num_components):
        n = N_vals[i]
        M_lower, M_upper = M_ranges[i][0], M_ranges[i][1]
        del_M = M_upper - M_lower
        f_lower, f_upper = f_ranges[i][0], f_ranges[i][1]
        del_f = f_upper - f_lower

        M[n_:n_ + n] = np.random.random((n, )) * del_M + M_lower
        f[n_:n_ + n] = np.random.random((n, )) * del_f + f_lower

        n_ += n

    # Generate more in correlation style
    factor = (1 - a) + 2 * a * np.random.random((N, ))
    S = factor * f
    S[S >= 1] = 1.0
    return M, f, S


def generate_random_problem_instance(
        N: int = 100,
        M_min: int = 100,
        M_max: int = 1000,
        f_min: float = 0.5,
        f_max: float = 0.9,
        random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
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


def get_decreasing_CS(LowerCS: np.ndarray, UpperCS: np.ndarray,
                      Idx: np.ndarray, M: np.ndarray,
                      f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Produces a CS on the remaining misstated fraction of money, rather than
    m^*, which is the total misstated fraction.

    Arguments
        LowerCS, UpperCS:
            lower and upper CSes for m^*
        Idx:
            The order in which the indices are queried (i.e., list of indices in query order)
        M:
            Transactions values
        f:
            Misstatement fractions
    Returns
        LowerCS_, UpperCS_:
            lower and upper CSes for the remaining fraction of m^*
    """
    N = len(M)
    assert (len(f) == N) and (len(Idx) == N)

    Idx = Idx.astype(int)
    M_ = M[Idx]
    Pi_ = M_ / M_.sum()
    f_ = f[Idx]

    m_star_t = np.zeros((N, ))
    m_star_t[1:] = np.cumsum(Pi_[:-1] * f_[:-1])
    LowerCS_ = np.maximum(LowerCS - m_star_t, 0)
    UpperCS_ = np.maximum(UpperCS - m_star_t, 0)

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
    """Finds the CS by linear search over a fixed grid.

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
    L, U = np.zeros((N, )), np.zeros((N, ))
    for t in range(N):
        vals = WW[:, t]
        sols = grid[np.where(vals < threshold)]
        if sols.size == 0:
            l, u = 0, 1
        else:
            l, u = min(sols), max(sols)
        L[t], U[t] = l, u
    return L, U


def predictive_correction1(
        LowerCS: np.ndarray,
        UpperCS: np.ndarray,
        Idx: np.ndarray,
        Pi: np.ndarray,
        f: np.ndarray,
        intersect: bool = True,
        logical: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Takes the running intersection and/or intersection with logical CS of
    existing CS Arguments LowerCS, UpperCS:

        LowerCS, UpperCS:
            Any valid lower and upper CS for m^*.
        Idx:
            The order in which the indices are queried (i.e., list of indices in query order)
        Pi:
            Weights induced transaction values
        f:
            Misstatement fractions
        intersect:
            if true, takes the running intersection of the input CS
        logical:
            if true, intersects the running CS with the logical CS
    Returns
        LowerCS, UpperCS:
            lower and upper CSes that are the result
    """
    Idx = Idx.astype(int)

    one_minus_pi_sum = 1 - np.cumsum(Pi[Idx])
    lower_logical = np.cumsum(f[Idx] * Pi[Idx])
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


def first_threshold_crossing(arr: np.ndarray,
                             th: float,
                             max_time: int = 100,
                             upward: bool = True):
    """Gets the first index/time an array value crosses a threshold
        Arguments
            arr:
                Input array of floats
            th:
                Input threshold
            max_time:
                Alternative value to return if the array never corsses the threshold
            upward:
                Direction of crossing - upward if true and downward if false
        Returns
            First time (indexed at 1) a value in the array crosses the threshold, and max_time otherwise"""
    if upward:
        if np.any(arr > th):
            return np.argmax(arr > th) + 1
        else:
            return max_time
    else:
        if np.any(arr < th):
            return np.argmax(arr < th) + 1
        else:
            return max_time


def get_arg_values():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        default='hist',
        help=
        "Either `hist` for stopping time histogram or `cs` for just a CS of a single trial or `gain`"
    )
    parser.add_argument(
        '--small_prop',
        type=float,
        default=0.2,
        help="Proportion of transactions that will have small values")
    parser.add_argument(
        '--f_method',
        type=str,
        default='prop',
        help=
        '`prop` means that the f values are proportion to pis. `inv` means that the f values are inverse of pis'
    )
    parser.add_argument('--method_suite',
                        type=str,
                        default='betting',
                        help="Suite of methods to compare in this experiment")
    parser.add_argument('--out_path',
                        type=str,
                        default='fig.tex',
                        help='Output path to save figure to')
    parser.add_argument(
        '--cs_metric',
        type=str,
        default='width',
        help='Choose whether to plot CS width or CS boundaries.')
    parser.add_argument('--legend_flag',
                        type=str,
                        default='legend',
                        help='Choose whether to plot a legend or not.')
    parser.add_argument(
        '--post_process',
        type=str,
        default='separate',
        help=
        'Choose whether to do `separate`, `logical`, or `none` post processing of CSes.'
    )
    parser.add_argument('--N',
                        type=int,
                        default=200,
                        help='Number of samples in each trial.')
    parser.add_argument('--seed', type=int, default=322, help='RNG seed')
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.05,
        help='CI width at which to stop sampling for hist comparison.')
    args = parser.parse_args()
    out_dir = os.path.dirname(args.out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.random.seed(args.seed)
    return args
