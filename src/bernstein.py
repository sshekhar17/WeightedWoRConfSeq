from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def phi(lambda_seq: np.ndarray, c_seq: np.ndarray) -> np.ndarray:
    """Phi function for empirical Bernstein martingale bound.

    See Lemma 4.1 of "Exponential inequalities for martingales with applications"
    Fan, Grama, Liu EJP 2015

    :param lambda_seq: array of lambdas
    :param c_seq: array of choices of c - should be same dimensions at `lambda_seq`
    :return: phi values for choices of lambda and c
    """
    assert np.all(c_seq > 0), c_seq
    return ((-1 * np.log(1 - c_seq * lambda_seq)) -
            c_seq * lambda_seq) / np.square(c_seq)
    #return -(np.log(1 - lambda_seq) + lambda_seq)


def eb_boundary(Zs: np.ndarray,
                Pi_t: np.ndarray,
                f_t: np.ndarray,
                lbs: np.ndarray,
                alpha: float,
                t0: Optional[int] = None,
                ests: Optional[np.ndarray] = None,
                lambda_strategy: Optional[str] = 'opt_uniform') -> np.ndarray:
    """Outputs an upper empirical Bernstein CS.

    :param xs:              samples from a distribution with mean bounded in [0, 1] (shape (N,)).
    :param lbs:             series of lower bounds on the support of the sample at each time step (shape (N,)).
    :param alpha:           error level of confidence sequence.
    :param t0:              time step to optimize tightness of CS at (if not optimizing uniformly in `lambda_strategy`)
    :param ests:            predictable estimates of next sample used to derive the empirical variance process (i.e. time process) (shape (N,))
    :param lambda_strategy: strategy for constructing lambda_t

    :return: empirical Bernstein upper CS on the mean (over running average of means) of `xs` (shape (N,))
    """
    N = Zs.shape[0]
    t = np.arange(1, N + 1)
    # if ests is None:
    #     ests = np.append(1, (np.cumsum(xs) / t)[:-1])
    ests = np.clip(ests, 0, 1)
    c_t = np.abs(lbs - ests)
    var_t = np.square(Zs - ests)

    c_sq_t = np.square(c_t)
    avg_var_t = np.append(1, (np.cumsum(var_t) / t)[:-1])
    avg_c_var_t = np.append(1, (np.cumsum(var_t * c_sq_t) / t)[:-1])
    #pred_avg_c_var_t = (avg_c_var_t * (t - 1) / t) + (c_sq_t * avg_var_t / t)
    pred_avg_c_var_t = (c_sq_t * avg_var_t / t)
    pred_inv_sum_t = np.append(0,
                               np.cumsum(1 / (var_t))[:-1]) + (1 / (avg_var_t))

    if lambda_strategy is None:
        lambda_strategy = 'approx_uniform'
    if lambda_strategy == 'approx_uniform':
        lambda_den_t = t * np.log(t + 1) * avg_var_t
    elif lambda_strategy == 'approx_fixed':
        lambda_den_t = t0 * pred_avg_c_var_t
    elif lambda_strategy == 'opt_uniform':
        lambda_den_t = avg_var_t * pred_inv_sum_t
    elif lambda_strategy == 'opt_fixed':
        lambda_den_t = c_sq_t * avg_var_t * pred_inv_sum_t * (t0 / t)
    elif lambda_strategy == 'oracle':
        lambda_den_t = var_t * np.sum(1 / var_t)

    lambda_num = 8 * np.log(1 / alpha)
    lambda_t = np.sqrt(lambda_num / lambda_den_t)
    lambda_t = np.minimum(lambda_t, 1 / (2 * c_t))
    #print(f"ls: {lambda_strategy}", lambda_t)
    #print(f"ls_den: {lambda_strategy}", lambda_den_t)
    lambda_sum_t = np.cumsum(lambda_t)

    xs = Zs + (np.append(0, np.cumsum(Pi_t * f_t)[:-1]))
    mu_hat_t = np.cumsum(lambda_t * xs) / lambda_sum_t
    ind_margin = var_t * phi(lambda_t, c_t) / lambda_t

    margin_t = (np.log(1 / alpha) +
                np.cumsum(var_t * phi(lambda_t, c_t))) / lambda_sum_t

    diagnostics = (lambda_t, var_t, c_t, avg_var_t, avg_c_var_t,
                   pred_avg_c_var_t, pred_inv_sum_t, ind_margin, margin_t,
                   mu_hat_t, np.cumsum(xs) / t)
    return mu_hat_t - margin_t, diagnostics
