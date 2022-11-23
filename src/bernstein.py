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


def eb_boundary(xs: np.ndarray,
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
    N = xs.shape[0]
    t = np.arange(1, N + 1)
    if ests is None:
        ests = np.append(1, (np.cumsum(xs) / t)[:-1])
    ests = np.clip(ests, 0, 1)
    c_t = np.abs(lbs - ests)
    var_t = np.square(xs - ests)

    c_sq_t = np.square(c_t)
    avg_var_t = np.append(1, (np.cumsum(var_t) / t)[:-1])
    avg_c_var_t = np.append(1, (np.cumsum(var_t * c_sq_t) / t)[:-1])
    pred_avg_c_var_t = avg_c_var_t * (t - 1) / t + (c_sq_t * avg_var_t / t)
    pred_inv_sum_t = np.append(
        0,
        np.cumsum(1 / (var_t * c_sq_t))[:-1]) + avg_var_t * c_sq_t

    if lambda_strategy is None:
        lambda_strategy = 'approx_uniform'
    if lambda_strategy == 'approx_uniform':
        lambda_den_t = t * np.log(t + 1) * pred_avg_c_var_t
    elif lambda_strategy == 'approx_fixed':
        lambda_den_t = t0 * pred_avg_c_var_t
    elif lambda_strategy == 'opt_uniform':
        lambda_den_t = c_sq_t * avg_var_t * pred_inv_sum_t * np.log(
            pred_inv_sum_t + 1)
    elif lambda_strategy == 'opt_fixed':
        lambda_den_t = c_sq_t * avg_var_t * pred_inv_sum_t * (t0 / t)

    lambda_num = 8 * np.log(1 / alpha)
    lambda_t = np.minimum(np.sqrt(lambda_num / lambda_den_t), 1 / (1.1 * c_t))
    lambda_sum_t = np.cumsum(lambda_t)
    mu_hat_t = np.cumsum(lambda_t * xs) / lambda_sum_t

    margin_t = (np.log(1 / alpha) +
                np.cumsum(var_t * phi(lambda_t, c_t))) / lambda_sum_t
    return mu_hat_t - margin_t


def _eb_bound(idxs: np.ndarray,
              Q: np.ndarray,
              alpha: float,
              f: np.ndarray,
              M: np.ndarray,
              c: np.ndarray,
              s_mode: str,
              S: Optional[np.ndarray],
              use_S_var: bool,
              lambda_formula: str = 'recent_weight_uniform',
              t0: Optional[int] = None) -> np.ndarray:
    """Produces an empirical Bernstein lower bound for weighted sampling w/o
    replacement.

    Let N be the total number of items

    :param idxs:         order in which items are to be sampled (length N).
    :param Q:            sampling weights of each item (length N)
    :param alpha:        desired level of confidence (hyperparameter for lambdas)
    :param f:            true misstated fraction of each transaction (length N)
    :param M:            monetary value of each transaction (length N)
    :param sample_dists: turns either (M, f) or (M, S) into weights of length N
    :param c:            parameters for empirical Bernstein function (length N)
    :param s_mode:       what to input into the weight_fn -  "oracle" to use f, "score" to use S, and "cost" to use just M
    :param S:            scores for each item i.e. estimates of the item's misstated fraction (length N)
    :param use_S_var:    whether to use S to estimate the empirical variance

    :return: an upper CS on the mean of all the items
    """

    N = M.shape[0]
    t = np.arange(1, N + 1)

    pi = M / (np.sum(M))

    pif = (pi * f)[idxs]
    Z_t = pif / Q

    m_hat_t = Z_t + np.append(0, np.cumsum((pif)[:-1]))
    mu_hat_t = np.cumsum(m_hat_t) / t

    c_t = np.append(c, mu_hat_t[:-1])
    c_t[c_t == 0] = c
    # c_t = np.full(N, c)

    if use_S_var and S is not None:
        # if we have scores, and we can use them for being the prediction in the empirical variance term
        rev_sample_scores = (M * S)[idxs][::-1] / np.sum(M)
        ests = np.cumsum(rev_sample_scores)[::-1]
        v_t = np.square(Z_t - ests)
    else:  # use current mu estimate for empirical variance
        v_t = np.square(Z_t - np.append(0, mu_hat_t[:-1]))

    # if S is not None:
    #     # use score for sampling distribution
    #     rev_sample_scores = (M * S)[idxs][::-1] / np.sum(M)
    #     mstar_ests = np.cumsum(rev_sample_scores)[::-1] + np.cumsum(
    #         np.append(0, pif[:-1]))
    #     est_vars = []
    #     weights = (M * S)[idxs]
    #     realities = np.append(0, np.cumsum((M * f)[idxs] / np.sum(M))[:-1])
    #     for idx in range(N):
    #         rem_weights = weights[idx:] / np.sum(M)
    #         rem_q = rem_weights / np.sum(rem_weights)
    #         est_vars.append(
    #             np.sum(rem_q *
    #                    np.square(rem_weights / rem_q + realities[idx])))
    #     lambdas = mstar_ests / np.array(est_vars)

    # else:
    est_vars = np.append(1, np.cumsum(v_t))[:-1] / t

    t0 = N / 2 if t0 is None else t0
    if lambda_formula == 'recent_weight_fixed':
        var_term = est_vars * np.square(c_t) * t0
    elif lambda_formula == 'recent_weight_uniform':
        var_term = est_vars * np.square(c_t) * t * np.log(t + 1)
    elif lambda_formula == 'opt_weight_fixed':
        cur_weight_sq = np.square(c_t) * est_vars

        # if we pretend the invsum so far is representative of the population invsum,
        # we shoudl multiply the inv_sum by the proportion between the final optimal
        # point and the the time step we're at now.
        var_term = cur_weight_sq * np.cumsum(
            1 / cur_weight_sq) * (t0 / np.arange(t))
    elif lambda_formula == 'opt_weight_uniform':
        cur_weight_sq = np.square(c_t) * est_vars
        inv_sum = np.cumsum(1 / cur_weight_sq)
        var_term = cur_weight_sq * inv_sum * np.log(inv_sum + 1)

    lambdas = np.sqrt(8 * np.log(2 / alpha) / var_term)

    # Clip lambdas so they cannot exceed upper bound in in phi function
    lambdas = np.minimum(lambdas, 1 / (2 * c_t))

    assert np.all(lambdas > 0)

    psi = phi(lambdas, c_t)
    margin = (np.log(2 / alpha) + np.cumsum(v_t * psi)) / np.cumsum(lambdas)

    mu_hat_lambda_t = np.cumsum(lambdas * m_hat_t) / np.cumsum(lambdas)

    l = mu_hat_lambda_t - margin
    return l


def empirical_bernstein_cs(idxs: np.ndarray,
                           Q: np.ndarray,
                           alpha: float,
                           f: np.ndarray,
                           M: np.ndarray,
                           c=1,
                           s_mode: str = 'cost',
                           S=None,
                           converse_S=None,
                           use_S_var=False) -> Tuple[np.ndarray, np.ndarray]:
    """Produces an empirical Bernstein upper and lower bounds for weighted
    sampling w/o replacement.

    Let N be the total number of items

    :param idxs: order in which items are to be sampled (length N).
    :param Q: sampling weights of each item (length N)
    :param alpha: desired level of confidence (hyperparameter for lambdas)
    :param f: true misstated fraction of each transaction (length N)
    :param M: monetary value of each transaction (length N)
    :param c: parameters for empirical Bernstein function (length N)
    :param s_mode: what to input into the weight_fn -  "oracle" to use f,
                   "score" to use S, and None to use just M
    :param S: scores for each item, i.e. estimates of the item's misstated
              fraction, when calculating lower bound (length N)
    :param converse_S: scores for each item, i.e. estimates of the item's
                       misstated fraction, when calculating upper bound
                       (length N)
    :param use_S_var: whether to use S to estimate the empirical variance
    """
    l = _eb_bound(idxs, Q, alpha, f, M, c, s_mode, S, use_S_var)
    u = 1. - _eb_bound(
        idxs, Q, alpha, 1. - f, M, c, s_mode,
        converse_S if converse_S is not None else None,
        use_S_var)  # Note that 1 - is applied to the result and the fs.
    return l, u
