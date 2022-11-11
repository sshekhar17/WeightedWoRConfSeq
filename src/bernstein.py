from typing import Callable, Optional

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


def _eb_bound(idxs: np.ndarray, Q: np.ndarray, alpha: float, f: np.ndarray,
              M: np.ndarray, weight_fn: Callable[[np.ndarray, np.ndarray],
                                                 np.ndarray], c: np.ndarray,
              s_mode: Optional[str], S: Optional[np.ndarray], use_S_var: bool):
    """Produces an empirical Bernstein lower bound for weighted sampling w/o
    replacement.

    Let N be the total number of items

    :param idxs: order in which items are to be sampled (length N).
    :param Q: sampling weights of each item (length N)
    :param alpha: desired level of confidence (hyperparameter for lambdas)
    :param f: true misstated fraction of each transaction (length N)
    :param M: monetary value of each transaction (length N)
    :param weight_fn: turns either (M, f) or (M, S) into weights of length N
    :param c: parameters for empirical Bernstein function (length N)
    :param s_mode: what to input into the weight_fn -  "oracle" to use f, "score" to use S, and None to use just M
    :param S: scores for each item i.e. estimates of the item's misstated fraction (length N)
    :param use_S_var: whether to use S to estimate the empirical variance
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
        # if we have scores, and we are using them for empirical variance...
        rev_sample_scores = (M * S)[idxs][::-1] / np.sum(M)
        ests = np.cumsum(rev_sample_scores)[::-1]
        v_t = np.square(Z_t - ests)
    else:  # use current mu estimate for empirical variacne
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
    est_vars = []
    # use f (if benchmarking against an oracle algo) or weights based on S
    #
    s_mode_map = {"oracle": f, None: np.ones(M.shape), "score": S}
    weights = weight_fn(M, s_mode_map[s_mode])[idxs]
    realities = np.append(0, np.cumsum((M * f)[idxs] / np.sum(M))[:-1])
    for idx in range(1, N):
        if idx <= N // 2:
            cur_q = weights[:idx] / np.sum(weights[:idx])
            est_vars.append(
                np.sum(cur_q * np.square(pif[:idx] / cur_q + realities[idx])))
        else:
            rem_len = N - idx
            block_ct = idx // rem_len
            rem_N = block_ct * rem_len
            block_pif, block_q = pif[:rem_N].reshape(
                rem_len,
                -1).sum(axis=1), weights[:rem_N].reshape(rem_len,
                                                         -1).sum(axis=1)
            est_vars.append(
                np.sum(block_q *
                       (np.square(block_pif / block_q + realities[idx]))))

    lambdas = np.append(1, mu_hat_t[:-1] / np.array(est_vars))

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
                           weight_fn: Callable[[np.ndarray, np.ndarray],
                                               np.ndarray],
                           c=1,
                           s_mode=Optional[str],
                           S=None,
                           converse_S=None,
                           use_S_var=False):
    """Produces an empirical Bernstein upper and lower bounds for weighted
    sampling w/o replacement.

    Let N be the total number of items

    :param idxs: order in which items are to be sampled (length N).
    :param Q: sampling weights of each item (length N)
    :param alpha: desired level of confidence (hyperparameter for lambdas)
    :param f: true misstated fraction of each transaction (length N)
    :param M: monetary value of each transaction (length N)
    :param weight_fn: turns either (M, f) or (M, S) into weights of length N
    :param c: parameters for empirical Bernstein function (length N)
    :param s_mode: what to input into the weight_fn -  "oracle" to use f, "score" to use S, and None to use just M
    :param S: scores for each item, i.e. estimates of the item's misstated fraction, when calculating lower bound (length N)
    :param converse_S: scores for each item, i.e. estimates of the item's misstated fraction, when calculating upper bound (length N)
    :param use_S_var: whether to use S to estimate the empirical variance
    """
    l = _eb_bound(idxs, Q, alpha, f, M, weight_fn, c, s_mode, S, use_S_var)
    u = 1. - _eb_bound(
        idxs, Q, alpha, 1. - f, M, weight_fn, c, s_mode,
        converse_S if converse_S is not None else None,
        use_S_var)  # Note that 1 - is applied to the result and the fs.
    return l, u
