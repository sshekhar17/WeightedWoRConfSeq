import numpy as np


def hoeffding_boundaries(xs,
                         ubs,
                         lbs,
                         alpha,
                         t0=None,
                         lambda_strategy='approx_uniform'):
    N = xs.shape[0]
    t = np.arange(1, N + 1)

    c_t = ubs - lbs
    lambda_num = 8 * np.log(2 / alpha)
    ct_sq = np.square(c_t)
    avg_c_sq_t = np.cumsum(ct_sq) / t
    inv_sum_t = np.cumsum(1 / ct_sq)

    if lambda_strategy is None:
        lambda_strategy = 'opt_uniform'
    if lambda_strategy == 'approx_uniform':
        lambda_den_t = t * np.log(t + 1) * ct_sq
    elif lambda_strategy == 'approx_fixed':
        lambda_den_t = t0 * avg_c_sq_t
    elif lambda_strategy == 'opt_uniform':
        lambda_den_t = ct_sq * inv_sum_t * np.log(t + 1)
    elif lambda_strategy == 'opt_fixed':
        lambda_den_t = ct_sq * inv_sum_t * (t0 / t)

    lambda_t = np.minimum(np.sqrt(lambda_num / lambda_den_t), 1 / (2 * c_t))
    lambda_sum_t = np.cumsum(lambda_t)
    margin_t = (np.log(2 / alpha) +
                np.cumsum(np.square(lambda_t) * ct_sq / 8)) / lambda_sum_t
    mu_hat_t = np.cumsum(lambda_t * xs) / lambda_sum_t
    return mu_hat_t - margin_t, mu_hat_t + margin_t


# def hoeffding_cs(idxs, Q, max_mq, alpha, f, M, S, t0=None):
#     N = M.shape[0]
#     t = np.arange(1, N + 1)
#
#     pi = M / (np.sum(M))
#
#     if S is None:
#         pif = (pi * f)[idxs]
#     else:
#         pif = (pi * (f - S))[idxs] +
#     Z_t = pif / Q
#
#     m_hat_t = Z_t + np.append(0, np.cumsum((pif)[:-1]))
#
#     # mu_hat_t = np.cumsum(m_hat_t) / t
#
#     # # c_t = np.maximum(max_mq, np.append(0, mu_hat_t[:-1]))
#     # c_t = max_mq + 1
#
#     # lambda_num = 2 * np.log(2 / alpha)
#
#     # if t0 is None:
#     #     lambda_den = 0.25 * t * np.log(t + 1) * np.square(c_t)
#     # else:
#     #     lambda_den = 0.25 * np.full((N, ), t0) * np.square(c_t)
#     # lambdas = np.minimum(np.sqrt(lambda_num / lambda_den), 1 / (c_t))
#     # psi = np.square(lambdas) * np.square(c_t) / 8
#     # margin = (np.log(2 / alpha) + np.cumsum(psi)) / np.cumsum(lambdas)
#
#     # mu_hat_lambda_t = np.cumsum(lambdas * m_hat_t) / np.cumsum(lambdas)
#     # u = mu_hat_lambda_t + margin
#     # l = mu_hat_lambda_t - margin
#     return l, u
