import numpy as np


def hoeffding_cs(idxs, Q, max_mq, alpha, f, M, t0=None):
    N = M.shape[0]
    t = np.arange(1, N + 1)

    pi = M / (np.sum(M))

    pif = (pi * f)[idxs]
    Z_t = pif / Q

    m_hat_t = Z_t + np.append(0, np.cumsum((pif)[:-1]))
    mu_hat_t = np.cumsum(m_hat_t) / t

    c_t = np.maximum(max_mq, np.append(0, mu_hat_t[:-1]))

    lambda_num = 2 * np.log(2 / alpha)

    if t0 is None:
        lambda_den = 0.25 * t * np.log(t + 1) * np.square(c_t)
    else:
        lambda_den = 0.25 * np.full((N, ), t0) * np.square(c_t)
    lambdas = np.minimum(np.sqrt(lambda_num / lambda_den), 1 / (c_t))
    psi = np.square(lambdas) * np.square(c_t) / 8
    margin = (np.log(2 / alpha) + np.cumsum(psi)) / np.cumsum(lambdas)

    mu_hat_lambda_t = np.cumsum(lambdas * m_hat_t) / np.cumsum(lambdas)

    u = mu_hat_lambda_t + margin
    l = mu_hat_lambda_t - margin
    return l, u
