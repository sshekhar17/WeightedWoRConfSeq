import numpy as np


def hoeffding_boundaries(
        Z_t: np.ndarray,
        pi: np.ndarray,
        f: np.ndarray,
        ubs: np.ndarray,
        lbs: np.ndarray,
        alpha: float,
        t0: optional[int] = None,
        lambda_strategy: str = 'approx_uniform'
) -> Tuple[np.ndarray, np.ndarray]:
    """Boundaries of a Hoeffding CS.

    Arguments
        Z_t:
            (Z_t) sequence of IPW pi(I_t) * f(I_t) pseudo outcomes drawn without replacement
        pi:
            weights of each transactions i.e. pi(i)
        f:
            true misstatement fraction of each transaction
        ubs, lbs:
            upper and lower bounds on the support of Z_t based on q_t and pi
        alpha:
            error rate i.e. output a 1 - alpha CS
        t0:
            time which we desire the CS to be tight at. otherwise, if None, the CS is intended to be reasonably tight at all times. This changes lambda values.
        lambda_strategy:
            The type of lambda sequence to use for the CS
    Returns
        Lower and upper bounds for a Hoeffding based 1-a CS
    """

    N = Z_t.shape[0]
    t = np.arange(1, N + 1)

    c_t = ubs - lbs
    ct_sq = np.square(c_t)
    avg_c_sq_t = np.cumsum(ct_sq) / t
    inv_sum_t = np.cumsum(1 / ct_sq)

    if lambda_strategy is None:
        lambda_strategy = 'approx_uniform'
    if lambda_strategy == 'approx_uniform':
        lambda_den_t = t * np.log(t + 1) * ct_sq
    elif lambda_strategy == 'approx_fixed':
        lambda_den_t = t0 * avg_c_sq_t
    elif lambda_strategy == 'opt_uniform':
        lambda_den_t = ct_sq * inv_sum_t * np.log(t + 1)
    elif lambda_strategy == 'opt_fixed':
        lambda_den_t = ct_sq * inv_sum_t * (t0 / t)

    lambda_num = 8 * np.log(2 / alpha)
    lambda_t = np.minimum(np.sqrt(lambda_num / lambda_den_t), 1 / (2 * c_t))
    lambda_sum_t = np.cumsum(lambda_t)
    margin_t = (np.log(2 / alpha) +
                np.cumsum(np.square(lambda_t) * ct_sq / 8)) / lambda_sum_t

    m_hat_t = Z_t + np.append(0, np.cumsum(f * pi)[:-1])
    mu_hat_t = np.cumsum(lambda_t * m_hat_t) / lambda_sum_t
    diagnostics = (c_t, lambda_t, margin_t, m_hat_t, mu_hat_t,
                   np.cumsum(m_hat_t) / t)
    return mu_hat_t - margin_t, mu_hat_t + margin_t, diagnostics
