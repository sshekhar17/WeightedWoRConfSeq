"""Sequential implementation of the weighted CS using sampling WoR."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils import generate_MFS, predictive_correction1
from hoeffding import hoeffding_boundaries
from bernstein import eb_boundary

### the overview
#for t=1,2,...
#1   get the next probability distribution
#2  compute the minimum and maximum values of the next bet
#3  compute the limits of the allowed lambda value
#4  compute the next lambda value
#5  draw the next index
#6  update the seen and unseen sets of indices


################################################################################
def get_idx(val, arr):
    """find the position of the term val in the array 'arr', return -1 if it
    doesn't exist."""
    for i, a in enumerate(arr):
        if a == val:
            return i
    return -1


def get_Confidence_Interval(wealth, grid, delta=0.05):
    """Computes the confidence interval from the cross-sectional value of the
    process W_t(m)

    Parameters:
        wealth  : (nG,) numpy array denoting \{W_t(m): m \in grid\}
        grid    : (nG,) numpy array, usually a uniform grid in [0,1]
        delta   : float, tolerance level in (0,1)
    """
    not_rejected = []
    error_flag = False
    for i, w in enumerate(wealth):
        if w < 1 / delta:
            not_rejected.append(grid[i])
    if len(not_rejected) == 0:
        print('All the grid values have been rejected!!!')
        error_flag = True
        return 0, 1, error_flag
    else:
        l, u = min(not_rejected), max(not_rejected)
        return l, u, error_flag


################################################################################


# get the next probability distribution
def get_sampling_distribution2(unseen,
                               M,
                               f,
                               S,
                               method_name='propMS',
                               samp_func=None,
                               samp_kwargs=None):
    """Returns a probability distribution on the unseen transactions.

    Parameters:
        unseen      : ndarray   unseen transaction indices
        M           : ndarray   array of reported monetary values
        f           : ndarray   array with the true misstated fractions
        S           : ndarray   side-information or score values
        method_name : str       name of the sampling method
                                valid choices are "propMS", "propM", "oracle"
                                and "uniform"
        samp_func   : callable  function handle of the sampling function
                                (if a method other than the above four are to be used)
        samp_kwargs : dict      keyword arguments for the samp_func function

    Returns:
        q           : ndarray   a probability distribution over the indices
                                represented by the array "unseen"
    """
    # check if the sampling function handle is provided
    if (samp_func is None):
        # extract the unseen parts of the data
        assert len(unseen) > 0
        unseen = unseen.astype(int)
        M, f, S = M[unseen], f[unseen], S[unseen]
        # set up the keyword arguments dict
        samp_kwargs = {'M_': M, 'f_': f, 'S_': S}
        # define the methods based on method_name
        if method_name == 'propMS':
            samp_func = lambda M_, f_, S_: M_ * S_ / np.sum(M_ * S_)
        elif method_name == 'propM':
            samp_func = lambda M_, f_, S_: M_ / np.sum(M_)
        elif method_name == 'oracle':
            samp_func = lambda M_, f_, S_: M_ * f_ / np.sum(M_ * f_)
        elif method_name == 'uniform':
            samp_func = lambda M_, f_, S_: np.ones(M_.shape) / len(M_)
        else:
            raise Exception(
                f'{method_name} is not a valid sampling-method name')
    # now call the sampling function
    assert callable(samp_func)
    q = samp_func(**samp_kwargs)
    # return the distribution
    return q


def get_range_of_next_payoff(m,
                             unseen,
                             q_t,
                             mu_0,
                             M,
                             S,
                             method_name='propMS',
                             f_over_S_range=None,
                             use_CV=False,
                             beta=0.5):
    """Compute the interval within which the next payoff value must lie.

    Parameters:
        m               :float  in (0,1)
        unseen          :ndarray  array of unseen transaction indices
        q_t             :ndarray  probability distribution over unseen
        mu_0            :float      the weighted mean computed so far
                            \sum_{i=1}^{t-1} \pi(I_i) \times f(I_i)
        M               :ndarray    monetary values of the transactions
        S               :ndarray    side information or score values
        method_name     :str        name of the sampling method
        f_over_S_range  :list       range of the relative accuracy of f/S
        use_CV          :bool       use control-variates if true
        beta            :float      weight of the control variate term

    Return
        lower, upper    :float  lower and upper bounds on the next
                                payoff values
    """
    unseen = unseen.astype(int)
    S_, M_ = S[unseen], M[unseen]
    mu_m = m - mu_0
    # get the minimum and maximum values of the next payoff
    if method_name == 'oracle':
        method_name = 'propMS'
        f_over_S_range = [1, 1]  # f is known exactly
    if method_name == 'propMS':
        if f_over_S_range is None:
            s_min = max(min(S_), 1e-15)
            f_over_S_range = [0, 1 / s_min]
        # get the range of values
        temp = np.sum(S_ * M_) / np.sum(M)
        lower = f_over_S_range[0] * temp - mu_m
        upper = f_over_S_range[1] * temp - mu_m
    elif method_name == 'propM':
        temp = np.sum(M_) / np.sum(M)
        lower = 0 * temp - mu_m
        upper = 1 * temp - mu_m
    else:  # direcly use worst case probability
        lower = 0 - mu_m  #f=0
        upper = max((M_ / np.sum(M)) / (q_t + 1e-15)) - mu_m  #f=1
    if use_CV:
        s_mean = np.sum(q_t * S_)
        if beta > 0:
            s_max = beta * (max(S_) - s_mean)
            s_min = beta * (min(S_) - s_mean)
        else:
            s_min = beta * (max(S_) - s_mean)
            s_max = beta * (min(S_) - s_mean)
        # update the lower and upper values of the payoff
        # NOTE: we use the convention that the cross-variate
        # term, \beta \times (S_{I_t} - s_mean) is SUBTRACTED
        # from the original payoff term
        lower = lower - s_max
        upper = upper - s_min
        # sanity check
        assert upper >= lower
    return lower, upper


def get_range_of_next_bet(min_payoff, max_payoff, lambda_max=2, tol=1e-10):
    """Return the range in which the bet must lie to ensure non-negativity of
    the wealth process.

    Parameters:
        min_payoff      :float lower bound on the next payoff term
        max_payoff      :float upper bound on the next payoff term
        lambda_max      :float  an upper bound on the maximum bet value

    Returns:
        lower, upper    :float end points of the allowed interval
                            in which the next bet~(i.e., lambda_t) must lie
    """
    m0, m1 = min_payoff, max_payoff
    assert m1 >= m0
    if m0 >= 0:
        lower = -1 / (m1 + tol)
        upper = lambda_max
    elif m1 > 0 > m0:
        lower = -1 / (m1 + tol)
        upper = 1 / (abs(m0) + tol)
    else:  #m1<=0
        lower = -lambda_max
        upper = 1 / (abs(m0) + tol)
    lower = max(lower, -lambda_max)
    upper = min(upper, lambda_max)
    # sanity check
    assert min(lower * min_payoff, lower * max_payoff, upper * min_payoff,
               upper * max_payoff) > -1
    return lower, upper


def get_next_CV_bet(cv_vals,
                    payoff_vals,
                    beta_max=0.5,
                    tol=1e-10,
                    smoothing_term=1.0):
    """Calculate the weight to be assigned to the control-variate term.

    Parameters:
        cv_vals     :ndarray    control-variate terms from prior rounds
        payoff_vals :ndarray    payoff terms (without the cv) from the prior rounds
        beta_max    :float      maximum value of the weight of cv-terms

    Return:
        beta        :float      the weight of the cv term for the next round

    Notes:
        we SUBTRACT the cv term form the original payoff; that is
        cv_payoff = payoff - beta * cv
    """

    if len(cv_vals) == 0:
        return 0
    assert len(cv_vals) == len(payoff_vals)
    C, B = cv_vals, payoff_vals
    numerator = (C * B).sum() + smoothing_term
    denominator = (C * C).sum() + tol + 2 * smoothing_term
    beta = (numerator / denominator)
    beta = min(beta_max, max(-beta_max, beta))
    return beta


def get_next_bet(values, betting_method='kelly', tol=1e-10, lambda_max=None):
    """Return the next bet.

    Parameters:
        values          :ndarray    previous payoff (or payoff - beta*cv) values
        betting_method  :str        only implemented approximate kelly betting
                                    #TODO: add ONS betting strategy
        lambda_max      :float      upper bound on the bet magnitude

    Returns:
        lambda_         :float      the next bet in [-lambda_max, lambda_max]
    """

    if len(values) == 0:
        return 0
    if betting_method == 'kelly':
        B_ = np.sum(values)
        B2_ = np.sum(values * values)
        lambda_ = B_ / (B2_ + tol)
    else:
        raise Exception('Only implemented Kelly betting')
    if lambda_max is not None:
        lambda_ = min(lambda_max, max(-lambda_max, lambda_))
    return lambda_


def one_step_update(grid,
                    wealth,
                    seen,
                    unseen,
                    M,
                    f,
                    S,
                    Payoff_vals,
                    cv_vals,
                    beta_vals,
                    method_name='propMS',
                    samp_func=None,
                    samp_kwargs=None,
                    use_CV=False,
                    beta_max=0.5,
                    f_over_S_range=None,
                    lambda_max=2.5,
                    alpha=0.05):
    unseen = unseen.astype(int)
    Pi = M / np.sum(M)
    Pi_ = Pi[unseen]
    f_, M_, S_ = f[unseen], M[unseen], S[unseen]
    mu_0 = 0
    if seen is not None:
        seen.astype(int)
        _Pi, _f = Pi[seen], f[seen]
        mu_0 = np.sum(_Pi * _f)
    # step 1: get the sampling probability
    q_t = get_sampling_distribution2(unseen,
                                     M,
                                     f,
                                     S,
                                     method_name=method_name,
                                     samp_func=samp_func,
                                     samp_kwargs=samp_kwargs)
    # draw the next transaction index
    It = np.random.choice(a=unseen, p=q_t)
    idx = get_idx(It, unseen)
    # compute the next control variate value
    if use_CV:
        cv = S_[idx] - np.sum(q_t * S_)

    current_payoffs = np.zeros(grid.shape)
    current_wealth = np.zeros(grid.shape)
    Lambda_vals = np.zeros(grid.shape)
    for i, m in enumerate(grid):
        payoff_vals = Payoff_vals[i]
        # step 2: get the beta value if CV is used
        if use_CV:
            beta = get_next_CV_bet(cv_vals, payoff_vals, beta_max=beta_max)
            values = payoff_vals - (beta_vals * cv_vals)
        else:
            beta = 0
            values = payoff_vals
        # step 3: get the range of the next observations
        min_payoff, max_payoff = get_range_of_next_payoff(
            m,
            unseen,
            q_t,
            mu_0,
            M,
            S,
            method_name=method_name,
            f_over_S_range=f_over_S_range,
            use_CV=use_CV,
            beta=beta)
        # step 4: get the range of allowed lambda values
        l_min, l_max = get_range_of_next_bet(min_payoff,
                                             max_payoff,
                                             lambda_max=lambda_max)
        assert l_min <= l_max  # sanity check
        # step 5: get the lambda value for this m
        lambda_ = get_next_bet(values, betting_method='kelly')
        lambda_ = max(l_min, min(l_max, lambda_))
        # step 6: get the next payoff value
        payoff = (Pi_[idx] * f_[idx]) / (q_t[idx] + 1e-15) - (m - mu_0)
        current_payoffs[i] = payoff
        val = payoff
        if use_CV:
            val -= beta * cv
        # step 7: update the wealth for this m
        if (lambda_ * val <= -1):
            print(f'Warning: lambda * payoff < -1 !!! for m={m}:')
            print(
                f'lambda = {lambda_:.2f}, and payoff = {val:.2f}, product = {lambda_*val:.2f}'
            )
            print(f'setting lambda to 0')
            print('\n')
            lambda_ = 0
        current_wealth[i] = wealth[i] * (1 + lambda_ * val)
        Lambda_vals[i] = lambda_

    # Update everything
    if seen is None:
        seen = np.array([It])
    else:
        seen = np.append(seen, It)
    unseen = np.delete(unseen, obj=idx)
    # add the current payoffs as a new column
    Payoff_vals = np.concatenate((Payoff_vals, current_payoffs.reshape(
        (-1, 1))),
                                 axis=1)
    L, U, error_flag = get_Confidence_Interval(current_wealth, grid, alpha)
    if use_CV:
        cv_vals = np.append(cv_vals, cv)
        beta_vals = np.append(beta_vals, beta)
        return seen, unseen, It, current_wealth, Payoff_vals, L, U, cv_vals, beta_vals, error_flag, Lambda_vals
    else:
        return seen, unseen, It, current_wealth, Payoff_vals, L, U, error_flag, Lambda_vals


def run_one_expt(M,
                 f,
                 S,
                 method_name='propMS',
                 cs='Bet',
                 lambda_max=2.5,
                 beta_max=0.5,
                 nG=100,
                 use_CV=False,
                 f_over_S_range=None,
                 alpha=0.05,
                 logical_CS=False,
                 intersect=False,
                 return_payoff=False,
                 lambda_strategy=None,
                 cv_max=np.inf):
    N = len(M)
    LowerCS, UpperCS = np.zeros((N, )), np.ones((N, ))
    grid = np.linspace(0, 1, nG)
    wealth = np.ones((nG, ))
    Wealth = np.ones((nG, 1))
    Payoff_vals = np.zeros((nG, 1))
    seen = None  # indicies queried so far
    unseen = np.arange(N)  # indicies not queried so far
    Transaction_Indices = np.zeros((N, ))
    if use_CV:
        cv_vals = np.zeros((1, ))
        beta_vals = np.zeros((1, ))
    else:
        cv_vals = None
        beta_vals = None
    Error_flag = False
    if cs == 'Bet':
        for t in range(N):
            result = one_step_update(grid=grid,
                                     wealth=wealth,
                                     seen=seen,
                                     unseen=unseen,
                                     M=M,
                                     f=f,
                                     S=S,
                                     Payoff_vals=Payoff_vals,
                                     cv_vals=cv_vals,
                                     beta_vals=beta_vals,
                                     method_name=method_name,
                                     use_CV=use_CV,
                                     beta_max=beta_max,
                                     lambda_max=lambda_max,
                                     f_over_S_range=f_over_S_range,
                                     alpha=alpha)
            if use_CV:
                seen, unseen, It, wealth, Payoff_vals, L, U, cv_vals, beta_vals, error_flag, lambda_ = result
            else:
                seen, unseen, It, wealth, Payoff_vals, L, U, error_flag, lambda_ = result
            Transaction_Indices[t] = It
            Wealth = np.concatenate((Wealth, wealth.reshape((-1, 1))), axis=1)
            if t >= 1:
                LowerCS[t], UpperCS[t] = L, U

            Error_flag = (Error_flag or error_flag)
    else:
        unseen = list(range(N))
        Pi = M / np.sum(M)
        I_t, Z_t, Z_om_t = [], [], []
        ubs, lbs, lb_oms = [], [], []
        for _ in range(N):
            q_t = get_sampling_distribution2(np.array(unseen),
                                             M,
                                             f,
                                             S,
                                             method_name=method_name)

            if use_CV:
                S_adj = np.minimum(S[unseen], cv_max / q_t)
                possible_Z = lambda f_: (f_ - S_adj) * Pi[
                    unseen] / q_t + np.sum(Pi[It] * f[It]) + np.sum(Pi[unseen]
                                                                    * S_adj)

                S_adj_om = np.minimum(1 - S[unseen], cv_max / q_t)
                possible_Z_om = lambda f_: (f_ - S_adj_om) * Pi[
                    unseen] / q_t + np.sum(Pi[It] * (1 - f[It])) + np.sum(Pi[
                        unseen] * S_adj_om)
            else:
                possible_Z = lambda f_: f_ * Pi[unseen] / q_t + np.sum(Pi[
                    I_t] * f[I_t])
                possible_Z_om = lambda f_: f_ * Pi[unseen] / q_t + np.sum(Pi[
                    I_t] * (1. - f[I_t]))
            ubs.append(np.max(possible_Z(1)))
            lbs.append(np.min(possible_Z(0)))
            lb_oms.append(np.min(possible_Z_om(0)))

            sample_idx = np.random.choice(np.arange(len(unseen)), p=q_t)
            Z_t.append(possible_Z(f[unseen])[sample_idx])
            Z_om_t.append(possible_Z(1. - f[unseen])[sample_idx])
            I_t.append(unseen[sample_idx])
            unseen.remove(unseen[sample_idx])

        Transaction_Indices = np.array(I_t)

        if cs == 'Hoef.':
            LowerCS, UpperCS = hoeffding_boundaries(
                xs=np.array(Z_t),
                lbs=np.array(lbs),
                ubs=np.array(ubs),
                alpha=alpha,
                lambda_strategy=lambda_strategy)
        else:  # cs == 'Emp. Bern.':
            LowerCS, UpperCS = (
                eb_boundary(np.array(Z_t),
                            np.array(lbs),
                            alpha / 2,
                            lambda_strategy=lambda_strategy), 1. -
                eb_boundary(np.array(Z_om_t), np.array(lb_oms), alpha / 2))
    if logical_CS or intersect:
        LowerCS, UpperCS = predictive_correction1(LowerCS,
                                                  UpperCS,
                                                  Idx=Transaction_Indices,
                                                  Pi=M / M.sum(),
                                                  f=f,
                                                  intersect=intersect,
                                                  logical=logical_CS)
    if return_payoff:
        return grid, Wealth, LowerCS, UpperCS, Transaction_Indices, Error_flag, Payoff_vals
    else:
        return grid, Wealth, LowerCS, UpperCS, Transaction_Indices, Error_flag


def main(A=0.1, use_CV=False):
    N = 200
    N1 = 150
    N2 = N - N1
    f_over_S_range = [1 - A, 1 + A]
    f_ranges = [[0.4, 0.5], [1e-3, 2 * 1e-3]]
    # M_ranges = [ [1e5, 1e6], [1e2, 1*1e3]],
    M_ranges = [[1e2, 2e2], [5e2, 8e2]]
    M, f, S = generate_MFS(
        N_vals=(N1, N2),
        N=N,  # total number of transactions = sum(N_vals)
        M_ranges=M_ranges,
        f_ranges=f_ranges,
        a=A)

    nG = 100
    lambda_max = 2
    result_propMS = run_one_expt(M,
                                 f,
                                 S,
                                 method_name='propMS',
                                 lambda_max=lambda_max,
                                 beta_max=0.5,
                                 nG=nG,
                                 use_CV=use_CV,
                                 f_over_S_range=f_over_S_range,
                                 alpha=0.05,
                                 logical_CS=False,
                                 intersect=False,
                                 return_payoff=False)
    grid, _, L1, U1, _, _ = result_propMS

    result_propM = run_one_expt(M,
                                f,
                                S,
                                method_name='propM',
                                lambda_max=lambda_max,
                                beta_max=0.5,
                                nG=nG,
                                use_CV=use_CV,
                                f_over_S_range=None,
                                alpha=0.05,
                                logical_CS=False,
                                intersect=False,
                                return_payoff=False)
    _, _, L2, U2, _, _ = result_propM

    result_unif = run_one_expt(M,
                               f,
                               S,
                               method_name='uniform',
                               lambda_max=lambda_max,
                               beta_max=0.5,
                               nG=nG,
                               use_CV=use_CV,
                               f_over_S_range=None,
                               alpha=0.05,
                               logical_CS=False,
                               intersect=False,
                               return_payoff=False)
    _, _, L3, U3, _, _ = result_unif

    NN = np.arange(1, N + 1)
    plt.plot(NN, U1 - L1, label='propMS')
    plt.plot(NN, U2 - L2, label='propM')
    plt.plot(NN, U3 - L3, label='uniform')
    plt.legend()


def testCV(A=0.1):
    N = 200
    N1 = 150
    N2 = N - N1
    f_over_S_range = [1 - A, 1 + A]
    f_ranges = [[0.4, 0.5], [1e-3, 2 * 1e-3]]
    # M_ranges = [ [1e5, 1e6], [1e2, 1*1e3]],
    M_ranges = [[1e2, 2e2], [5e2, 8e2]]
    M, f, S = generate_MFS(
        N_vals=(N1, N2),
        N=N,  # total number of transactions = sum(N_vals)
        M_ranges=M_ranges,
        f_ranges=f_ranges,
        a=A)

    nG = 100
    lambda_max = 2
    result_propMS = run_one_expt(M,
                                 f,
                                 S,
                                 method_name='propM',
                                 lambda_max=lambda_max,
                                 beta_max=0.5,
                                 nG=nG,
                                 use_CV=False,
                                 f_over_S_range=f_over_S_range,
                                 alpha=0.05,
                                 logical_CS=False,
                                 intersect=False,
                                 return_payoff=False)
    grid, _, L1, U1, _, _ = result_propMS

    result_propM = run_one_expt(M,
                                f,
                                S,
                                method_name='propM',
                                lambda_max=lambda_max,
                                beta_max=0.5,
                                nG=nG,
                                use_CV=True,
                                f_over_S_range=None,
                                alpha=0.05,
                                logical_CS=False,
                                intersect=False,
                                return_payoff=False)
    _, _, L2, U2, _, _ = result_propM

    NN = np.arange(1, N + 1)
    plt.plot(NN, U1 - L1, label='propMS-no-cv')
    plt.plot(NN, U2 - L2, label='propM-cv')
    plt.legend()


if __name__ == '__main__':
    testCV(0.9999)
