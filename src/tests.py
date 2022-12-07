import numpy as np

from hoeffding import hoeffding_boundaries
from bernstein import eb_boundary


def make_bounded_pop(size):
    return np.random.beta(10, 20, size=size)


def test_hoeffding(fig_path, trials, size, cs, lambda_strategy, alpha):
    np.random.seed(322)
    pop = make_bounded_pop(size)
    mu = np.mean(pop)
    np.random.seed(322)
    for i in range(trials):
        samples = np.random.permutation(pop)
        if cs == 'Hoef.':
            l, u = hoeffding_boundaries(samples, np.ones(samples.shape),
                             np.zeros(samples.shape), alpha, lambda_strategy)
        elif cs == 'Emp. Bern.':
            l, u =
