import numpy as np
import scipy

def poisson_pmf(x, lambda_0):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if (x.astype(int) != x).any():
        raise ValueError("Input `x` must be an int or an ndarray of int-like objects")
    return np.where(x >= 0, lambda_0 ** x * np.exp(-lambda_0) / np.clip(scipy.special.factorial(x), 1, float('inf')), 0)


def double_poisson_pmf(x0, x1, lambda_0, lambda_1, lambda_2, eps=1e-20):
    joint_poisson = poisson_pmf(x0, lambda_0) * poisson_pmf(x1, lambda_1)
    if lambda_2.any() != 0:
        joint_poisson *= np.exp(-lambda_2) 
        # this part is hard to vectorize
        intermediate_results = np.zeros_like(joint_poisson)
        if isinstance(x0, int):
            support = np.arange(min(x0, x1) + 1)
            comb_term = scipy.special.comb(x0, support) * scipy.special.comb(x1, support) * scipy.special.factorial(support)
            expratio_term = (lambda_2 / (lambda_0 * lambda_1)) ** support
            intermediate_results = np.dot(comb_term, expratio_term)
        else:
            for i in range(x0.shape[0]): # probably the bottleneck
                support = np.arange(min(x0[i], x1[i]) + 1)
                comb_term = scipy.special.comb(x0[i], support) * scipy.special.comb(x1[i], support) * scipy.special.factorial(support)
                expratio_term = (lambda_2[i] / (lambda_0[i] * lambda_1[i])) ** support
                intermediate_results[i] = np.dot(comb_term, expratio_term)
        joint_poisson *= intermediate_results

    return np.clip(joint_poisson, eps, 1.)