import numpy as np

from utils import double_poisson_pmf

def poisson(params, x, y, weights=None):
    """
        Returns the negative log-likelihood (under a univariate Poisson distribution) of 
        observations y given parameters `params` and `x`.

        This is included for completeness only. Intended to be equivalent to

        ```
        import statsmodels.api as sm
        sm.GLM(y, x, family=sm.families.Poisson()).fit()
        ```

    """
    loglambda_ = x.dot(params)
    if weights is None:
        return -(y * loglambda_ - np.exp(loglambda_)).sum() / len(x)
    else:
        """
            M-step for diagonal-inflated bivariate Poisson regression requires a reweighted Poisson regression,
            weighted by essentially an estimate P(draw) or P(y_0 == y_1)
        """
        assert len(weights) == len(y)
        return -(weights * (y * loglambda_ - np.exp(loglambda_))).sum() / len(x)

def double_poisson(params, x, y0, y1, learned_covariance=True):
    lambda_ = np.exp(x.dot(params))
    return -np.log(double_poisson_pmf(y0, y1, lambda_[:, 0], lambda_[:, 1], lambda_[:, 2])).sum() / len(x)

def inflated_double_poisson(params, x, y0, y1, p, inflated_dist, learned_covariance=True):
    lambda_ = np.exp(x.dot(params))
    poisson_probs = double_poisson_pmf(y0, y1, lambda_[:, 0], lambda_[:, 1], lambda_[:, 2]) * (1 - p)
    probs = np.where(y0 == y1, poisson_probs, poisson_probs + p * inflated_dist.pmf(y0))
    return -np.log(probs).sum() / len(x)