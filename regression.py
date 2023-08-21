import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm

from log_likelihoods import poisson, double_poisson, inflated_double_poisson
from utils import double_poisson_pmf

def poisson_regression(x, y, weights=None, init_guess=None, optim_method="L-BFGS-B", options={'gtol': 1e-3}):
    assert x.ndim == 2, "`x` must be two-dimensional."
    if init_guess is None:
        init_guess = np.ones(x.shape[-1])
    result = minimize(poisson, init_guess, args=(x, y, weights), method=optim_method, options=options)
    if not result.success:
        raise RuntimeError(f"Regression unsuccessful: {result.status}, ({result.message})")
    solution = result.x
    return solution

def double_poisson_regression(x, y0, y1, n_iterations=2000, init_guess=None, warm_mstep=True,
                              learned_covariance=True, diagonal_pmf=None, init_p=0.5, init_theta=None, verbose=20):
    """
        Uses the trivariate factorization of the double Poisson distribution used by Karlis 
        and Ntzoufras to derive an EM algorithm for the fitting of double Poisson parameters.
        The EM algorithm proceeds according to this general flow:

        E-step: given X, Y0, Y1, and current parameter guess, impute preliminary estimate of Z_2
        M-step: Given preliminary estimates of Z_2, X, Y0, Y1, update current parameter guess

        For the independent case it is sufficient to run separate Poisson regressions.

    """
    if init_guess is None:
        beta = np.random.randn(x.shape[1], 3)
    else:
        beta = init_guess
    
    if diagonal_pmf is not None:
        theta = init_theta
        p = init_p
    prev_nll = float('inf')
    for i in tqdm(range(n_iterations)):
        # E-step
        if diagonal_pmf is None:
            nll = double_poisson(beta, x, y0, y1)
        else:
            nll = inflated_double_poisson(beta, x, y0, y1, p, diagonal_pmf)
        if i % verbose == 0:
            print(f"EM it. {i}: nll - {nll}")
        if nll >= prev_nll and i > 1: # get rid of the "lucky guess" phenomenon
            print(f"Converged on it. {i}: nll - {nll}")
            break
        prev_nll = nll
        lambda_ = np.exp(x.dot(beta)) # x should have shape (N, d); params should have shape (d, 3), result is (N, 3). This is technically from the M-step 
        lambda_0, lambda_1, lambda_2 = lambda_[:, 0], lambda_[:, 1], lambda_[:, 2] # TODO: account for case where we don't learn lambda_2
        latent_z = lambda_2 * double_poisson_pmf(y0 - 1, y1 - 1, lambda_0, lambda_1, lambda_2) / double_poisson_pmf(y0, y1, lambda_0, lambda_1, lambda_2)
        latent_z[np.minimum(y0, y1) == 0] = 0

        if diagonal_pmf:
            p_diag = p * diagonal_pmf.pmf(y0)
            p_non_diag = (1 - p) * double_poisson_pmf(y0, y0, lambda_0, lambda_1, lambda_2)
            inflation_factors =  p_diag / (p_diag + p_non_diag)
            latent_v = np.where(y0 == y1, inflation_factors, 0)
            # get inflation factors as well

        # M-step
        # use old betas to guess
        # if diagonal-inflated, 
        beta_0 = poisson_regression(x, y0 - latent_z, weights=(1 - latent_v) if diagonal_pmf else None, init_guess=beta[:, 0] if warm_mstep else np.random.randn(*beta[:, 0].shape)) 
        beta_1 = poisson_regression(x, y1 - latent_z, weights=(1 - latent_v) if diagonal_pmf else None, init_guess=beta[:, 1] if warm_mstep else np.random.randn(*beta[:, 1].shape)) 
        beta_2 = poisson_regression(x, latent_z, weights=(1 - latent_v) if diagonal_pmf else None, init_guess=beta[:, 2] if warm_mstep else np.random.randn(*beta[:, 2].shape))
        beta = np.stack([beta_0, beta_1, beta_2], axis=1)
        if diagonal_pmf:
            p = latent_v.mean()
            theta = diagonal_pmf.update(y0, latent_v)
    if diagonal_pmf:
        return beta, p, theta
    else:
        return beta
