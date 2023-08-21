from argparse import ArgumentParser
from itertools import product
import os

import numpy as np
from tqdm.auto import tqdm

from inflation_distributions import Discrete
from regression import poisson_regression, double_poisson_regression
from utils import double_poisson_pmf

def demo_univariate_poisson():
    np.random.seed(42)
    x = np.random.randn(1000, 5)
    betas = np.arange(5) - 2
    y = np.random.poisson(np.exp(x.dot(betas)))
    betas_hat = poisson_regression(x, y, init_guess=np.random.randn(*betas.shape))
    print("True parameters:", betas)
    print("Estimated parameters:", betas_hat)
    print("MSE:", np.square(betas -  betas_hat).sum())

def demo_double_poisson():
    # WARNING: Convergence is finicky for len(x) < 1000
    np.random.seed(42)
    x = np.random.randn(5000, 5) 
    betas = np.stack([(np.arange(5) - 3) / 10, (np.arange(5) - 2) / 10, (np.arange(5) - 1) / 10], axis=1) # shape: (5, 3)
    y = np.random.poisson(np.exp(x.dot(betas))) # shape: (100, 3)
    betas_hat = double_poisson_regression(x, y[:, 0] + y[:, 2], y[:, 1] + y[:, 2], init_guess=np.random.randn(*betas.shape))
    print("True parameters:\n", betas)
    print("Estimated parameters:\n", betas_hat)
    print("MSE:", np.square(betas -  betas_hat).sum())

def demo_inflated_double_poisson():
    # WARNING: Convergence is finicky for len(x) < 1000
    np.random.seed(42)
    x = np.random.randn(5000, 5) 
    betas = np.stack([(np.arange(5) - 3) / 10, (np.arange(5) - 2) / 10, (np.arange(5) - 1) / 10], axis=1) # shape: (5, 3)
    theta = [0.6, 0.2, 0.1, 0.05, 0.05]
    p = 0.3
    diagonal_pmf = Discrete(theta)

    N = 15
    lambdas = np.exp(x.dot(betas))
    lambda_0, lambda_1, lambda_2 = lambdas[:, 0], lambdas[:, 1], lambdas[:, 2]
    if not os.path.isfile("./data/y0_inflated_double_poisson.npz") or not os.path.isfile("./data/y1_inflated_double_poisson.npz"):
        y0 = []
        y1 = []
        for k in tqdm(range(x.shape[0]), desc="Generating random data"):
            random_array = np.array([
                [
                    (1 - p) * double_poisson_pmf(i, j, lambda_0[k], lambda_1[k], lambda_2[k]) + (p * diagonal_pmf.pmf(i) if i == j else 0) for j in range(N)
                ] for i in range(N)
            ])
            instance_probs = random_array.ravel()
            tuples = list(product(range(N), range(N)))
            
            assert instance_probs.sum() > 0.99, instance_probs.sum()
            points0, points1 = tuples[np.random.choice(np.arange(N*N), p=instance_probs / instance_probs.sum())]
            y0.append(points0)
            y1.append(points1)
        y0 = np.array(y0)
        y1 = np.array(y1)
        np.savetxt("./data/y0_inflated_double_poisson.npz", y0)
        np.savetxt("./data/y1_inflated_double_poisson.npz", y1)
    else:
        y0 = np.loadtxt("./data/y0_inflated_double_poisson.npz").astype(int)
        y1 = np.loadtxt("./data/y1_inflated_double_poisson.npz").astype(int)

    guess_pmf = Discrete(np.ones_like(diagonal_pmf.theta) / len(diagonal_pmf.theta))
    betas_hat, p_hat, theta_hat = double_poisson_regression(x, y0, y1, diagonal_pmf=guess_pmf, init_p=0.5, init_guess=np.random.randn(*betas.shape))
    print("True parameters:", betas, p, theta, sep="\n")
    print("Estimated parameters:", betas_hat, p_hat, theta_hat[:diagonal_pmf.nonzero], sep="\n")
    print("MSE (beta):", np.square(betas -  betas_hat).sum())
    print("p-bias:", p_hat - p)
    print("MSE (theta):", np.square(theta - theta_hat[:diagonal_pmf.nonzero]).sum())

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--type", type=str, choices=["univariate", "double", "inflated"], required=True)
    args = psr.parse_args()
    if args.type == "univariate":
        demo_univariate_poisson()
    elif args.type  == "inflated":
        demo_inflated_double_poisson()
    else:
        demo_double_poisson()
    