import numpy as np

from utils import poisson_pmf

class InflationDistribution(object):
    def __init__(self, theta):
        self.theta = theta

    def pmf(self, x):
        pass

    def update(self, x, v):
        pass

class Geometric(InflationDistribution):
    def __init__(self, theta):
        super().__init__(theta)
        if theta < 0 or theta > 1:
            raise ValueError("Geometric parameter must be in [0, 1]")

    def pmf(self, x):
        return np.where(x >= 1, (1 - self.theta) ** (x - 1) * self.theta, 0)
    
    def update(self, x, v):
        self.theta = v.sum() / ((v * x).sum() + v.sum())
        return self.theta

class Poisson(InflationDistribution):
    def pmf(self, x):
        return poisson_pmf(x, self.theta)
    
    def update(self, x, v):
        self.theta = (v * x).sum() / v.sum()
        return self.theta

class Discrete(InflationDistribution):
    def __init__(self, theta, capacity=100):
        self.theta = np.zeros(capacity)
        self.theta[:len(theta)] = theta
        self.nonzero = len(theta)

    def pmf(self, x):
        return np.where((x < 0) | (x >= len(self.theta)), 0, self.theta[x])
    
    def update(self, x, v):
        self.theta[1:] = np.array([((x == i) * v).sum() for i in range(1, len(self.theta))]) / v.sum()
        self.theta[0] = 1 - self.theta[1:self.nonzero].sum()
        self.theta[self.nonzero:] = 0
        return self.theta



