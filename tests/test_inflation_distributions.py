import os
PYPOISSON_DIR = os.path.dirname(os.path.dirname(__file__))

import sys
if sys.path[0] != PYPOISSON_DIR:
    sys.path.insert(0, PYPOISSON_DIR)

import numpy as np
import unittest
from inflation_distributions import Geometric, Poisson, Discrete

class TestGeometric(unittest.TestCase):

    def setUp(self):
        self.pmf = Geometric(0.5)

    def test_pmf(self):
        self.assertEqual(self.pmf.pmf(1), 0.5)

class TestPoisson(unittest.TestCase):
    def setUp(self):
        self.pmf = Poisson(2)

    def test_pmf(self):
        self.assertEqual(self.pmf.pmf(3), 8 * np.exp(-2) / 6)

if __name__ == '__main__':
    unittest.main()