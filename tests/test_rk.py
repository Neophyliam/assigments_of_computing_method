import unittest
import math
import numpy as np

from solver import rk


def f(x, y):
    f1 = y[1]
    f2 = y[2]
    f3 = y[2] + y[1] - y[0] + 2*x - 3
    return np.array([f1, f2, f3])


class TestRK(unittest.TestCase):

    def test_rk(self):
        y = rk(f, [-1., 3, 2], [0., 1.], 0.05)
        expect_result = math.e + 1
        error = y[-1][0] - expect_result
        print('error for test case of RK: {}'.format(error))
