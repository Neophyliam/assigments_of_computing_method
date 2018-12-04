import unittest
import numpy as np

from solver import chase, chase2


class TestChase(unittest.TestCase):

    epsilon = 1e-10

    def test_5x5(self):
        A = np.array([
                [1., 2, 0, 0, 0],
                [2,  3, 1, 0, 0],
                [0, -3, 4, 2, 0],
                [0,  0, 4, 7, 1],
                [0,  0, 0,-5, 6]])
        d = np.array([5., 9, 2, 19, -4])
        x = chase(A, d)
        expected_x = np.array([1., 2, 1, 2, 1])
        self.assertTrue(np.allclose(x, expected_x, atol=self.epsilon))

    def test_20x20(self):
        A = np.zeros((20, 20))
        for i in range(20):
            A[i, i] = 4.
            if i != 0:
                A[i, i-1] = -1.
            if i != 19:
                A[i, i+1] = -1.
        a = np.zeros(20)
        a[:] = 4.
        b = np.zeros(20)
        b[:] = -1.
        c = np.zeros(20)
        c[:] = -1.
        d = np.zeros(20)
        d[0] = d[-1] = 3.
        d[1:-1] = 2.
        expected_x = np.ones(20)
        x1 = chase(A, d)
        x2 = chase2(b, a, c, d)
        self.assertTrue(np.allclose(x1, expected_x, atol=self.epsilon))
        self.assertTrue(np.allclose(x2, expected_x, atol=self.epsilon))
