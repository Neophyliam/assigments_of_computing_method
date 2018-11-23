import unittest
import numpy as np

from solver import cg


class TestCG(unittest.TestCase):

    epsilon = 1e-10

    def test_1x1(self):
        a = np.array([[2.]])
        b = np.array([[1.]])
        x0 = np.array([[0.]])
        expected_x = np.array([[0.5]])
        x = cg(a, b, x0)
        self.assertTrue(np.allclose(x, expected_x))

    def test_3x3(self):
        a = np.array([[2., 0, 1],
                      [0, 1, 0],
                      [1, 0, 2]])
        b = np.array([3., 1, 3])
        x0 = np.array([0., 0, 0])
        expected_x = np.ones((3, 1))
        x = cg(a, b, x0)
        self.assertTrue(np.allclose(x, expected_x, atol=self.epsilon))

    def nd_abx(self, n):
        # N-dimension argument `a`, `b`, `x0` and expected result `x` generator
        a = np.zeros((n, n))
        for i in range(n):
            a[i, i] = -2.
        for i in range(n-1):
            a[i, i+1] = 1.
        for i in range(1, n):
            a[i, i-1] = 1.
        b = np.zeros((n, 1))
        b[0, 0] = -1.
        b[n-1, 0] = -1.
        x0 = np.zeros((n, 1))
        x = np.ones((n, 1))
        return a, b, x0, x

    def test_100x100(self):
        a, b, x0, expected_x = self.nd_abx(100)
        x = cg(a, b, x0)
        self.assertTrue(np.allclose(x, expected_x, atol=self.epsilon))

    def test_200x200(self):
        a, b, x0, expected_x = self.nd_abx(200)
        x = cg(a, b, x0)
        self.assertTrue(np.allclose(x, expected_x, atol=self.epsilon))

    def test_400x400(self):
        a, b, x0, expected_x = self.nd_abx(400)
        x = cg(a, b, x0)
        self.assertTrue(np.allclose(x, expected_x, atol=self.epsilon))


if __name__ == '__main__':
    unittest.main()
