import unittest
import numpy as np

from solver import qr


class TestQR(unittest.TestCase):

    epsilon = 1e-10

    def test_1x1(self):
        a0 = np.array([[3.0]])
        b0 = np.array([3.0])
        expected_x0 = np.array([1.])
        result = (qr(a0, b0) - expected_x0) < self.epsilon
        self.assertTrue(all(result))

    def test_2x2(self):
        a1 = np.array([[1., 1.],
                       [1., -1.]])
        b1 = np.array([[1.],
                       [0.]])
        expected_x1 = np.array([0.5, 0.5])
        result = (qr(a1, b1) - expected_x1) < self.epsilon
        self.assertTrue(all(result))

    def test_3x3(self):
        a2 = np.array([[3., 14, 9],
                       [6, 43, 3],
                       [6, 22, 15]])
        b2 = np.array([0., 0, 0])
        expected_x2 = np.array([0., 0., 0.])
        result = (qr(a2, b2) - expected_x2) < self.epsilon
        self.assertTrue(all(result))

    def test_4x3(self):
        a3 = np.array([[3., 14, 9],
                       [6, 43, 3],
                       [6, 22, 15],
                       [1, 1, 1]])
        b3 = np.array([26., 52, 43, 3])
        expected_x3 = np.array([1., 1., 1.])
        result = (qr(a3, b3) - expected_x3) < self.epsilon
        self.assertTrue(all(result))

    def test_7x7(self):
        a4 = np.array([[5., 4, 7, 5, 6, 7, 5],
                       [4, 12, 8, 7, 8, 8, 6],
                       [7, 8., 10, 9, 8, 7, 7],
                       [5, 7, 9, 11, 9, 7, 5],
                       [6, 8, 8, 9, 10, 8, 9],
                       [7, 8, 7, 7, 8, 10, 10],
                       [5, 6, 7, 5, 9, 10, 10]])
        b4 = np.array([39., 53, 56, 53, 58, 57, 52])
        expected_x4 = np.ones(7)
        result = (qr(a4, b4) - expected_x4) < self.epsilon
        self.assertTrue(all(result))


if __name__ == '__main__':
    unittest.main()
