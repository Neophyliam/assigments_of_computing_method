import unittest
import numpy as np

from solver import qr


class TestQR(unittest.TestCase):

    epsilon = 1e-10

    def test_1x1(self):
        a = np.array([[3.0]])
        b = np.array([3.0])
        expected_x = np.array([1.])
        x = qr(a, b)
        self.assertTrue(np.allclose(x, expected_x, atol=self.epsilon))

    def test_2x2(self):
        a = np.array([[1., 1.],
                       [1., -1.]])
        b = np.array([[1.],
                       [0.]])
        expected_x = np.array([0.5, 0.5])
        x = qr(a, b)
        self.assertTrue(np.allclose(x, expected_x, atol=self.epsilon))

    def test_3x3(self):
        a = np.array([[3., 14, 9],
                       [6, 43, 3],
                       [6, 22, 15]])
        b = np.array([0., 0, 0])
        expected_x = np.array([0., 0., 0.])
        x = qr(a, b)
        self.assertTrue(np.allclose(x, expected_x, atol=self.epsilon))

    def test_4x3(self):
        a = np.array([[3., 14, 9],
                       [6, 43, 3],
                       [6, 22, 15],
                       [1, 1, 1]])
        b = np.array([26., 52, 43, 3])
        expected_x = np.array([1., 1., 1.])
        x = qr(a, b)
        self.assertTrue(np.allclose(x, expected_x, atol=self.epsilon))

    def test_7x7(self):
        a = np.array([[5., 4, 7, 5, 6, 7, 5],
                       [4, 12, 8, 7, 8, 8, 6],
                       [7, 8., 10, 9, 8, 7, 7],
                       [5, 7, 9, 11, 9, 7, 5],
                       [6, 8, 8, 9, 10, 8, 9],
                       [7, 8, 7, 7, 8, 10, 10],
                       [5, 6, 7, 5, 9, 10, 10]])
        b = np.array([39., 53, 56, 53, 58, 57, 52])
        expected_x = np.ones(7)
        x = qr(a, b)
        self.assertTrue(np.allclose(x, expected_x, atol=self.epsilon))


if __name__ == '__main__':
    unittest.main()
