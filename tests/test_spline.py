import unittest
import numpy as np

from spline import spline, plot_points


class TestChase(unittest.TestCase):

    epsilon = 1e-10

    def test_bound1(self):
        x = [-3., -1., 0., 3., 4.]
        y = [7., 11., 26., 56., 29.]
        results = np.array(spline(x, y, 1))
        expected_results = np.array([
            [1., 9, 25, 28],
            [-1, 3, 19, 26],
            [-2, 3, 19, 26],
            [5, -60, 208, -163]
        ])
        self.assertTrue(np.allclose(results, expected_results, atol=self.epsilon))

    def test_plot_points(self):
        x = [-3., -1., 0., 3., 4.]
        y = [7., 11., 26., 56., 29.]
        coeffs = np.array(spline(x, y, 1))
        plot_x, plot_y = plot_points(x, coeffs)
        self.assertEqual(len(plot_x), len(plot_y))
        print('number of plot points: %d' % len(plot_x))
        x1 = plot_x[:-1]
        x2 = plot_x[1:]
        for i in range(len(plot_x)-1):
            self.assertGreater(x2[i], x1[i])