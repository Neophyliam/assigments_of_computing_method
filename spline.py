import numpy as np

from solver import qr, chase2


def get_coeff(x, prev_x, y, prev_y, m, prev_m):
    h = x - prev_x
    coeff3 = (m - prev_m)/(6*h)
    coeff2 = (x*prev_m-prev_x*m)/(2*h)
    coeff1 = (prev_x**2*m-x**2*prev_m)/(2*h)+(y-prev_y)/h+h*(prev_m-m)/6
    coeff0 = (prev_m*x**3-m*prev_x**3)/(6*h)+(prev_y*x-y*prev_x)/h-\
             h*(prev_m*x-m*prev_x)/6
    return coeff3, coeff2, coeff1, coeff0


def spline(x, y, bound, *args):
    """
    `x` and `y` are x values and y values in a list of discrete points.
    `bound` specifies the type of boundary condition. It is a value in 
    [1, 2, 3].
    `args` are specific values for each corresponding boundary condition.
    """
    pass
