import numpy as np


def rk(f, y0, bound, h):
    x0 = bound[0]
    xn = bound[1]
    point_num = int((xn-x0)/h) + 1
    h = (xn-x0)/(point_num-1)
    x = np.linspace(x0, xn, point_num)
    y = np.empty(point_num, dtype=np.ndarray)
    y[0] = np.array(y0)
    for i in range(1, len(x)):
        k1 = h*f(x[i-1], y[i-1])
        k2 = h*f(x[i-1]+h/2, y[i-1]+k1/2.)
        k3 = h*f(x[i-1]+h/2, y[i-1]+k2/2.)
        k4 = h*f(x[i-1]+h, y[i-1]+k3)
        y[i] = y[i-1] + (k1+2*k2+2*k3+k4)/6.0
    return y
