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

    Return a list of coefficients in each segment.
    """
    n = len(x)
    if len(y) != n:
        raise AttributeError('Argument `x` and `y` does not match')
    if bound not in [1, 2, 3]:
        raise ValueError('Argument `bound` must be one of [1, 2, 3]')
    h = np.empty(n)
    mu = np.empty(n)
    lambda_ = np.empty(n)
    d = np.empty(n)
    for i in range(1, n):
        h[i] = x[i] - x[i-1]
    for i in range(1, n-1):
        mu[i] = h[i]/(h[i]+h[i+1])
        lambda_[i] = 1. - mu[i]
        d[i] = 6./(h[i]+h[i+1])*((y[i+1]-y[i])/h[i+1]-(y[i]-y[i-1])/h[i])
    M = np.empty(n)
    if bound == 1:
        if len(args) == 0:
            args = (0., 0.)
        if len(args) != 2:
            raise AttributeError('Need the moments in both end points when '
                    '`bound`==1')
        M[0] = args[0]
        M[-1] = args[1]
        a = np.empty(n-2)
        a[1:] = mu[2:n-1]
        b = 2.*np.ones(n-2)
        c = np.empty(n-2)
        c[:-1] = lambda_[1:n-2]
        d2 = np.empty(n-2)
        d2[1:-1] = d[2:n-2]
        d2[0] = d[1] - mu[1]*M[0]
        d2[-1] = d[n-2] - lambda_[n-2] * M[-1]
        result = chase2(a, b, c, d2)
        M[1:-1] = result
    elif bound == 2:
        if len(args) != 2:
            raise AttributeError('Need the derivative values in both end '
                    'points when `bound`==2')
        d[0] = 6./h[1]*((y[1]-y[0])/h[1]-args[0])
        d[-1] = 6./h[-1]*(args[1]-(y[-1]-y[-2])/h[-1])
        mu[-1] = 1.
        lambda_[0] = 1.
        diag = 2.*np.ones(n)
        M = chase2(mu, diag, lambda_, d)
    elif bound == 3:
        d[-1] = 6./(h[-1]+h[1])*((y[1]-y[-1])/h[1]-(y[-1]-y[-2])/h[-1])
        mu[-1] = h[-1]/(h[-1]+h[1])
        lambda_[-1] = 1. - mu[-1]
        a = np.zeros((n-1, n-1))
        for i in range(n-1):
            a[i, i] = 2.
            if i == 0:
                a[i, i+1] = lambda_[1]
                a[i, -1] = mu[1]
            elif i == n-2:
                a[i, 0] = lambda_[-1]
                a[i, i-1] = mu[-1]
            else:
                a[i, i-1] = mu[i+1]
                a[i, i+1] = lambda_[i+1]
        d = d[1:]
        result = qr(a, d)
        M[1:] = result
        M[0] = M[-1]
    coeffs = []
    for i in range(1, n):
        coeffs.append(get_coeff(x[i], x[i-1], y[i], y[i-1], M[i], M[i-1]))
    return coeffs


def plot_points(x, coeffs, n=100):
    """
    Give a series of plot point.

    `x` should be same as the `x` argument in `spline` function, which
    specify the domain for each spline function. 

    `coeffs` is the result of `spline` function.

    `n` is the desired number of plot points from `x[0]` to `x[-1]`.

    Return a list of x values for plot points and a list of y values for 
    plot points.
    """
    seg_num = len(x) - 1
    result_x, result_y = [], []
    for i in range(seg_num):
        point_num = round((x[i+1]-x[i])/(x[-1]-x[0])*n)
        coeff = np.array(coeffs[i]).reshape(1, 4)
        x_in_seg = np.linspace(x[i], x[i+1], point_num)
        bases = np.array([x_in_seg**3, x_in_seg**2, x_in_seg, np.ones(len(x_in_seg))])
        y_in_seg = np.dot(coeff, bases).reshape(-1)
        result_x.extend(x_in_seg)
        result_y.extend(y_in_seg)
        last_x = result_x.pop()
        last_y = result_y.pop()
    result_x.append(last_x)
    result_y.append(last_y)
    return result_x, result_y
