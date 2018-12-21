import numpy as np


def chase(A, d):
    if d.ndim >= 2:
        d = d.reshape(d.size)
    n = d.shape[0]
    if A.ndim != 2 or A.shape[0] != n or A.shape[1] != n:
        raise AttributeError('The shape of argument `A` and `d` '
                'does not match')
    a0 = np.diag(A, -1)
    b = np.diag(A, 0)
    c0 = np.diag(A, 1)
    a = np.empty(n)
    a[1:] = a0
    c = np.empty(n)
    c[:-1] = c0
    return chase2(a, b, c, d)


def chase2(a, b, c, d):
    if a.ndim != 1 or b.ndim != 1 or c.ndim != 1 or d.ndim != 1:
        raise AttributeError('Arguments must be vectors')
    n = d.shape[0]
    if a.shape[0] != n or b.shape[0] != n or c.shape[0] != n:
        raise AttributeError('The shape of argument does not match')
    u = np.zeros(n)
    y = np.zeros(n)
    u[0] = b[0]
    y[0] = d[0]
    for i in range(1, n):
        l = a[i]/u[i-1]
        u[i] = b[i] - l * c[i-1]
        y[i] = d[i] - l * y[i-1]
    x = np.zeros(n)
    x[-1] = y[-1]/u[-1]
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - c[i] * x[i+1]) / u[i]
    return x
