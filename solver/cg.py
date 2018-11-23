import numpy as np


def cg(a, b, x0, eps=1e-10):
    if a.ndim != 2:
        raise AttributeError('Illegal argument `a`')
    if b.ndim != 2:
        b = b.reshape(-1, 1)
    if x0.ndim != 2:
        x0 = x0.reshape(-1, 1)
    if a.shape[0] != b.shape[0] or a.shape[0] != x0.shape[0]:
        raise AttributeError('The shape of argument `a` and `b` '
            'or `a` and `x0` does not match')
    n = a.shape[0]
    r = b - np.dot(a, x0)
    d = r
    x = x0.copy()
    for k in range(n):
        alpha = np.dot(r.T, r) / np.dot(d.T, np.dot(a, d))
        x = x + alpha * d
        old_r, r = r, b - np.dot(a, x)
        if np.linalg.norm(r, 2) <= eps or k + 1 == n:
            return x
        beta = np.linalg.norm(r, 2)**2/np.linalg.norm(old_r, 2)**2
        d = r + beta * d
