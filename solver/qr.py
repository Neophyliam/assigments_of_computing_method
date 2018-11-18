import numpy as np


def qr(a, b):
    # matrix `a` has m rows, n columns
    m, n = a.shape
    if m < n:
        raise AttributeError('Illegal argument `a`')
    b = b.reshape(-1, 1)
    if b.shape[0] != m or b.shape[1] != 1:
        raise AttributeError('Illegal argument `b`')
    # initialize vector `d`
    d = np.zeros(n)
    # make matrix `a` a combination of orginal matrix `a` and `b`
    a = np.hstack((a, b))
    for k in range(0, n-1):
        sigma = np.sqrt(np.square(a[k:, k]).sum())
        if a[k, k] >= 0:
            sigma = -sigma
        d[k] = sigma
        alpha = sigma * (sigma - a[k, k])
        a[k, k] = a[k, k] - sigma
        for j in range(k + 1, n + 1):
            beta = (a[k, k]*a[k, j]+np.sum(a[k+1:, k]*a[k+1:, j]))/alpha
            a[k, j] = a[k, j] - beta * a[k, k]
            for i in range(k + 1, m):
                a[i, j] = a[i, j] - beta * a[i, k]
            # Can the above 3 lines be substituted by these two lines?
            # for i in range(k, m + 1):
            #     a[i, j] = a[i, j] - beta * a[i, k]
    if m == n:
        d[n-1] = a[n-1, n-1]
    if m > n:
        sigma = np.sqrt(np.square(a[n-1:, n-1]).sum())
        if a[n-1, n-1] >= 0:
            sigma = -sigma
        d[n-1] = sigma
        alpha = sigma * (sigma - a[n-1, n-1])
        a[n-1, n-1] = a[n-1, n-1] - sigma
        beta = (a[n-1, n-1] * a[n-1, n] + np.sum(a[n:, n-1] * a[n:, n])) / alpha
        a[n-1, n] = a[n-1, n] - beta * a[n-1, n-1]
        for i in range(n, m):
            a[i, n] = a[i, n] - beta * a[i, n-1]
    # Can the above 3 lines be substituted by these two lines?
    # for i in range(n-1, m):
    #     a[i, j] = a[i, j] - beta * a[i, k]
    x = np.zeros(n)
    x[n-1] = a[n-1, n] / d[n-1]
    for k in range(n-2, -1, -1):
        x[k] = (a[k, n] - np.sum(a[k, k+1:n] * x[k+1:])) / d[k]
    return x
