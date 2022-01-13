import numpy as np

def omp(D, X, L):
    n = X.shape[0]
    P = X.shape[1]
    K = D.shape[1]
    A = np.zeros((K, P))
    for k in range(0, P):
        a = []
        x = X[:, k]
        residual = x
        index = []
        for j in range(0, L):
            proj = np.abs(np.dot(D.T, residual))
            pos = proj.argmax(axis=0)
            index.append(pos)
            a = np.dot(np.linalg.pinv(D[:, index]), x)
            residual = x - np.dot(D[:, index], a)
            if np.linalg.norm(residual, ord=2) < pow(10, -6):
                break
        tmp = np.zeros(K)
        tmp[index] = a
        A[:, k] = tmp
    return A