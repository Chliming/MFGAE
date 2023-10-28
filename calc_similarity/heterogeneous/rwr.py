import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale


def _scaleSimMat(A):
    """Scale rows of similarity matrix"""
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=0) == 0)
    col = A.sum(axis=0)
    A = A.astype(float)/col[:, None]
    return A


def RWR(A, K=3, alpha=0.98):
    """Random Walk on graph"""
    A = _scaleSimMat(A)
    # Random surfing
    n = A.shape[0]
    P0 = np.eye(n, dtype=float)
    P = P0.copy()
    M = np.zeros((n, n), dtype=float)
    for i in range(0, K):
        P = alpha * np.dot(P, A) + (1. - alpha) * P0
        M = M + P
    return M


network = np.loadtxt(r".data\integration_sim\heterogeneous_3_2.txt")

network = RWR(network)
network = minmax_scale(network)

#np.savetxt(r".data\integration_sim\heterogeneous_rwr_3_2.txt", network)

print(1)