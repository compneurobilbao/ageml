"""Define processing functions for AgeML package"""

import numpy as np

from scipy import stats

def find_correlations(X, Y):
    """Explore relationship between each individual features X and target variable Y.

    Parameters
    ----------
    X: 2D-Array with features; shape=(n,m)
    Y: 1D-Array with age; shape=n"""

    corr_coefs = [stats.pearsonr(Y, X[:, i])[0] for i in range(X.shape[1])]
    order = [o for o in np.argsort(np.abs(corr_coefs))[::-1] if corr_coefs[o] is not np.nan]
    return corr_coefs, order