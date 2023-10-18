"""Define processing functions for AgeML package"""

import numpy as np

from scipy import stats

def find_correlations(X, Y):
    """Explore relationship between each individual features X and target variable Y."""
    corr_coefs = [stats.pearsonr(Y, X[:, i])[0] for i in range(X.shape[1])]
    order = [o for o in np.argsort(np.abs(corr_coefs))[::-1] if corr_coefs[o] is not np.nan]
    return corr_coefs, order
