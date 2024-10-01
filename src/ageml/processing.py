"""Define processing functions for AgeML package"""

import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


def find_correlations(X, Y):
    """Explore relationship between each individual features X and target variable Y.

    Parameters
    ----------
    X: 2D-Array with features; shape=(n,m)
    Y: 1D-Array with age; shape=n"""

    # Check for NaNs in arrays, raise ValueError if found
    if any(np.isnan(Y.flatten())):
        raise ValueError("NaN entrie(s) found in Y.")
    elif any(np.isnan(X.flatten())):
        raise ValueError("NaN entrie(s) found in X.")
    else:
        corrs = [stats.pearsonr(Y, X[:, i]) for i in range(X.shape[1])]
        corr_coefs, p_values = zip(*corrs)
        order = [o for o in np.argsort(np.abs(corr_coefs))[::-1] if corr_coefs[o] is not np.nan]
        return corr_coefs, order, p_values


def features_mutual_info(X, y):
    """Sort features by mutual information with target variable
    
    Inputs:
    -------
    X: numpy array with the features (n_samples, n_features)
    y: numpy array with the target variable (n_samples, )

    Output:
    -------
    order: list with the indices of the top features sorted by mutual information with the target variable
    mi_scores: numpy array with the mutual information scores of the selected features
    """

    # Calculate mutual information between each feature and the target variable
    mi_scores = mutual_info_regression(X, y)

    # Order from highest to lowest based on mutual information scores
    order = np.argsort(-mi_scores)
    
    return order, mi_scores


def feature_mutual_info_discrimination(X, y):
    """Sort features by discrimination between two groups and select top features using mutual information
    
    Inputs:
    -------
    X: numpy array with the features (n_samples, n_features)
    y: numpy array with the target labels (n_samples, ) (This must be zeros and ones)

    Output:
    -------
    order: list with the indices of the top discriminative features sorted by mutual information
    mi_scores: numpy array with the mutual information scores for the selected features
    """

    # Check if y contains only 0s and 1s
    if not np.array_equal(np.unique(y), np.array([0, 1])):
        raise ValueError("Target labels y must contain only 0s and 1s.")

    # Compute mutual information scores
    mi_scores = mutual_info_classif(X, y)

    # Order from highest to lowest based on mutual information scores
    order = np.argsort(-mi_scores)

    return order, mi_scores

def covariate_correction(X, Z, beta=None):
    """Correct for covariates Z in X using linear OLS.

    Parameters
    ----------
    X: 2D-Array with features; shape=(n,m)
    Z: 2D-Array with covariates; shape=(n,k)

    Returns
    -------
    X_residual: 2D-Array with corrected features; shape=(n,m)
    beta: 2D-Array with coefficients; shape=(m,k)
    """

    # Check NaN values
    if any(np.isnan(X.flatten())):
        raise ValueError("NaN entrie(s) found in Y.")
    elif any(np.isnan(Z.flatten())):
        raise ValueError("NaN entrie(s) found in X.")
    elif beta is not None and any(np.isnan(beta.flatten())):
        raise ValueError("NaN entrie(s) found in Z.")

    # Check shapes
    if X.shape[0] != Z.shape[0]:
        raise ValueError("X and Z must have the same number of rows.")

    # Estimate coefficients
    if beta is None:
        beta = np.linalg.inv(Z.T @ Z) @ Z.T @ X

    # Subtract the effect of Z from X
    X_residual = X - Z @ beta

    return X_residual, beta


def cohen_d(group1, group2):
    # Calculate the size of each group
    n1, n2 = len(group1), len(group2)
    
    # Calculate the mean of each group
    mean1, mean2 = np.mean(group1), np.mean(group2)
    
    # Calculate the variance of each group
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Calculate the pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    d = (mean1 - mean2) / pooled_std
    return d
