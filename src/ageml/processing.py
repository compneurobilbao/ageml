"""Define processing functions for AgeML package"""

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict
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

class FoldMetrics(ABC):
    """Abstract base class for fold metrics"""
    @classmethod
    @abstractmethod
    def get_metric_names(cls) -> List[str]:
        """Return the list of metric names for this metric type"""
        pass

@dataclass
class RegressionFoldMetrics(FoldMetrics):
    mae: float
    rmse: float
    r2: float
    p: float
    
    @classmethod
    def get_metric_names(cls) -> List[str]:
        return ['mae', 'rmse', 'r2', 'p']


@dataclass
class ClassificationFoldMetrics(FoldMetrics):
    auc: float
    accuracy: float
    sensitivity: float
    specificity: float
    
    @classmethod
    def get_metric_names(cls) -> List[str]:
        return ['auc', 'accuracy', 'sensitivity', 'specificity']


class CVMetricsHandler:
    def __init__(self, task_type: str):
        if task_type not in ['regression', 'classification']:
            raise ValueError('task_type must be either "regression" or "classification"')
        else:
            self.task_type = task_type
        self.train_metrics: List[FoldMetrics] = []
        self.test_metrics: List[FoldMetrics] = []
    
    def add_fold_metrics(self, fold_train: FoldMetrics, fold_test: FoldMetrics):
        # Validate the input types match the task type
        expected_class = RegressionFoldMetrics if self.task_type == 'regression' else ClassificationFoldMetrics
        
        if not isinstance(fold_train, expected_class) or not isinstance(fold_test, expected_class):
            raise TypeError(f"Metrics should be of type {expected_class.__name__} for task_type '{self.task_type}'")
        
        self.train_metrics.append(fold_train)
        self.test_metrics.append(fold_test)
    
    def _calculate_summary(self, metrics_list: List[FoldMetrics]) -> Dict[str, Dict[str, float]]:
        if not metrics_list:
            return {}
        
        # Get metric names from the class of the first metrics object
        metric_names = metrics_list[0].__class__.get_metric_names()
        
        all_metrics = {}
        for metric_name in metric_names:
            all_metrics[metric_name] = [getattr(m, metric_name) for m in metrics_list]
        
        summary = {}
        for metric_name, values in all_metrics.items():
            summary[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                '95ci': tuple(stats.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=stats.sem(values)))
            }
        return summary
    
    def get_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        return {
            'train': self._calculate_summary(self.train_metrics),
            'test': self._calculate_summary(self.test_metrics)
        }
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        summary = self.get_summary()
        data = []
        
        # Get metric names from the actual metrics objects
        metrics_class = RegressionFoldMetrics if self.task_type == 'regression' else ClassificationFoldMetrics
        metrics = metrics_class.get_metric_names()
        
        for split in ['train', 'test']:
            for metric in metrics:
                for stat, value in summary[split][metric].items():
                    data.append({
                        'split': split,
                        'metric': metric,
                        'statistic': stat,
                        'value': value
                    })
        
        return pd.DataFrame(data)