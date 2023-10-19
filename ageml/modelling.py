"""Implement the modelling software.

Used in the AgeML project to enable the modelling of age.

Classes:
--------
AgeML - able to fit age models and predict age.
"""

import numpy as np

# Sklearn and Scipy do not automatically load submodules (avoids overheads)
from scipy import stats
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing

class AgeML:

    """Able to fit age models and predict age.

    This class allows the set up the pipeline for age modelling, to
    fit the models and to predict based on the fitted models.

    Parameters
    -----------

    Public methods:
    ---------------
    set_scaler(self, norm, **kwargs): Sets the scaler to use in the pipeline.

    set_model(self, model_type, **kwargs): Sets the model to use in the pipeline.

    set_pipeline(self): Sets the pipeline for age modelling.

    set_CV_params(self, CV_split, seed): Set the parameters of the Cross Validation scheme.

    calculate_metrics(self, y_true, y_pred): Calculates MAE, RMSE, R2 and p (Pearson's corelation)

    summary_metrics(self, array): Calculates mean and standard deviations of metrics.

    fit_age(self, X, y): Fit the age model.

    predict_age(self, X): Predict age with fitted model.
    """

    def __init__(self):
        self.scaler = None
        self.model = None
        self.pipeline = None
        self.pipelineFit = False
        self.CV_split = None
        self.seed = None

    def set_scaler(self, norm, **kwargs):
        """Sets the scaler to use in the pipeline.

        Parameters
        ----------
        norm: type of scaler to use
        **kwargs: to input to sklearn scaler object"""

        # Mean centered and unit variance
        if norm == 'standard':
            self.scaler = preprocessing.StandardScaler(**kwargs)

    def set_model(self, model_type, **kwargs):
        """Sets the model to use in the pipeline.

        Parameters
        ----------
        model_type: type of model to use
        **kwargs: to input to sklearn modle object"""

        # Linear Regression
        if model_type == 'linear':
            self.model = linear_model.LinearRegression(**kwargs)


    def set_pipeline(self):
        """Sets the model to use in the pipeline."""

        pipe = []
        if self.scaler is not None:
            pipe.append(('scaler', self.scaler))
        if self.model is not None:
            pipe.append(('model', self.model))
        else:
            raise TypeError("Must set a valid model before setting pipeline.")
        self.pipeline = pipeline.Pipeline(pipe)

    def set_CV_params(self, CV_split, seed=None):
       """Set the parameters of the Cross Validation Scheme.

       Parameters
       ----------
       CV_split: number of splits in CV scheme
       seed: seed to set random state."""

       self.CV_split = CV_split
       self.seed = seed

    def calculate_metrics(self, y_true, y_pred):
        """Calculates MAE, RMSE, R2 and p (Pearson's corelation)

        Parameters
        ----------
        y_true: 1D-Array with true ages; shape=n
        y_pred: 1D-Array with predicted ages; shape=n"""

        MAE = metrics.mean_absolute_error(y_true, y_pred)
        rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
        r2 = metrics.r2_score(y_true, y_pred)
        p, pval = stats.pearsonr(y_true, y_pred)
        return MAE, rmse, r2, p


    def summary_metrics(self, array):
        """Calculates mean and standard deviations of metrics.

        Parameters:
        -----------
        array: 2D-array with metrics in axis=0; shape=(n, 4)"""

        means = np.mean(array, axis=0)
        stds = np.std(array, axis=0)
        summary = []
        for mean, std in zip(means, stds):
            summary.append(mean)
            summary.append(std)
        return summary

    def fit_age(self, X, y):
        """Fit the age model.

        Parameters
        ----------
        X: 2D-Array with features; shape=(n,m)
        Y: 1D-Array with age; shape=n"""

        # Check that pipeline has been properly constructed
        if self.pipeline is None:
            raise TypeError("Must set a valid pipeline before running fit.")

        # Variables of interes
        pred_age = np.zeros(y.shape)
        metrics_train = []
        metrics_test = []

        # Apply cross-validation
        kf = model_selection.KFold(n_splits=self.CV_split, random_state=self.seed)
        for i, (train, test) in enumerate(kf.split(X)):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            #Train model
            self.pipeline.fit(X_train, y_train)

            # Predictions
            y_pred_train = self.pipeline.predict(X_train)
            y_pred_test = self.pipeline.predict(X_test)

            # Metrics
            print('Fold N: %d' % (i+1))
            metrics_train.append(self.calculate_metrics(y_train, y_pred_train))
            print('Train: MAE %.2f, RMSE %.2f, R2 %.3f, p %.3f' % metrics_train[i])
            metrics_test.append(self.calculate_metrics(y_test, y_pred_test))
            print('Test: MAE %.2f, RMSE %.2f, R2 %.3f, p %.3f' % metrics_test[i])

            # Save results
            pred_age[test] = y_pred_test

        # Calculate metrics over all splits
        print('Summary metrics over all CV splits')
        summary_train = self.summary_metrics(metrics_train)
        print('Train: MAE %.2f ± %.2f, RMSE %.2f ± %.2f, R2 %.3f ± %.3f, p %.3f ± %.3f'
              % tuple(summary_train))
        summary_test = self.summary_metrics(metrics_test)
        print('Test: MAE %.2f ± %.2f, RMSE %.2f ± %.2f, R2 %.3f ± %.3f, p %.3f ± %.3f'
              % tuple(summary_test))

        # Final model trained on all data
        self.pipeline.fit(X, y)
        self.pipelineFit = True

        return pred_age

    def predict_age(self, X):
        """Predict age with fitted model.

        Parameters:
        -----------
        X: 2D-Array with features; shape=(n,m)"""

        # Check that model has previously been fit
        if not self.pipelineFit:
            raise ValueError('Must fit the pipline befor calling predict.')

        return self.pipeline.predict(X)
