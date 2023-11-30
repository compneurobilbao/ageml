"""Implement the modelling software.

Used in the AgeML project to enable the modelling of age.

Classes:
--------
AgeML - able to fit age models and predict age.
Classifier - classifier of class labels based on deltas.
"""

import numpy as np
import scipy.stats as st

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
    scaler_type: string indicating type of scaler to use
    scaler_params: dictionary to pass as **kwargs to scaler
    model_type: string indicating type of model to use
    model_params: dictionary to pass as **kwargs to model
    CV_split: integer number of CV splits
    seed: integer seed for randomization

    Public methods:
    ---------------
    set_scaler(self, norm, **kwargs): Sets the scaler to use in the pipeline.

    set_model(self, model_type, **kwargs): Sets the model to use in the pipeline.

    set_pipeline(self): Sets the pipeline for age modelling.

    set_CV_params(self, CV_split, seed): Set the parameters of the Cross Validation scheme.

    calculate_metrics(self, y_true, y_pred): Calculates MAE, RMSE, R2 and p (Pearson's corelation)

    summary_metrics(self, array): Calculates mean and standard deviations of metrics.

    fit_age_bias(self, y_true, y_pred): Fit a linear age bias correction model.

    predict_age_bias(self, y_true, y_pred): Apply age bias correction.

    fit_age(self, X, y): Fit the age model.

    predict_age(self, X): Predict age with fitted model.
    """

    def __init__(self, scaler_type, scaler_params, model_type, model_params, CV_split, seed):
        """Initialise variables."""

        # Set required modelling parts
        self.set_scaler(scaler_type, **scaler_params)
        self.set_model(model_type, **model_params)
        self.set_pipeline()
        self.set_CV_params(CV_split, seed)

        # Initialise flags
        self.pipelineFit = False
        self.age_biasFit = False

    def set_scaler(self, norm, **kwargs):
        """Sets the scaler to use in the pipeline.

        Parameters
        ----------
        norm: type of scaler to use
        **kwargs: to input to sklearn scaler object"""

        # Mean centered and unit variance
        if norm == "standard":
            self.scaler = preprocessing.StandardScaler(**kwargs)
        else:
            raise ValueError("Must select an available scaler type.")

    def set_model(self, model_type, **kwargs):
        """Sets the model to use in the pipeline.

        Parameters
        ----------
        model_type: type of model to use
        **kwargs: to input to sklearn modle object"""

        # Linear Regression
        if model_type == "linear":
            self.model = linear_model.LinearRegression(**kwargs)
        else:
            raise ValueError("Must select an available model type.")

    def set_pipeline(self):
        """Sets the model to use in the pipeline."""

        pipe = []
        if self.scaler is None or self.model is None:
            raise ValueError("Must set a valid model or scaler before setting pipeline.")
        
        pipe.append(("scaler", self.scaler))
        pipe.append(("model", self.model))
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

    def fit_age_bias(self, y_true, y_pred):
        """Fit a linear age bias correction model.

        Parameters
        ----------
        y_true: 1D-Array with true age; shape=n
        y_pred: 1D-Array with predicted age; shape=n."""

        # Train linear regression model
        self.age_bias = linear_model.LinearRegression(fit_intercept=True)
        self.age_bias.fit(y_true.reshape(-1, 1), y_pred)
        self.age_biasFit = True

    def predict_age_bias(self, y_true, y_pred):
        """Apply age bias correction.

        Parameters
        ----------
        y_true: 1D-Array with true age; shape=n
        y_pred: 1D-Array with predicted age; shape=n."""

        # Check that age bias has been fit
        if not self.age_biasFit:
            raise TypeError("Must fun fit_age_bias before attempting to predict.")

        # Apply linear correction
        y_corrected = y_pred + (y_true - self.age_bias.predict(y_true.reshape(-1, 1)))

        return y_corrected

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
        corrected_age = np.zeros(y.shape)
        metrics_train = []
        metrics_test = []

        # Apply cross-validation
        kf = model_selection.KFold(n_splits=self.CV_split, random_state=self.seed, shuffle=True)
        for i, (train, test) in enumerate(kf.split(X)):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            # Train model
            self.pipeline.fit(X_train, y_train)

            # Predictions
            y_pred_train = self.pipeline.predict(X_train)
            y_pred_test = self.pipeline.predict(X_test)

            # Metrics
            print("Fold N: %d" % (i + 1))
            metrics_train.append(self.calculate_metrics(y_train, y_pred_train))
            print("Train: MAE %.2f, RMSE %.2f, R2 %.3f, p %.3f" % metrics_train[i])
            metrics_test.append(self.calculate_metrics(y_test, y_pred_test))
            print("Test: MAE %.2f, RMSE %.2f, R2 %.3f, p %.3f" % metrics_test[i])

            # Fit and apply age-bias correction
            self.fit_age_bias(y_train, y_pred_train)
            y_pred_test_no_bias = self.predict_age_bias(y_test, y_pred_test)

            # Save results of hold out
            pred_age[test] = y_pred_test
            corrected_age[test] = y_pred_test_no_bias

        # Calculate metrics over all splits
        print("Summary metrics over all CV splits")
        summary_train = self.summary_metrics(metrics_train)
        print("Train: MAE %.2f ± %.2f, RMSE %.2f ± %.2f, R2 %.3f ± %.3f, p %.3f ± %.3f" % tuple(summary_train))
        summary_test = self.summary_metrics(metrics_test)
        print("Test: MAE %.2f ± %.2f, RMSE %.2f ± %.2f, R2 %.3f ± %.3f, p %.3f ± %.3f" % tuple(summary_test))

        # Final model trained on all data
        self.pipeline.fit(X, y)
        self.pipelineFit = True
        y_pred = self.pipeline.predict(X)
        self.fit_age_bias(y, y_pred)

        return pred_age, corrected_age

    def predict_age(self, X, y=None):
        """Predict age with fitted model.

        Parameters:
        -----------
        X: 2D-Array with features; shape=(n,m)
        y: 1D-Array with age; shape=n"""

        # Check that model has previously been fit
        if not self.pipelineFit:
            raise ValueError("Must fit the pipline before calling predict.")
        if y is not None and not self.age_biasFit:
            raise ValueError("Must fit the age bias before calling predict with bias correction.")

        # Predict age
        y_pred = self.pipeline.predict(X)

        # Apply age bias correction
        if y is not None:
            y_corrected = self.predict_age_bias(y, y_pred)
        else:
            y_corrected = y_pred

        return y_pred, y_corrected


class Classifier:
    
    """Classifier of class labels based on deltas.
    
    This class allows the differentiation of two groups based
    on differences in their deltas based on a logistic regresor.

    Public methods:
    ---------------
    set_model(self): Sets the model to use in the pipeline.

    fit_model(self, X, y): Fit the model.
    """

    def __init__(self):
        """Initialise variables."""

        # Set required modelling parts
        self.set_model()

        # Set default parameters
        # TODO: let user choose this
        self.CV_split = 5
        self.seed = 0
        self.thr = 0.5
        self.ci_val = 0.95

        # Initialise flags
        self.modelFit = False

    def set_model(self):
        """Sets the model to use in the pipeline."""

        self.model = linear_model.LogisticRegression()
    
    def fit_model(self, X, y):
        """Fit the model.

        Parameters
        ----------
        X: 2D-Array with features; shape=(n,m)
        y: 1D-Array with labbels; shape=n"""

        # Arrays to store  values
        accs, aucs, spes, sens = [], [], [], []
        y_preds = np.empty(shape=y.shape)
    
        kf = model_selection.KFold(n_splits=self.CV_split, shuffle=True, random_state=self.seed)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
    
            # Fit the model using the training data
            self.model.fit(X_train, y_train)

            # Use model to predict probability of tests
            y_pred = self.model.predict_proba(X_test)[::, 1]
            y_preds[test_index] = y_pred

            # Calculate AUC of model
            auc = metrics.roc_auc_score(y_test, y_pred)
            aucs.append(auc)
    
            # Calculate relevant metrics
            acc = metrics.accuracy_score(y_test, y_pred > self.thr)
            tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred > self.thr).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fp)
            accs.append(acc)
            sens.append(sensitivity)
            spes.append(specificity)

        # Compute confidence intervals
        ci_accs = st.t.interval(alpha=self.ci_val, df=len(accs) - 1, loc=np.mean(accs), scale=st.sem(accs))
        ci_aucs = st.t.interval(alpha=self.ci_val, df=len(aucs) - 1, loc=np.mean(aucs), scale=st.sem(aucs))
        ci_sens = st.t.interval(alpha=self.ci_val, df=len(sens) - 1, loc=np.mean(sens), scale=st.sem(sens))
        ci_spes = st.t.interval(alpha=self.ci_val, df=len(spes) - 1, loc=np.mean(spes), scale=st.sem(spes))

        # Print results
        print('Summary metrics over all CV splits (95% CI)')
        print('AUC: %.3f [%.3f-%.3f]' % (np.mean(aucs), ci_aucs[0], ci_aucs[1]))
        print('Accuracy: %.3f [%.3f-%.3f]' % (np.mean(accs), ci_accs[0], ci_accs[1]))
        print('Sensitivity: %.3f [%.3f-%.3f]' % (np.mean(sens), ci_sens[0], ci_sens[1]))
        print('Specificity: %.3f [%.3f-%.3f]' % (np.mean(spes), ci_spes[0], ci_spes[1]))

        # Final model trained on all data
        self.model.fit(X, y)

        # Set flag
        self.modelFit = True

        return y_preds
    
    def predict(self, X):
        """Predict class labels with fitted model.

        Parameters:
        -----------
        X: 2D-Array with features; shape=(n,m)"""

        # Check that model has previously been fit
        if not self.modelFit:
            raise ValueError("Must fit the classifier before calling predict.")

        # Predict class labels
        y_pred = self.model.predict_proba(X)[::, 1]

        return y_pred
