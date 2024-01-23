"""Implement the modelling software.

Used in the AgeML project to enable the modelling of age.

Classes:
--------
AgeML - able to fit age models and predict age.
Classifier - classifier of class labels based on deltas.
"""

import numpy as np
import scipy.stats as st
from xgboost import XGBRegressor

# Sklearn and Scipy do not automatically load submodules (avoids overheads)
from scipy import stats
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


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
    optimize_hyperparams (Bool): perform hyperparameter optimization

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
    
    # Scaler dictionary
    scaler_dict = {
        "maxabs": MaxAbsScaler,
        "minmax": MinMaxScaler,
        "normalizer": Normalizer,
        "power": PowerTransformer,
        "quantile": QuantileTransformer,
        "robust": RobustScaler,
        "standard": StandardScaler,
    }
    scaler_hyperparameters = {'maxabs': {},
                              'minmax': {},
                              'normalizer': {},
                              'power': {'method': ['yeo-johnson', 'box-cox']},
                              'quantile': {'n_quantiles': [10, 1000],
                                           'output_distribution': ['normal', 'uniform']},
                              'robust': {},
                              'standard': {}}
    
    # Model dictionary
    model_dict = {
        "linear_reg": linear_model.LinearRegression,
        "ridge": linear_model.Ridge,
        "lasso": linear_model.Lasso,
        "linear_svr": svm.SVR,
        "xgboost": XGBRegressor,  # XGBoost
        "rf": RandomForestRegressor,
    }
    model_hyperparameter_ranges = {'ridge': {'alpha': [-3, 3]},
                                   'lasso': {'alpha': [-3, 3]},
                                   'linear_svr': {'C': [-3, 3],
                                                  'epsilon': [-3, 3]},
                                   'xgboost': {'max_depth': [-3, 3],
                                               'min_child_weight': [-3, 3],
                                               'subsample': [-3, 3],
                                               'colsample_bytree': [-3, 3],
                                               'eta': [-3, 3],
                                               'gamma': [-3, 3],
                                               'lambda': [-3, 3],
                                               'alpha': [-3, 3]},
                                   'rf': {'n_estimators': [-3, 3],
                                          'max_depth': [-3, 3],
                                          'min_samples_split': [-3, 3],
                                          'min_samples_leaf': [-3, 3],
                                          'max_features': [-3, 3], }}

    model_hyperparameter_types = {'ridge': {'alpha': 'log'},
                                  'lasso': {'alpha': 'log'},
                                  'linear_svr': {'C': 'log',
                                                 'epsilon': 'log'},
                                  'xgboost': {'max_depth': 'int',
                                              'min_child_weight': 'int',
                                              'subsample': 'float',
                                              'colsample_bytree': 'float',
                                              'eta': 'float',
                                              'gamma': 'float',
                                              'lambda': 'float',
                                              'alpha': 'float'},
                                  'rf': {'n_estimators': 'int',
                                         'max_depth': 'int',
                                         'min_samples_split': 'int',
                                         'min_samples_leaf': 'int',
                                         'max_features': 'int'}}

    def __init__(self, scaler_type, scaler_params, model_type, model_params, CV_split, seed,
                 hyperparameter_tuning: int = 0, feature_extension: int = 0):
        """Initialise variables."""

        # Scaler dictionary
        self.scaler_type = scaler_type
        self.scaler_dict = AgeML.scaler_dict
        
        # Model dictionary
        self.model_type = model_type
        self.model_dict = AgeML.model_dict
        # Hyperparameters and feature extension
        self.hyperparameter_tuning = hyperparameter_tuning
        self.feature_extension = feature_extension
        
        # Set required modelling parts
        self.set_scaler(scaler_type, **scaler_params)
        self.set_model(model_type, **model_params)
        self.set_pipeline()
        self.set_CV_params(CV_split, seed)
        self.set_hyperparameter_grid()

        # Initialise flags
        self.pipelineFit = False
        self.age_biasFit = False

    def set_hyperparameter_grid(self):
        """Builds the hyperparameter grid of the selected model upon AgeML object initialization

        Returns:
            dict: dictionary with the hyperparameter grid
        """
        param_grid = {}
        if self.model_type in AgeML.model_dict.keys() and self.hyperparameter_tuning > 0:
            hyperparam_ranges = AgeML.model_hyperparameter_ranges[self.model_type]
            hyperparam_types = AgeML.model_hyperparameter_types[self.model_type]
            # Initialize output grid
            for hyperparam_name in list(hyperparam_types.keys()):
                bounds = hyperparam_ranges[hyperparam_name]
                if hyperparam_types[hyperparam_name] == 'log':
                    param_grid[f"model__{hyperparam_name}"] = np.logspace(bounds[0], bounds[1],
                                                                          int(self.hyperparameter_tuning))
                elif hyperparam_types[hyperparam_name] == 'int':
                    param_grid[f"model__{hyperparam_name}"] = np.rint(np.linspace(bounds[0], bounds[1],
                                                                      int(self.hyperparameter_tuning))).astype(int)
                elif hyperparam_types[hyperparam_name] == 'float':
                    param_grid[f"model__{hyperparam_name}"] = np.logspace(bounds[0], bounds[1],
                                                                          int(self.hyperparameter_tuning))
        else:
            print("No hyperparameter grid was built for the selected model. No hyperparameters available.")
        self.hyperparameter_grid = param_grid

    def set_scaler(self, norm, **kwargs):
        """Sets the scaler to use in the pipeline.

        Parameters
        ----------
        norm: type of scaler to use
        **kwargs: to input to sklearn scaler object"""

        # Mean centered and unit variance
        if norm in ["no", "None"]:
            self.scaler = None
        elif norm not in self.scaler_dict.keys():
            raise ValueError(f"Must select an available scaler type. Available: {list(self.scaler_dict.keys())}")
        else:
            self.scaler = self.scaler_dict[norm](**kwargs)

    def set_model(self, model_type, **kwargs):
        """Sets the model to use in the pipeline.

        Parameters
        ----------
        model_type: type of model to use
        **kwargs: to input to sklearn model object"""

        # Linear Regression
        if model_type not in self.model_dict.keys():
            raise ValueError(f"Must select an available model type. Available: {list(self.model_dict.keys())}")
        else:
            self.model = self.model_dict[model_type](**kwargs)

    def set_pipeline(self):
        """Sets the model to use in the pipeline."""

        pipe = []
        if self.model is None:
            raise ValueError("Must set a valid model before setting pipeline.")
        
        # Scaler and whether it has to be optimized
        if self.scaler is not None:
            pipe.append(("scaler", self.scaler))
        # Feature extension
        if self.feature_extension != 0:
            pipe.append(("feature_extension", preprocessing.PolynomialFeatures(degree=self.feature_extension)))
        # Model
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
        y: 1D-Array with age; shape=n"""

        # Check that pipeline has been properly constructed
        if self.pipeline is None:
            raise TypeError("Must set a valid pipeline before running fit.")

        # Variables of interes
        pred_age = np.zeros(y.shape)
        corrected_age = np.zeros(y.shape)
        metrics_train = []
        metrics_test = []

        # Optimize hyperparameters if required
        if self.hyperparameter_grid != {}:
            print("Running Hyperparameter optimization...")
            opt_pipeline = model_selection.GridSearchCV(self.pipeline, self.hyperparameter_grid, cv=self.CV_split,
                                                        scoring="neg_mean_absolute_error")
            opt_pipeline.fit(X, y)
            print(f"Hyperoptimization best parameters: {opt_pipeline.best_params_}")
            # Set best parameters in pipeline
            self.pipeline.set_params(**opt_pipeline.best_params_)

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
        # Print comparison with mean age as only predictor to have a reference of a dummy regressor
        dummy_rmse = np.sqrt(np.mean((y - np.mean(y)) ** 2))
        dummy_mae = np.mean(np.abs(y - np.mean(y)))
        print("When using mean of ages as predictor for each subject (dummy regressor):\n"
              "MAE: %.2f, RMSE: %.2f" % (dummy_mae, dummy_rmse))
        print("Age range: %.2f" % (np.max(y) - np.min(y)))

        # Fit model on all data
        self.pipeline.fit(X, y)
        y_pred = self.pipeline.predict(X)
        
        self.pipelineFit = True
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
