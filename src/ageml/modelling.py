"""Implement the modelling software.

Used in the AgeML project to enable the modelling of age.

Classes:
--------
AgeML - able to fit age models and predict age.
Classifier - classifier of class labels based on deltas.
"""

import copy

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
from hpsklearn import HyperoptEstimator, any_regressor, any_preprocessing
from hyperopt import tpe


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
    scaler_hyperparameters = {
        "maxabs": {},
        "minmax": {},
        "normalizer": {},
        "power": {"method": ["yeo-johnson", "box-cox"]},
        "quantile": {
            "n_quantiles": [10, 1000],
            "output_distribution": ["normal", "uniform"],
        },
        "robust": {},
        "standard": {},
    }

    # Model dictionary
    model_dict = {
        "linear_reg": linear_model.LinearRegression,
        "ridge": linear_model.Ridge,
        "lasso": linear_model.Lasso,
        "linear_svr": svm.SVR,
        "xgboost": XGBRegressor,  # XGBoost
        "rf": RandomForestRegressor,
        "hyperopt": HyperoptEstimator,
    }
    model_hyperparameter_ranges = {
        "ridge": {"alpha": [-3, 3]},
        "lasso": {"alpha": [-3, 3]},
        "linear_svr": {"C": [-3, 3], "epsilon": [-3, 3]},
        "xgboost": {
            "eta": [-3, 3],
            "gamma": [-3, 3],
            "max_depth": [0, 100],
            "min_child_weight": [0, 100],
            "max_delta_step": [0, 100],
            "subsample": [-3, 3],
            "colsample_bytree": [0.001, 1],
            "colsample_bylevel": [0.001, 1],
            "colsample_bynode": [0.001, 1],
            "lambda": [-3, 3],
            "alpha": [-3, 3],
        },
        "rf": {
            "n_estimators": [1, 100],
            "max_depth": [1, 100],
            "min_samples_split": [1, 100],
            "min_samples_leaf": [1, 100],
            "max_features": [1, 100],
            "min_impurity_decrease": [0, 1],
            "max_leaf_nodes": [1, 100],
            "min_weight_fraction_leaf": [-3, 3],
        },
    }

    model_hyperparameter_types = {
        "ridge": {"alpha": "log"},
        "lasso": {"alpha": "log"},
        "linear_svr": {"C": "log", "epsilon": "log"},
        "xgboost": {
            "eta": "float",
            "gamma": "float",
            "max_depth": "int",
            "min_child_weight": "int",
            "max_delta_step": "int",
            "subsample": "float",
            "colsample_bytree": "float",
            "colsample_bylevel": "float",
            "colsample_bynode": "float",
            "lambda": "log",
            "alpha": "log",
        },
        "rf": {
            "n_estimators": "int",
            "max_depth": "int",
            "min_samples_split": "int",
            "min_samples_leaf": "int",
            "max_features": "int",
            "min_impurity_decrease": "log",
            "max_leaf_nodes": "int",
        },
    }

    def __init__(
        self,
        scaler_type,
        scaler_params,
        model_type,
        model_params,
        CV_split,
        seed,
        hyperparameter_tuning: int = 0,
        hyperparameter_params: dict = {},
        feature_extension: int = 0,
    ):
        """Initialise variables."""

        # Scaler dictionary
        self.scaler_type = scaler_type
        self.scaler_dict = AgeML.scaler_dict

        # Model dictionary
        self.model_type = model_type
        self.model_dict = AgeML.model_dict
        # Hyperparameters and feature extension
        self.hyperparameter_tuning = hyperparameter_tuning
        self.hyperparameter_params = hyperparameter_params
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
        """Build the hyperparameter grid of the selected model upon AgeML object initialization

        Returns:
            dict: dictionary with the hyperparameter grid
        """
        param_grid = {}
        conditions = [
            self.model_type in AgeML.model_dict.keys(),
            self.hyperparameter_tuning > 0,
            self.model_type != "hyperopt",
            self.model_type != "linear_reg",
        ]
        if all(conditions):
            hyperparam_types = AgeML.model_hyperparameter_types[self.model_type]
            invalid_hyperparams = [param for param in self.hyperparameter_params.keys() if param not in hyperparam_types.keys()]
            if len(invalid_hyperparams) > 0:
                raise ValueError(f"Hyperparameter(s) {invalid_hyperparams} not available for the selected model '{self.model_type}'.")

            # Initialize output grid
            for hyperparam_name in list(self.hyperparameter_params.keys()):
                low, high = self.hyperparameter_params[hyperparam_name]
                if hyperparam_types[hyperparam_name] == "log":
                    param_grid[f"model__{hyperparam_name}"] = np.logspace(low, high, int(self.hyperparameter_tuning))
                elif hyperparam_types[hyperparam_name] == "int":
                    param_grid[f"model__{hyperparam_name}"] = np.rint(np.linspace(low, high, int(self.hyperparameter_tuning))).astype(int)
                elif hyperparam_types[hyperparam_name] == "float":
                    param_grid[f"model__{hyperparam_name}"] = np.linspace(low, high, int(self.hyperparameter_tuning))
        self.hyperparameter_grid = param_grid

    def set_scaler(self, norm, **kwargs):
        """Sets the scaler to use in the pipeline.

        Parameters
        ----------
        norm: type of scaler to use
        **kwargs: to input to sklearn scaler object"""

        # Mean centered and unit variance
        if norm in ["no", "None"] or self.model_type == "hyperopt":
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
        elif model_type == "hyperopt":
            self.model = HyperoptEstimator(
                regressor=any_regressor("age_regressor"),
                preprocessing=any_preprocessing("age_preprocessing"),
                algo=tpe.suggest,
            )
        else:
            self.model = self.model_dict[model_type](**kwargs)

        self.model_type = model_type

    def set_pipeline(self):
        """Sets the model to use in the pipeline."""

        pipe = []
        if self.model is None:
            raise ValueError("Must set a valid model before setting pipeline.")

        # Scaler and whether it has to be optimized
        if self.scaler is not None and self.model_type != "hyperopt":
            pipe.append(("scaler", self.scaler))
        # Feature extension
        if self.feature_extension != 0 and self.model_type != "hyperopt":
            pipe.append(("feature_extension", preprocessing.PolynomialFeatures(degree=self.feature_extension)))
        # Model
        if self.model_type != "hyperopt":
            pipe.append(("model", self.model))
            self.pipeline = pipeline.Pipeline(pipe)
        else:
            self.pipeline = None

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

        mae = metrics.mean_absolute_error(y_true, y_pred)
        rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
        r2 = metrics.r2_score(y_true, y_pred)
        p, _ = stats.pearsonr(y_true, y_pred)
        return mae, rmse, r2, p

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
        if self.pipeline is None and self.model_type != "hyperopt":
            raise TypeError("Must set a valid pipeline before running fit.")

        if self.model_type != "hyperopt":
            # Optimize hyperparameters if required
            if self.hyperparameter_tuning > 1:
                print("Running Hyperparameter optimization...")
            else:
                print("No hyperparameter optimization will be done.")
            # Generate the hyperparameter grid. If {}, default values are used
            hyperparameter_grid = model_selection.ParameterGrid(self.hyperparameter_grid)
            pipelines = []
            # Generate pipelines with different hyperparameters from the grid
            for grid_point in hyperparameter_grid:
                original_pipeline = copy.deepcopy(self.pipeline)
                point_pipeline = original_pipeline.set_params(**grid_point)
                pipelines.append(point_pipeline)
            # Variables of interest
            pred_age = np.zeros(y.shape[0])
            corrected_age = np.zeros(y.shape[0])
            best_split_mae = 1e10
            mae_means_test = []
            # Loop through the pipelines, and then loop through the CV splits
            for cv_pipeline in pipelines:
                print(f"\nRunning CV splits with pipeline:\n{cv_pipeline}")
                temp_pred_age = np.zeros(y.shape[0])
                temp_corr_age = np.zeros(y.shape[0])
                split_metrics_train = []
                split_metrics_test = []
                kf_hyperopt = model_selection.KFold(n_splits=self.CV_split, random_state=self.seed, shuffle=True)
                for i, (train, test) in enumerate(kf_hyperopt.split(X)):
                    X_train, X_test = X[train], X[test]
                    y_train, y_test = y[train], y[test]

                    # Train model with the hyperparameters
                    cv_pipeline.fit(X_train, y_train)

                    # Predictions on train and test set
                    y_pred_train = cv_pipeline.predict(X_train)
                    y_pred_test = cv_pipeline.predict(X_test)

                    # Metrics in Train and Test sets
                    split_metrics_train.append(self.calculate_metrics(y_train, y_pred_train))
                    split_metrics_test.append(self.calculate_metrics(y_test, y_pred_test))

                    # Print metrics
                    print("\nFold N: %d" % (i + 1))
                    print("Train: MAE %.2f, RMSE %.2f, R2 %.3f, p %.3f" % split_metrics_train[i])
                    print("Test: MAE %.2f, RMSE %.2f, R2 %.3f, p %.3f" % split_metrics_test[i])

                    # Fit and apply age-bias correction
                    self.fit_age_bias(y_train, y_pred_train)
                    y_pred_test_no_bias = self.predict_age_bias(y_test, y_pred_test)

                    # Save results of hold out
                    temp_pred_age[test] = y_pred_test
                    temp_corr_age[test] = y_pred_test_no_bias

                # Compute the mean of scores over all CV splits
                mean_score_test = np.mean(split_metrics_test, axis=0)[0]  # Mean of MAE
                mae_means_test.append(mean_score_test)

                # If the mean MAE is better than the previous best, save the results
                if mean_score_test < best_split_mae:
                    best_split_mae = mean_score_test
                    pred_age = copy.deepcopy(temp_pred_age)
                    corrected_age = copy.deepcopy(temp_corr_age)

                # CV Summary of the selected hyperparameters
                print("\nSummary metrics over all CV splits:")
                summary_train = self.summary_metrics(split_metrics_train)
                print("Train: MAE %.2f ± %.2f, RMSE %.2f ± %.2f, R2 %.3f ± %.3f, p %.3f ± %.3f" % tuple(summary_train))
                summary_test = self.summary_metrics(split_metrics_test)
                print("Test: MAE %.2f ± %.2f, RMSE %.2f ± %.2f, R2 %.3f ± %.3f, p %.3f ± %.3f" % tuple(summary_test))

            # Select the best pipeline based on the mean scores
            best_hyperparam_index = np.argmin(mae_means_test)
            self.pipeline = pipelines[best_hyperparam_index]

            if self.hyperparameter_tuning > 0:
                print(f"\nHyperoptimization best parameters: {hyperparameter_grid[best_hyperparam_index]}")
                print(f"Best pipeline:\n{self.pipeline}")

        elif self.model_type == "hyperopt":
            print("Running Hyperparameter optimization with 'hyperopt' model option...")

            # Variables of interest
            pred_age = np.zeros(y.shape)
            corrected_age = np.zeros(y.shape)

            # Get the data split
            X_train = X[self.train_indices,]
            y_train = y[self.train_indices]
            X_test = X[self.test_indices,]
            y_test = y[self.test_indices]

            # Fit the hyperopt model and compute loss
            self.model.fit(X_train, y_train)
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            mae_train, rmse_train, r2_train, p_train = self.calculate_metrics(y_train, y_pred_train)
            mae_test, rmse_test, r2_test, p_test = self.calculate_metrics(y_test, y_pred_test)
            best_model = self.model.best_model()["learner"]
            best_preprocessing = self.model.best_model()["preprocs"]
            # Evaluate on test set
            pipe = [(f"preproc_{i}", preproc) for i, preproc in enumerate(best_preprocessing)]
            pipe.append(("model", best_model))

            self.pipeline = pipeline.Pipeline(pipe)
            print(f"Train: MAE {mae_train:.2f}, RMSE {rmse_train:.2f}, R2 {r2_train:.3f}, p {p_train:.3f}")
            print(f"Test: MAE {mae_test:.2f}, RMSE {rmse_test:.2f}, R2 {r2_test:.3f}, p {p_test:.3f}")
            print(
                "Hyperoptimization best parameters:\n"
                f"\t- Best preprocessing:\n\t\t{best_preprocessing}\n"
                f"\t- Best model:\n\t\t{best_model}"
            )

            # Fit and apply age-bias correction
            self.fit_age_bias(y_train, y_pred_train)
            y_pred_test_no_bias = self.predict_age_bias(y_test, y_pred_test)
            # Save results of hold out
            pred_age[self.test_indices] = y_pred_test
            corrected_age[self.test_indices] = y_pred_test_no_bias

        # Print comparison with mean age as only predictor to have a reference of a dummy regressor
        dummy_rmse = np.sqrt(np.mean((y - np.mean(y)) ** 2))
        dummy_mae = np.mean(np.abs(y - np.mean(y)))
        print(
            "\nWhen using mean of ages as predictor for each subject (dummy regressor):\n" "MAE: %.2f, RMSE: %.2f" % (dummy_mae, dummy_rmse)
        )
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

    Parameters
    -----------
    CV_split: integer number of CV splits
    seed: integer seed for randomization
    thr: threshold value
    ci_val: confidence interval value

    Public methods:
    ---------------
    set_model(self): Sets the model to use in the pipeline.

    set_CV_params(self, CV_split, seed): Set the parameters of the Cross Validation scheme.

    set_threshold(self, thr): Set the threshold for classification.

    set_ci(self, ci_val): Set the confidence interval for classification.

    fit_model(self, X, y): Fit the model.

    predict(self, X): Predict class labels with fitted model.
    """

    def __init__(self, CV_split: int = 5, seed=None, thr: float = 0.5, ci_val: float = 0.95):
        """Initialise variables."""

        # Set required modelling parts
        self.set_model()

        # Set CV parameters
        self.set_CV_params(CV_split=CV_split, seed=seed)

        # Set threshold and confidence interval
        self.set_threshold(thr)
        self.set_ci(ci_val)

        # Initialise flags
        self.modelFit = False

    def set_model(self):
        """Sets the model to use in the pipeline."""

        self.model = linear_model.LogisticRegression()

    def set_CV_params(self, CV_split, seed=None):
        """Set the parameters of the Cross Validation Scheme.

        Parameters
        ----------
        CV_split: number of splits in CV scheme
        seed: seed to set random state."""

        self.CV_split = CV_split
        self.seed = seed

    def set_threshold(self, thr):
        """Set the threshold for classification.

        Parameters
        ----------
        thr: threshold value"""

        self.thr = thr

    def set_ci(self, ci_val):
        """Set the confidence interval for classification.

        Parameters
        ----------
        ci_val: confidence interval value"""

        self.ci_val = ci_val

    def fit_model(self, X, y):
        """Fit the model.

        Parameters
        ----------
        X: 2D-Array with features; shape=(n,m)
        y: 1D-Array with labbels; shape=n"""

        # Arrays to store  values
        accs, aucs, spes, sens = [], [], [], []
        y = y.ravel()
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
            sensitivity = tp / (tp + fn)
            accs.append(acc)
            sens.append(sensitivity)
            spes.append(specificity)

        # Compute confidence intervals
        ci_accs = st.t.interval(confidence=self.ci_val, df=len(accs) - 1, loc=np.mean(accs), scale=st.sem(accs))
        ci_aucs = st.t.interval(confidence=self.ci_val, df=len(aucs) - 1, loc=np.mean(aucs), scale=st.sem(aucs))
        ci_sens = st.t.interval(confidence=self.ci_val, df=len(sens) - 1, loc=np.mean(sens), scale=st.sem(sens))
        ci_spes = st.t.interval(confidence=self.ci_val, df=len(spes) - 1, loc=np.mean(spes), scale=st.sem(spes))

        # Print results
        print("Summary metrics over all CV splits (%s CI)" % (self.ci_val))
        print("AUC: %.3f [%.3f-%.3f]" % (np.mean(aucs), ci_aucs[0], ci_aucs[1]))
        print("Accuracy: %.3f [%.3f-%.3f]" % (np.mean(accs), ci_accs[0], ci_accs[1]))
        print("Sensitivity: %.3f [%.3f-%.3f]" % (np.mean(sens), ci_sens[0], ci_sens[1]))
        print("Specificity: %.3f [%.3f-%.3f]" % (np.mean(spes), ci_spes[0], ci_spes[1]))

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
