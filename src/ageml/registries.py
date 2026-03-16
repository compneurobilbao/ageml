"""Central registries for AgeML models, scalers and metrics."""

import abc

import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from xgboost import XGBRegressor


class BaseModel(abc.ABC):
    """Abstract base class for all models in AgeML."""

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit the model."""
        pass

    @abc.abstractmethod
    def predict(self, X):
        """Predict using the fitted model."""
        pass


class ModelRegistry:
    """Registry for models that can be used in AgeML."""

    _registry = {}

    @classmethod
    def register(cls, name, model_class, hyperparameter_ranges=None, hyperparameter_types=None):
        """Register a model class with optional hyperparameter info."""
        cls._registry[name] = {
            "class": model_class,
            "hyperparameter_ranges": hyperparameter_ranges or {},
            "hyperparameter_types": hyperparameter_types or {},
        }

    @classmethod
    def get(cls, name):
        """Get a registered model class."""
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' not registered. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_models(cls):
        """List all registered model names."""
        return list(cls._registry.keys())


class ScalerRegistry:
    """Registry for scalers that can be used in AgeML."""

    _registry = {}

    @classmethod
    def register(cls, name, scaler_class, hyperparameters=None):
        """Register a scaler class with optional hyperparameters."""
        cls._registry[name] = {
            "class": scaler_class,
            "hyperparameters": hyperparameters or {},
        }

    @classmethod
    def get(cls, name):
        """Get a registered scaler class."""
        if name not in cls._registry:
            raise ValueError(f"Scaler '{name}' not registered. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_scalers(cls):
        """List all registered scaler names."""
        return list(cls._registry.keys())


class MetricRegistry:
    """Registry for metrics calculators."""

    _registry = {}

    @classmethod
    def register(cls, name, calculator_func):
        """Register a metric calculator function."""
        cls._registry[name] = calculator_func

    @classmethod
    def get(cls, name):
        """Get a registered metric calculator."""
        if name not in cls._registry:
            raise ValueError(f"Metric '{name}' not registered. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_metrics(cls):
        """List all registered metric names."""
        return list(cls._registry.keys())


def register_default_registries():
    """Register default models, scalers and metrics once."""
    if ModelRegistry._registry and ScalerRegistry._registry and MetricRegistry._registry:
        return

    ModelRegistry.register("linear_reg", linear_model.LinearRegression)
    ModelRegistry.register(
        "ridge",
        linear_model.Ridge,
        hyperparameter_ranges={"alpha": [-3, 3]},
        hyperparameter_types={"alpha": "log"},
    )
    ModelRegistry.register(
        "lasso",
        linear_model.Lasso,
        hyperparameter_ranges={"alpha": [-3, 3]},
        hyperparameter_types={"alpha": "log"},
    )
    ModelRegistry.register(
        "linear_svr",
        svm.SVR,
        hyperparameter_ranges={"C": [-3, 3], "epsilon": [-3, 3]},
        hyperparameter_types={"C": "log", "epsilon": "log"},
    )
    ModelRegistry.register(
        "xgboost",
        XGBRegressor,
        hyperparameter_ranges={
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
        hyperparameter_types={
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
    )
    ModelRegistry.register(
        "rf",
        RandomForestRegressor,
        hyperparameter_ranges={
            "n_estimators": [1, 100],
            "max_depth": [1, 100],
            "min_samples_split": [1, 100],
            "min_samples_leaf": [1, 100],
            "max_features": [1, 100],
            "min_impurity_decrease": [0, 1],
            "max_leaf_nodes": [1, 100],
            "min_weight_fraction_leaf": [-3, 3],
        },
        hyperparameter_types={
            "n_estimators": "int",
            "max_depth": "int",
            "min_samples_split": "int",
            "min_samples_leaf": "int",
            "max_features": "int",
            "min_impurity_decrease": "log",
            "max_leaf_nodes": "int",
        },
    )

    ScalerRegistry.register("maxabs", MaxAbsScaler)
    ScalerRegistry.register("minmax", MinMaxScaler)
    ScalerRegistry.register("normalizer", Normalizer)
    ScalerRegistry.register("power", PowerTransformer, hyperparameters={"method": ["yeo-johnson", "box-cox"]})
    ScalerRegistry.register(
        "quantile",
        QuantileTransformer,
        hyperparameters={"n_quantiles": [10, 1000], "output_distribution": ["normal", "uniform"]},
    )
    ScalerRegistry.register("robust", RobustScaler)
    ScalerRegistry.register("standard", StandardScaler)

    MetricRegistry.register(
        "regression_metrics",
        lambda y_true, y_pred: (
            metrics.mean_absolute_error(y_true, y_pred),
            np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
            metrics.r2_score(y_true, y_pred),
            stats.pearsonr(y_true, y_pred)[0],
        ),
    )


register_default_registries()
