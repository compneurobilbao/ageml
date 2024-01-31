# Available Models in ageml

This document contains a list of all the models available in the _ageml_ package, with a list of the hyperparameters that are optimized in _ageml_ (if specified to do so) and a link to their corresponding documentation. The majority of them come from the [scikit-learn](https://scikit-learn.org/stable/) package.

## Model List

- [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) (`linear_reg` in ageml)
  - __Hyperparameters__: None</br></br>
- [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) (`ridge` in ageml)
  - __Hyperparameters__: `alpha`</br></br>
- [Lasso Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) (`lasso` in ageml)
  - __Hyperparameters__: `alpha`</br></br>
- [XGBoost Regression](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) (`xgboost` in ageml)
  - __Hyperparameters__: `eta`, `gamma`, `max_depth`, `min_child_weight`, `max_delta_step`, `subsample`, `colsample_bytree`, `colsample_bylevel`, `colsample_bynode`, `lambda`, `alpha`</br></br>
- [Epsilon-Support Vector Regression](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR) (`linear_svr` in ageml)
  - __Hyperparameters__: `C`, `epsilon`</br></br>
- [Random Forest Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) (`rf` in ageml)
  - __Hyperparameters__: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `min_impurity_decrease`, `max_leaf_nodes`, `min_weight_fraction_leaf`
