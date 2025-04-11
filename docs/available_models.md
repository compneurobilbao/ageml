# Available Models in ageml

This document contains a list of all the models available in the _ageml_ package, with a list of the hyperparameters that are optimized in _ageml_ (if specified to do so) and a link to their corresponding documentation. Hyperparameter optimization is done respect to the Mean Absolute Error (MAE). The majority of them come from the [scikit-learn](https://scikit-learn.org/stable/) package.

## Model List

- [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) (`linear_reg` in `ageml`)
  - __Hyperparameters__: None</br></br>
- [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) (`ridge` in `ageml`)
  - __Hyperparameters__: `alpha`</br></br>
- [Lasso Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) (`lasso` in `ageml`)
  - __Hyperparameters__: `alpha`</br></br>
- [XGBoost Regression](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) (`xgboost` in `ageml`)
  - __Hyperparameters__: `eta`, `gamma`, `max_depth`, `min_child_weight`, `max_delta_step`, `subsample`, `colsample_bytree`, `colsample_bylevel`, `colsample_bynode`, `lambda`, `alpha`</br></br>
- [Epsilon-Support Vector Regression](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR) (`linear_svr` in `ageml`)
  - __Hyperparameters__: `C`, `epsilon`</br></br>
- [Random Forest Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) (`rf` in `ageml`)
  - __Hyperparameters__: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `min_impurity_decrease`, `max_leaf_nodes`, `min_weight_fraction_leaf`


### Model Hyperparameters

When specifying model hyperparameter ranges, be aware that a log operation can be applied to some of them, aligned to the `scikit-learn` API. See the table below for reference of the hyperparameters that have this constraint explicitly implemented in `ageml` (or alternatively, import and print the `AgeML.model_hyperparameter_types` dictionary):
<table>
  <tr>
    <th>Model</th>
    <th>Parameter</th>
    <th>Type</th>
  </tr>
  <tr>
    <td rowspan="1">Lasso Regression</td>
    <td>alpha</td>
    <td>log</td>
  </tr>
  <tr>
    <td rowspan="2">Epsilon-Support Vector Regression</td>
    <td>C</td>
    <td>log</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>log</td>
  </tr>
  <tr>
    <td rowspan="11">XGBoost Regression</td>
    <td>eta</td>
    <td>float</td>
  </tr>
  <tr>
    <td>gamma</td>
    <td>float</td>
  </tr>
  <tr>
    <td>max_depth</td>
    <td>int</td>
  </tr>
  <tr>
    <td>min_child_weight</td>
    <td>int</td>
  </tr>
  <tr>
    <td>max_delta_step</td>
    <td>int</td>
  </tr>
  <tr>
    <td>subsample</td>
    <td>float</td>
  </tr>
  <tr>
    <td>colsample_bytree</td>
    <td>float</td>
  </tr>
  <tr>
    <td>colsample_bylevel</td>
    <td>float</td>
  </tr>
  <tr>
    <td>colsample_bynode</td>
    <td>float</td>
  </tr>
  <tr>
    <td>lambda</td>
    <td>log</td>
  </tr>
  <tr>
    <td>alpha</td>
    <td>log</td>
  </tr>
  <tr>
    <td rowspan="9">Random Forest Regression</td>
    <td>n_estimators</td>
    <td>int</td>
  </tr>
  <tr>
    <td>max_depth</td>
    <td>int</td>
  </tr>
  <tr>
    <td>min_samples_split</td>
    <td>int</td>
  </tr>
  <tr>
    <td>min_samples_leaf</td>
    <td>int</td>
  </tr>
  <tr>
    <td>max_features</td>
    <td>int</td>
  </tr>
  <tr>
    <td>min_impurity_decrease</td>
    <td>log</td>
  </tr>
  <tr>
    <td>max_leaf_nodes</td>
    <td>int</td>
  </tr>
</table>