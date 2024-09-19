import os
from typing import List
import importlib.resources as pkg_resources

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from ageml import datasets

datasets_path = str(pkg_resources.files(datasets))

# RNG for reproducibility
seed = 107146869338163146621163044826586732901
rng = np.random.default_rng(seed)


def generate_correlated_data(Y, n_samples: int, n_features: int, correlation_levels: List[float]):
    """
    Generate synthetic data with features X1, X2, ..., Xp that have different
    levels of correlation with a continuous target variable Y (age).

    Parameters:
    - n_samples: Number of samples to generate
    - n_features: Number of features to generate
    - correlation_levels: A list of length n_features specifying correlation
      levels of each feature with Y (values between 0 and 1)

    Returns:
    - DataFrame with synthetic data (features and target)
    """

    # Step 1: Generate the target variable Y (continuous, e.g., age)
    Y_std = np.std(Y)
    Y_mean = np.mean(Y)

    # Step 2: Generate correlated features X1, X2, ..., Xp
    X = np.zeros((n_samples, n_features))

    for i in range(n_features):
        correlation = correlation_levels[i]

        # Generate noise, independent of Y_std
        noise = rng.normal(0, 1, n_samples)

        mean_x_i = rng.uniform(-50, 50, 1)
        std_x_i = rng.uniform(1, 10, 1)
        # Create feature as a combination of Y_std and noise
        X[:, i] = mean_x_i + std_x_i * (correlation * (Y - Y_mean) / Y_std + np.sqrt(1 - correlation**2) * noise)

    # Combine the features and target into a DataFrame
    data = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(n_features)])
    data["Y"] = Y  # Add the target column (Y)

    return data


def generate_synthetic_data(save: bool = False, output_dir: str = None):
    ###################
    # GENERATE CONTROLS
    n_samples = 600  # Number of samples
    n_features = 12  # Number of features
    low_lim = -0.6
    high_lim = 0.5
    correlation_levels = np.arange(low_lim, high_lim, (high_lim - low_lim) / n_features)  # Correlation levels between -0.9 and 0.9

    # Generate ages
    Y = rng.uniform(50, 80, n_samples)  # Example: age between 50 and 80

    # Generate the synthetic data
    synthetic_controls = generate_correlated_data(Y, n_samples, n_features, correlation_levels)

    ###############################
    # TRAIN A LINEAR REGRESSION MODEL
    # Predict age (Y) with linear regressor using X1, X2, ..., X12

    # Split the data into training and test sets
    X = synthetic_controls.drop(columns=["Y"]).to_numpy()
    Y = synthetic_controls["Y"].to_numpy()

    # Train a linear regression model (overfit it)
    model = LinearRegression()
    model.fit(X, Y)

    # Predict the target variable on the test set
    Y_pred = model.predict(X)

    ###############################
    # GENERATE POSITIVE DELTA GROUP
    # Sample 200 random samples from the synthetic data
    n_samples = 200
    random_indices = rng.integers(0, synthetic_controls.shape[0], n_samples)
    random_sample = synthetic_controls.iloc[random_indices]
    random_sample_age = random_sample["Y"]
    random_sample_vars = random_sample.drop(columns=["Y"])
    sample_age_copy = random_sample_age.copy().to_numpy()

    sample_age_copy += 6 * rng.normal(1, 0.5, n_samples)

    # Predict the age Y on this group
    Y_pred_pos_group = model.predict(random_sample_vars.to_numpy())

    # Concatenate features and target (age)
    positive_delta_group = np.concatenate([random_sample_vars.to_numpy(), sample_age_copy[:, np.newaxis]], axis=1)
    positive_delta_group = pd.DataFrame(positive_delta_group, columns=synthetic_controls.columns)

    ###############################
    # GENERATE NEGATIVE DELTA GROUP
    # Sample 200 random samples from the synthetic data
    n_samples = 200
    random_indices = rng.integers(0, synthetic_controls.shape[0], n_samples)
    random_sample = synthetic_controls.iloc[random_indices]
    random_sample_age = random_sample["Y"]
    random_sample_vars = random_sample.drop(columns=["Y"])
    sample_age_copy = random_sample_age.copy().to_numpy()

    sample_age_copy += -8 * rng.normal(1, 0.5, n_samples)

    # Predict the age Y on this group
    Y_pred_neg_group = model.predict(random_sample_vars.to_numpy())

    # Concatenate features and target (age)
    negative_delta_group = np.concatenate([random_sample_vars.to_numpy(), sample_age_copy[:, np.newaxis]], axis=1)
    negative_delta_group = pd.DataFrame(negative_delta_group, columns=synthetic_controls.columns)

    ####################
    # All synthetic data
    all_synthetic_data = pd.concat([synthetic_controls, positive_delta_group, negative_delta_group])

    ###############
    # CLINICAL DATA
    clinical_data = pd.DataFrame(columns=["CN", "G1", "G2"])

    # CN group is 600 True and 400 false
    clinical_data["CN"] = [True] * 600 + [False] * 400
    # G1 group is 600 False, 200 True and 200 False
    clinical_data["G1"] = [False] * 600 + [True] * 200 + [False] * 200
    # G2 group is 800 False and 200 True
    clinical_data["G2"] = [False] * 800 + [True] * 200

    ##############
    # FACTORS DATA
    # Compute age deltas
    Y_pred = model.predict(synthetic_controls.drop(columns=["Y"]))
    age_delta_cn = Y_pred - synthetic_controls["Y"]

    Y_pred_pos_group = model.predict(positive_delta_group.drop(columns=["Y"]))
    age_delta_pos = Y_pred_pos_group - positive_delta_group["Y"]

    Y_pred_neg_group = model.predict(negative_delta_group.drop(columns=["Y"]))
    age_delta_neg = Y_pred_neg_group - negative_delta_group["Y"]

    age_deltas = np.concatenate([age_delta_cn, age_delta_pos, age_delta_neg])

    # Create a dataframe with MOCAScore, of Heavy Drinking Score and Physical Activity Score
    factors_data = pd.DataFrame(columns=["MOCAScore", "HeavyDrinkingScore", "PhysicalActivityScore"])

    # Normalize the deltas
    standard_deltas = (age_deltas - np.mean(age_deltas)) / np.std(age_deltas)

    # MOCAScore -> The more higher MOCAScore, the higher delta. Correlation: 0.7
    corr = 0.6
    factors_data["MOCAScore"] = corr * standard_deltas + np.sqrt(1 - corr**2) * rng.normal(0, 1, len(standard_deltas))

    # HeavyDrinking -> The more you drank, the higher delta. Correlation: 0.5
    corr = 0.3
    factors_data["HeavyDrinkingScore"] = corr * standard_deltas + np.sqrt(1 - corr**2) * rng.normal(0, 1, len(standard_deltas))

    # PhysicalActivity -> The more you did physical activity, the lower delta. Correlation: -0.4
    corr = -0.4
    factors_data["PhysicalActivityScore"] = corr * standard_deltas + np.sqrt(1 - corr**2) * rng.normal(0, 1, len(standard_deltas))

    #################
    # COVARIABLE DATA (Sex, YoE)
    # Generate a sex covariate completely randomly, with no correlation with Y
    covar_data = pd.DataFrame(columns=["Sex", "YoE"])
    covar_data["Sex"] = rng.integers(0, 2, all_synthetic_data.shape[0])

    # YoE has to be correlated with age
    mean_yoe = 10
    std_yoe = 5
    corr = 0.2

    noise = rng.normal(0, 1, all_synthetic_data.shape[0])

    yoe = mean_yoe + std_yoe * (corr * standard_deltas + np.sqrt(1 - corr**2) * noise)

    covar_data["YoE"] = yoe

    ##############
    # SYSTEMS DATA
    system_A = "system_A:X1,X2,X3,X4"
    system_B = "system_B:X5,X6,X7,X8,X9,X10,X11,X12"

    systems_data = system_A + "\n" + system_B

    # Reset the index of the dataframes
    all_synthetic_data.rename(columns={"Y": "age"}, inplace=True)
    all_synthetic_data.reset_index(drop=True, inplace=True)
    clinical_data.reset_index(drop=True, inplace=True)
    factors_data.reset_index(drop=True, inplace=True)
    covar_data.reset_index(drop=True, inplace=True)

    if save:
        # SAVE THE DATA
        if output_dir is None:
            save_path = os.path.abspath(os.path.dirname(__file__))
        else:
            output_dir = os.path.abspath(os.path.expanduser(output_dir))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_path = output_dir

        features_path = os.path.join(save_path, "toy_features.csv")
        clinical_path = os.path.join(save_path, "toy_clinical.csv")
        factors_path = os.path.join(save_path, "toy_factors.csv")
        covar_data_path = os.path.join(save_path, "toy_covar.csv")
        systems_path = os.path.join(save_path, "toy_systems.txt")

        all_synthetic_data.to_csv(features_path, index=True, header=True)
        clinical_data.to_csv(clinical_path, index=True, header=True)
        factors_data.to_csv(factors_path, index=True, header=True)
        covar_data.to_csv(covar_data_path, index=True, header=True)

        # Write systems data to a .txt file exactly as it is
        with open(systems_path, "w") as f:
            f.write(systems_data)


######################
# Synthetic Data Class
class SyntheticData:
    """
    Class for storing the synthetic data that was generated in the generate_synthetic_data function
    """

    def __init__(self):
        """Load the data from the CSV files packaged with the library.

        Raises:
            FileNotFoundError: _description_
        """
        datasets_path = str(pkg_resources.files(datasets))
        self.data_paths = {
            "features": os.path.join(datasets_path, "toy_features.csv"),
            "clinical": os.path.join(datasets_path, "toy_clinical.csv"),
            "covariates": os.path.join(datasets_path, "toy_covar.csv"),
            "factors": os.path.join(datasets_path, "toy_factors.csv"),
            "systems": os.path.join(datasets_path, "toy_systems.txt"),
        }

        # Check if file exists. If not, return FileNotFound error
        if not all([os.path.exists(path) for path in self.data_paths.values()]):
            raise FileNotFoundError("Missing toy data. Please generate it.")

        # Load the data
        self._load_data()

    def _load_data(self):
        self._data = {}
        for data_type, data_path in self.data_paths.items():
            if data_type == "systems":
                with open(data_path, "r") as f:
                    self._data[data_type] = f.read()
            else:
                # TODO: Maybe change to a faster format (parquet, feather, etc.), faster loading to compensate pandas import slowness.
                self._data[data_type] = pd.read_csv(data_path, index_col=0)

    def __getattr__(self, name):
        if name in self._data.keys():
            return self._data[name]
