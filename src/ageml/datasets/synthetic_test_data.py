import os
import importlib.resources as pkg_resources
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ageml import datasets


datasets_path = str(pkg_resources.files(datasets))


def generate_synthetic_features(file_name: str = "synthetic_features.csv",
                                N: int = 100,
                                datasets_path: str = datasets_path) -> None:  # pragma: no cover
    """Generate synthetic features file and save it to a CSV file. For testing purposes and playing around.

    Args:
        file_name (str, optional): File name to save the data to. Defaults to "synthetic_features.csv".
        N (int, optional): Number of subjects. Defaults to 100.
        datasets_path (str, optional): Folder to save at. Defaults to datasets_path.
    """
    # FEATURES #

    # Weights of our synthetic variables
    weights = np.array([[0.77, -0.54, 0.095]])  # (1, 3)
    
    # 100 samples, 3 variables (X1, X2, X3)
    N = 100
    
    # Set random seed for the noise
    np.random.seed(1)
    # Generate the normally distributed random variables
    X1 = np.random.normal(loc=5, scale=2, size=N)
    X2 = np.random.normal(loc=-6, scale=3, size=N)
    X3 = np.random.normal(loc=4, scale=1, size=N)
    X = np.array([X1, X2, X3])
    # Set weights
    weights = np.array([8, 1, 8])
    # Compute Y=weights@X + epsilon
    epsilon = np.random.normal(loc=0, scale=15, size=N)
    Y = weights @ X + epsilon
    # Concatenate
    synth_data = np.concatenate([X.transpose(), Y.reshape([N, 1])], axis=1)
    # Save into dataframe
    df_synth_features = pd.DataFrame(synth_data, columns=['X1', 'X2', 'X3', 'Y'])
    
    # Sanity check in file_name
    if ".csv" not in file_name:
        file_name += ".csv"

    # Save the features
    # TODO: Maybe change to parquet format, faster loading to compensate pandas slowness.
    save_path = os.path.join(datasets_path, file_name)
    df_synth_features.to_csv(save_path, index=True)


def generate_synthetic_covariates(file_name: str = "synthetic_covariates.csv",
                                  N: int = 100,
                                  datasets_path: str = datasets_path) -> None:  # pragma: no cover
    """Generate synthetic covariates file and save it to a CSV file. For testing purposes and playing around.

    Args:
        file_name (str, optional): File name to save the data to. Defaults to "synthetic_covariates.csv".
        N (int, optional): Number of subjects. Defaults to 100.
        datasets_path (str, optional): Folder to save at. Defaults to datasets_path.
    """
    # COVARIATES #
    np.random.seed(1)
    # Generate random data for each column
    biological_gender_values = np.random.choice([0, 1], N)
    education_years_values = np.random.uniform(0, 20, N)
    ethnicity_values = np.random.choice([0, 1], N)

    # Define the mapping for gender and ethnicity
    gender_mapping = {0: "male", 1: "female"}
    ethnicity_mapping = {0: "ethnicity_A", 1: "ethnicity_B"}

    # Create a Pandas DataFrame
    data = {
        "biological_gender": [gender_mapping[val] for val in biological_gender_values],
        "education_years": education_years_values,
        "ethnicity": [ethnicity_mapping[val] for val in ethnicity_values]
    }

    df_covariates = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    save_path = os.path.join(datasets_path, file_name)
    df_covariates.to_csv(save_path, index=True)


def generate_synthetic_factors(file_name: str = "synthetic_factors.csv",
                               N: int = 100,
                               datasets_path: str = datasets_path) -> None:  # pragma: no cover
    """Generate synthetic factors file and save it to a CSV file. For testing purposes and playing around.

    Args:
        file_name (str, optional): File name to save the data to. Defaults to "synthetic_covariates.csv".
        N (int, optional): Number of subjects. Defaults to 100.
        datasets_path (str, optional): Folder to save at. Defaults to datasets_path.
    """
    # TODO: Add this code
    pass


class SyntheticData():
    """
    Class for storing the synthetic features that were generated in the generate_synthetic_data function. To be imported.
    """
    def __init__(self, type: str):
        """Initialization of the class. Loads the data from the CSV file.

        Args:
            type (str): Should be either "features", "covariates" or "factors".
            # TODO: Support "clinical" and "system" data.

        Raises:
            FileNotFoundError: _description_
        """
        datasets_path = str(pkg_resources.files(datasets))
        self.path_to_data = os.path.join(datasets_path, f"synthetic_{type}.csv")
        # Check if file exists. If not, return FileNotFound error
        if not os.path.exists(self.path_to_data):
            raise FileNotFoundError(f"File {self.path_to_data} does not exist. Please generate it first.")
        self._data = self._load_data()

    def _load_data(self):
        # TODO: Maybe change to parquet format, faster loading to compensate pandas slowness.
        df = pd.read_csv(self.path_to_data, index_col=0)
        return df

    def get_data(self):
        return self._data

    def get_np_data(self):
        return self.get_data().to_numpy()
