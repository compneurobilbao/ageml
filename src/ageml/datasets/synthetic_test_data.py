import os
# TODO: Cleaner import
# import importlib.resources as pkg_resources
from dataclasses import dataclass
import pandas as pd
import numpy as np


def generate_synthetic_data(file_name: str):
    """
    Procedure used to generate the synthetic data. Should not be used more than once, it is already
    Args:
        file_name (str): Path to where the file should be saved
    Returns:
    
    """
    # Weights of our synthetic variables
    weights = np.array([[1.32, -5.97, 0.66]])  # (1, 3)
    
    # 100 samples, 3 variables (X1, X2, X3)
    N = 100
    
    # w@X + N(0, 1) = Y
    # Set random seed for the noise
    np.random.seed(1)
    # Generate the data
    X1 = np.random.uniform(200, 1000, N)
    X2 = np.random.uniform(0.1, 20, N)
    X3 = np.random.uniform(-400, 30, N)
    noise = np.random.rand(N) - 0.5
    X = np.array([X1, X2, X3])
    Y = (weights @ X) + noise
    # Concatenate
    synth_data = np.concatenate([X.transpose(), Y.transpose()], axis=1)
    
    # Import pandas only for this function, otherwise it slows down importing the datasets.
    df_synth_data = pd.DataFrame(synth_data, columns=['X1', 'X2', 'X3', 'Y'])
    
    # Parse file_name
    if ".csv" not in file_name:
        file_name += ".csv"

    # Save data
    # TODO: Maybe change to parquet format, faster loading to compensate pandas slowness.
    df_synth_data.to_csv(file_name, index=False)


@dataclass
class SyntheticDataset:
    """
    Class for storing the data that was generated in the generate_synthetic_data function. To be imported.
    """
    def __init__(self):
        # TODO: Maybe change to parquet format, faster loading to compensate pandas slowness.
        self.path_to_data = os.path.join(os.path.dirname(__file__), "synthetic_dataset.csv")
        self._data = self._load_data()

    def _load_data(self):
        df = pd.read_csv(self.path_to_data, index_col=False)
        df.drop(columns="Unnamed: 0", inplace=True)
        return df

    def get_data(self):
        return self._data

    def get_np_data(self):
        return self.get_data().to_numpy()
