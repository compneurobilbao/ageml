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
    weights = np.array([[0.77, -0.54, 0.095]])  # (1, 3)
    
    # 100 samples, 3 variables (X1, X2, X3)
    N = 100
    
    # w@X + N(0, 1) = Y
    # Y
    # Set random seed for the noise
    np.random.seed(1)
    # Generate the normally distributed random variables
    Y = np.random.normal(loc=72, scale=10, size=[N, 1])
    X1 = np.divide(Y[:, 0], weights[0][0])
    X1 += np.random.normal(loc=0, scale=X1.std(), size=N)
    X2 = np.divide(Y[:, 0], weights[0][1])
    X2 += np.random.normal(loc=0, scale=X2.std(), size=N)
    X3 = np.divide(Y[:, 0], weights[0][2])
    X3 += np.random.normal(loc=0, scale=X3.std(), size=N)
    X = np.array([X1, X2, X3])
    # Concatenate
    synth_data = np.concatenate([X.transpose(), Y], axis=1)
    
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
        return df

    def get_data(self):
        return self._data

    def get_np_data(self):
        return self.get_data().to_numpy()
