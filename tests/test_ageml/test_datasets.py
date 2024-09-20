import os
import importlib.resources as pkg_resources

import ageml.datasets as datasets
import ageml.datasets.synthetic_data as data_generator
from ageml.datasets import SyntheticData

# Expected paths
datasets_path = str(pkg_resources.files(datasets))
expected_paths = {
    "features": os.path.join(datasets_path, "toy_features.csv"),
    "clinical": os.path.join(datasets_path, "toy_clinical.csv"),
    "covariates": os.path.join(datasets_path, "toy_covar.csv"),
    "factors": os.path.join(datasets_path, "toy_factors.csv"),
    "systems": os.path.join(datasets_path, "toy_systems.txt"),
}


def test_generate_synthetic_data():
    # Generate synthetic data
    data_generator.generate_synthetic_data(save=True)

    # Check that the data was generated
    for _, path in expected_paths.items():
        assert os.path.exists(path), f"Path does not exist: {path}"


def test_synthetic_data_class():
    # Instantiate class
    synth_data = SyntheticData()

    err_str = "Data paths in SyntheticData class do not match expected paths."
    assert synth_data.data_paths == expected_paths, err_str
