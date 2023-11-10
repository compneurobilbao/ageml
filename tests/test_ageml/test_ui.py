import pytest
import os
import shutil
import importlib.resources as pkg_resources
import pandas as pd

import ageml.ui as ui
import ageml.datasets as datasets
from ageml.ui import Interface
from ageml.datasets import SyntheticData
from .test_modelling import AgeMLTest


class ExampleArguments(object):
    def __init__(self):
        test_path = os.path.dirname(__file__)
        self.scaler_type = "standard"
        self.scaler_params = {"with_mean": True}
        self.model_type = "linear"
        self.model_params = {"fit_intercept": True}
        self.cv_split = 5
        self.seed = 42
        self.output = test_path


@pytest.fixture
def features_data_path():
    return os.path.join(str(pkg_resources.files(datasets)), "synthetic_features.csv")


@pytest.fixture
def dummy_interface():
    return Interface(args=ExampleArguments())


# To be executed in every test in this module
@pytest.fixture(autouse=True)
def cleanup(dummy_interface):
    # Before each test
    
    yield
    # After each test
    # Cleanup the directories that were generated
    shutil.rmtree(dummy_interface.dir_path)


def test_interface_setup(dummy_interface):
    expected_args = ExampleArguments()
    # Now check that the attributes are the same
    # NOTE: This is not a good test, but it's a start.
    # TODO: How to make it more scalable?
    assert dummy_interface.args.scaler_type == expected_args.scaler_type
    assert dummy_interface.args.scaler_params == expected_args.scaler_params
    assert dummy_interface.args.model_type == expected_args.model_type
    assert dummy_interface.args.model_params == expected_args.model_params
    assert dummy_interface.args.cv_split == expected_args.cv_split
    assert dummy_interface.args.seed == expected_args.seed
    assert dummy_interface.args.output == expected_args.output


def test_load_csv(dummy_interface, features_data_path):
    data = dummy_interface.load_csv(features_data_path)
    
    # Check that the data is a pandas dataframe
    assert type(data) == pd.core.frame.DataFrame
    # Check that the column in the dataframe are lowercase
    assert all([col.islower() for col in data.columns])


# def test_load_data(dummy_interface, features_data_path):
#     data = dummy_interface.load_data(features_data_path)
#     pass
#     # Check that t


# def test_run(dummy_interface):
#     dummy_interface.run()
    
#     # Check file existance
#     svg_path = os.path.join(dummy_interface.dir_path,
#                             'figures/age_distribution.svg')
#     assert os.path.exists(svg_path)
