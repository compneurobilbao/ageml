import pytest
import os
import sys
import shutil
import importlib.resources as pkg_resources
import pandas as pd

import ageml.ui as ui
import ageml.datasets as datasets
from ageml.ui import Interface, CLI


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
        self.features = os.path.join(str(pkg_resources.files(datasets)), "synthetic_features.csv")
        self.covariates = os.path.join(str(pkg_resources.files(datasets)), "synthetic_covariates.csv")
        self.factors = os.path.join(str(pkg_resources.files(datasets)), "synthetic_factors.csv")
        self.clinical = os.path.join(str(pkg_resources.files(datasets)), "synthetic_clinical.csv")


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
    assert isinstance(data, pd.core.frame.DataFrame)
    # Check that the column in the dataframe are lowercase
    assert all([col.islower() for col in data.columns])


def test_load_data(dummy_interface):
    # Load some data
    dummy_interface.load_data()
    
    # Check that the data is a pandas dataframe
    assert isinstance(dummy_interface.df_features, pd.core.frame.DataFrame)
    
    # Check that the column in the dataframe are lowercase
    assert all([col.islower() for col in dummy_interface.df_features.columns])


def test_run(dummy_interface):
    # Run the modelling pipeline
    dummy_interface.run()
    
    # Check for the existence of the output directory
    assert os.path.exists(dummy_interface.dir_path)
    
    # Check for the existence of the output figures
    figs = ["age_bias_correction", "age_distribution",
            "features_vs_age", "true_vs_pred_age"]
    svg_paths = [os.path.join(dummy_interface.dir_path,
                              f'figures/{fig}.svg') for fig in figs]
    assert all([os.path.exists(svg_path) for svg_path in svg_paths])
    
    # Check for the existence of the log
    log_path = os.path.join(dummy_interface.dir_path, "log.txt")
    assert os.path.exists(log_path)
    
    # Check for the existence of the output CSV
    csv_path = os.path.join(dummy_interface.dir_path, "predicted_age.csv")
    assert os.path.exists(csv_path)
    
    # Check that the output CSV has the right columns
    df = pd.read_csv(csv_path, header=0, index_col=0)
    assert all([col in df.columns for col in ["Age", "Predicted Age", "Corrected Age"]])


def test_cli_initialization(features_data_path):
    output_path = os.path.dirname(__file__)
    # Path sys.argv (command line arguments)
    # sys.argv[0] should be empty, so we set it to ''
    # TODO: Cleaner way to test CLI?
    sys.argv = ['', '-f', features_data_path,
                '-o', output_path]
    cli = CLI()
    
    # Check correct default initialization
    assert cli.args.features == features_data_path
    assert cli.args.model == ['linear']
    assert cli.args.scaler_type == 'standard'
    assert cli.args.cv == [5, 0]


# def test_cli_run(features_data_path):
#     output_path = os.path.dirname(__file__)
#     # Path sys.argv (command line arguments)
#     # sys.argv[0] should be empty, so we set it to ''
