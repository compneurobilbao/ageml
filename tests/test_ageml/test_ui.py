import os
import pytest
import shutil
import tempfile
import random
import string
import pandas as pd
import numpy as np

import ageml.messages as messages
from ageml.ui import Interface, CLI, AgeML


class ExampleArguments(object):
    def __init__(self):
        self.scaler_type = "standard"
        self.scaler_params = {"with_mean": True}
        self.model_type = "linear_reg"
        self.model_params = {"fit_intercept": True}
        self.model_cv_split = 2
        self.model_seed = 0
        self.classifier_cv_split = 2
        self.classifier_seed = 0
        self.classifier_thr = 0.5
        self.classifier_ci = 0.95
        test_path = os.path.dirname(__file__)
        self.output = test_path
        self.features = None
        self.covariates = None
        self.covar_name = None
        self.systems = None
        self.factors = None
        self.clinical = None
        self.ages = None
        self.group1 = None
        self.group2 = None
        self.hyperparameter_tuning = 0
        self.feature_extension = 0


@pytest.fixture
def features():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9,
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "age": [50, 55, 60, 65, 70, 75, 80, 85, 90, 57,
                    53, 57, 61, 65, 69, 73, 77, 81, 85, 89],
            "feature1": [1.3, 2.2, 3.9, 4.1, 5.7, 6.4, 7.5, 8.2, 9.4, 1.7,
                         1.4, 2.2, 3.8, 4.5, 5.4, 6.2, 7.8, 8.2, 9.2, 2.6],
            "feature2": [9.4, 8.2, 7.5, 6.4, 5.3, 4.1, 3.9, 2.2, 1.3, 9.4,
                         9.3, 8.1, 7.9, 6.5, 5.0, 4.0, 3.7, 2.1, 1.4, 8.3],
        }
    )
    df.set_index("id", inplace=True)
    return df


@pytest.fixture
def factors():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9,
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "factor1": [1.3, 2.2, 3.9, 4.1, 5.7, 6.4, 7.5, 8.2, 9.4, 1.3,
                        1.3, 2.2, 3.9, 4.1, 5.7, 6.4, 7.5, 8.2, 9.4, 2.2],
            "factor2": [0.1, 1.3, 2.2, 3.9, 4.1, 5.7, 6.4, 7.5, 8.2, 9.4,
                        4.7, 3.7, 2.3, 1.2, 0.9, 0.3, 0.2, 0.1, 0.1, 0.1],
        }
    )
    df.set_index("id", inplace=True)
    return df


@pytest.fixture
def covariates():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9,
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "gender": ['f', 'm', 'm', 'm', 'f', 'f', 'f', 'm', 'm', 'f',
                       'f', 'f', 'm', 'm', 'f', 'f', 'f', 'm', 'm', 'f'],
        }
    )
    df.set_index("id", inplace=True)
    return df


@pytest.fixture
def systems():
    return 'pottongosystem:feature1\nmondongsystem:feature2'


@pytest.fixture
def clinical():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9,
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "CN": [True, False, True, False, True, False, True, False, True, False,
                   True, False, True, False, True, False, True, False, True, False],
            "group1": [False, True, False, True, False, True, False, True, False, True,
                       False, True, False, True, False, True, False, True, False, True],
        }
    )
    df.set_index("id", inplace=True)
    return df


@pytest.fixture
def ages():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9,
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "age": [50, 55, 60, 65, 70, 75, 80, 85, 90, 57,
                    53, 57, 61, 65, 69, 73, 77, 81, 85, 89],
            "predicted_age_all": [55, 67, 57, 75, 85, 64, 87, 93, 49, 51,
                                  58, 73, 80, 89, 55, 67, 57, 75, 85, 64],
            "corrected_age_all": [51, 58, 73, 80, 89, 67, 57, 75, 85, 64,
                                  87, 93, 49, 55, 67, 57, 75, 85, 64, 87],
            "delta_all": [1, -2, 3, 0, -1, 2, 1, 0, -3, 1,
                          2, 1, 0, -1, 2, 1, 0, -3, 1, 2],
        }
    )
    df.set_index("id", inplace=True)
    return df


@pytest.fixture
def ages_multisystem():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9,
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "age": [50, 55, 60, 65, 70, 75, 80, 85, 90, 57,
                    53, 57, 61, 65, 69, 73, 77, 81, 85, 89],
            "predicted_age_pottongosystem": [55, 67, 57, 75, 85, 64, 87, 93, 49, 51,
                                             58, 73, 80, 89, 55, 67, 57, 75, 85, 64],
            "corrected_age_pottongosystem": [51, 58, 73, 80, 89, 67, 57, 75, 85, 64,
                                             87, 93, 49, 55, 67, 57, 75, 85, 64, 87],
            "delta_pottongosystem": [1, -2, 3, 0, -1, 2, 1, 0, -3, 1,
                                     2, 1, 0, -1, 2, 1, 0, -3, 1, 2],
            "predicted_age_mondongsystem": [55, 67, 57, 75, 85, 64, 87, 93, 49, 51,
                                            58, 73, 80, 89, 55, 67, 57, 75, 85, 64],
            "corrected_age_mondongsystem": [51, 58, 73, 80, 89, 67, 57, 75, 85, 64,
                                            87, 93, 49, 55, 67, 57, 75, 85, 64, 87],
            "delta_mondongsystem": [1, -2, 3, 0, -1, 2, 1, 0, -3, 1,
                                    2, 1, 0, -1, 2, 1, 0, -3, 1, 2],
        }
    )
    df.set_index("id", inplace=True)
    return df


def create_csv(df, path):
    # Generate random name for the csv file
    letters = string.ascii_lowercase
    csv_name = "".join(random.choice(letters) for i in range(20)) + ".csv"
    file_path = os.path.join(path, csv_name)
    df.to_csv(path_or_buf=file_path, index=True)
    return file_path


def create_txt(txt, path):
    # Generate random name for the txt file
    letters = string.ascii_lowercase
    txt_name = "".join(random.choice(letters) for i in range(20)) + ".txt"
    file_path = os.path.join(path, txt_name)
    with open(file_path, 'w') as f:
        f.write(txt)
    return file_path


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


@pytest.fixture
def monkeypatch():
    """Monkeypatch fixture"""
    return pytest.MonkeyPatch()


@pytest.fixture
def dummy_cli(monkeypatch):
    """Dummy CLI fixture"""
    # Create temporary directory
    temp_dir = tempfile.TemporaryDirectory()

    # Define a list of responses
    responses = [temp_dir.name, "q"]

    # Patch the input function
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    interface = CLI()
    return interface


def test_interface_setup(dummy_interface):
    expected_args = ExampleArguments()
    # Now check that the attributes are the same
    # NOTE: This is not a good test, but it's a start.
    # NOTE: How to make it more scalable?
    assert dummy_interface.args.scaler_type == expected_args.scaler_type
    assert dummy_interface.args.scaler_params == expected_args.scaler_params
    assert dummy_interface.args.model_type == expected_args.model_type
    assert dummy_interface.args.model_params == expected_args.model_params
    assert dummy_interface.args.model_cv_split == expected_args.model_cv_split
    assert dummy_interface.args.model_seed == expected_args.model_seed
    assert dummy_interface.args.output == expected_args.output


def test_load_csv(dummy_interface, features):
    features_path = create_csv(features, dummy_interface.dir_path)
    dummy_interface.args.features = features_path
    data = dummy_interface.load_csv('features')

    # Check that the data is a pandas dataframe
    assert isinstance(data, pd.core.frame.DataFrame)
    # Check that the column in the dataframe are lowercase
    assert all([col.islower() for col in data.columns])


def test_load_data(dummy_interface, features):
    # Load some data
    features_path = create_csv(features, dummy_interface.dir_path)
    dummy_interface.args.features = features_path
    dummy_interface.load_data()

    # Check that the data is a pandas dataframe
    assert isinstance(dummy_interface.df_features, pd.core.frame.DataFrame)

    # Check that the column in the dataframe are lowercase
    assert all([col.islower() for col in dummy_interface.df_features.columns])


def test_load_data_age_not_column(dummy_interface, features):
    # Remove age columng from features
    features.drop("age", axis=1, inplace=True)
    features_path = create_csv(features, dummy_interface.dir_path)
    dummy_interface.args.features = features_path

    # Test error risen
    with pytest.raises(KeyError) as exc_info:
        dummy_interface.load_data()
    assert exc_info.type == KeyError
    error_message = "Features file must contain a column name 'age', or any other case-insensitive variation."
    assert exc_info.value.args[0] == error_message


def test_load_data_cn_not_column(dummy_interface, clinical):
    # Test no CN column in clinical
    clinical.drop("CN", axis=1, inplace=True)
    clinical_path = create_csv(clinical, dummy_interface.dir_path)
    dummy_interface.args.clinical = clinical_path

    # Test error risen
    with pytest.raises(KeyError) as exc_info:
        dummy_interface.load_data()
    assert exc_info.type == KeyError
    error_message = "Clinical file must contain a column name 'CN' or any other case-insensitive variation."
    assert exc_info.value.args[0] == error_message


def test_load_data_ages_missing_column(dummy_interface, ages):
    # Test removal of columns
    cols = ["age", "predicted_age", "corrected_age", "delta"]
    for col in cols:
        # Remove column
        df = ages.copy()
        col_drop = [c for c in df.columns if c.startswith(col)]
        df.drop(col_drop[0], axis=1, inplace=True)
        file_path = create_csv(df, dummy_interface.dir_path)
        dummy_interface.args.ages = file_path

        # Test error risen
        with pytest.raises(KeyError) as exc_info:
            dummy_interface.load_data()
        assert exc_info.type == KeyError
        error_message = "Ages file missing the following column %s, or derived names." % col
        assert exc_info.value.args[0] == error_message


def test_load_data_required_file_types(dummy_interface):
    dummy_interface.args.features = None
    with pytest.raises(ValueError) as exc_info:
        dummy_interface.load_data(required=["features"])
    assert exc_info.type == ValueError
    assert exc_info.value.args[0] == "Features file must be provided."

    dummy_interface.args.clinical = None
    with pytest.raises(ValueError) as exc_info:
        dummy_interface.load_data(required=["clinical"])
    assert exc_info.type == ValueError
    assert exc_info.value.args[0] == "Clinical file must be provided."

    dummy_interface.args.factors = None
    with pytest.raises(ValueError) as exc_info:
        dummy_interface.load_data(required=["factors"])
    assert exc_info.type == ValueError
    assert exc_info.value.args[0] == "Factors file must be provided."


def test_load_data_clinical_not_boolean(dummy_interface, clinical):
    # Change booleans to other types
    clinical.loc[2, "CN"] = 1.3
    clinical.loc[3, "group1"] = "mondongo"  # excellent placeholder
    clinical_path = create_csv(clinical, dummy_interface.dir_path)
    dummy_interface.args.clinical = clinical_path

    # Test error risen
    with pytest.raises(TypeError) as exc_info:
        dummy_interface.load_data()
    assert exc_info.type == TypeError
    assert exc_info.value.args[0] == "Clinical columns must be boolean type. Check that all values are encoded as 'True' or 'False'."


def test_load_data_nan_values_warning(dummy_interface, features):
    # Remove from features a few values
    features.loc[2, "feature1"] = np.nan
    features.loc[3, "feature2"] = np.nan
    features_path = create_csv(features, dummy_interface.dir_path)
    dummy_interface.args.features = features_path

    with pytest.warns(UserWarning) as warn_record:
        dummy_interface.load_data()
    assert isinstance(warn_record.list[0].message, UserWarning)
    expected = f"Subjects with missing data in features: {[2, 3]}"
    assert warn_record.list[0].message.args[0] == expected


def test_load_data_different_indexes_warning(dummy_interface, features, clinical):
    # Drop subjects 2 and 3 from features
    features.drop([2, 3], axis=0, inplace=True)
    clinical.drop([4], axis=0, inplace=True)
    features_path = create_csv(features, dummy_interface.dir_path)
    clinical_path = create_csv(clinical, dummy_interface.dir_path)
    dummy_interface.args.features = features_path
    dummy_interface.args.clinical = clinical_path

    with pytest.warns(UserWarning) as warn_record:
        dummy_interface.load_data()
    assert isinstance(warn_record.list[0].message, UserWarning)
    expected = "Subjects in dataframe features not in dataframe clinical: [%d]" % (4)
    assert warn_record.list[0].message.args[0] == expected
    assert isinstance(warn_record.list[1].message, UserWarning)
    expected = "Subjects in dataframe clinical not in dataframe features: [%d, %d]" % (2, 3)
    assert warn_record.list[1].message.args[0] == expected


def test_age_distribution_warning(dummy_interface):
    dist1 = np.random.normal(loc=50, scale=1, size=100)
    dist2 = np.random.normal(loc=0, scale=1, size=100)
    dists = {'dist1': dist1, 'dist2': dist2}
    with pytest.warns(UserWarning) as warn_record:
        dummy_interface.age_distribution(dists)
    assert isinstance(warn_record.list[0].message, UserWarning)
    expected = "Age distributions %s and %s are not similar" % ("dist1", "dist2")
    assert warn_record.list[0].message.args[0].startswith(expected)


def test_run_age(dummy_interface, features):
    # Run the modelling pipeline
    features_path = create_csv(features, dummy_interface.dir_path)
    dummy_interface.args.features = features_path
    dummy_interface.run_age()

    # Check for the existence of the output directory
    assert os.path.exists(dummy_interface.dir_path)

    # Check for the existence of the output figures
    figs = [
        "age_bias_correction_all_all",
        "age_distribution_controls",
        "features_vs_age_controls_all",
        "chronological_vs_pred_age_all_all",
    ]
    svg_paths = [
        os.path.join(dummy_interface.dir_path, f"model_age/figures/{fig}.png") for fig in figs
    ]
    assert all([os.path.exists(svg_path) for svg_path in svg_paths])

    # Check for the existence of the log
    log_path = os.path.join(dummy_interface.dir_path, "log.txt")
    assert os.path.exists(log_path)

    # Check for the existence of the output CSV
    csv_path = os.path.join(dummy_interface.dir_path, "predicted_age.csv")
    assert os.path.exists(csv_path)

    # Check that the output CSV has the right columns
    df = pd.read_csv(csv_path, header=0, index_col=0)
    assert all([col in df.columns for col in ["age", "predicted_age_all", "corrected_age_all", "delta_all"]])


def test_run_age_clinical(dummy_interface, features, clinical):
    # Run the modelling pipeline
    features_path = create_csv(features, dummy_interface.dir_path)
    dummy_interface.args.features = features_path
    # Clinical
    clinical_path = create_csv(clinical, dummy_interface.dir_path)
    dummy_interface.args.clinical = clinical_path
    dummy_interface.run_age()

    # Check for the existence of the output directory
    assert os.path.exists(dummy_interface.dir_path)

    # Check for the existence of the output figures
    figs = [
        "age_bias_correction_all_all",
        "age_distribution_controls",
        "features_vs_age_controls_all",
        "chronological_vs_pred_age_all_all",
    ]
    svg_paths = [
        os.path.join(dummy_interface.dir_path, f"model_age/figures/{fig}.png") for fig in figs
    ]
    assert all([os.path.exists(svg_path) for svg_path in svg_paths])

    # Check for the existence of the log
    log_path = os.path.join(dummy_interface.dir_path, "log.txt")
    assert os.path.exists(log_path)

    # Check for the existence of the output CSV
    csv_path = os.path.join(dummy_interface.dir_path, "predicted_age.csv")
    assert os.path.exists(csv_path)

    # Check that the output CSV has the right columns
    df = pd.read_csv(csv_path, header=0, index_col=0)
    assert all([col in df.columns for col in ["age", "predicted_age_all", "corrected_age_all", "delta_all"]])


def test_run_age_cov(dummy_interface, features, covariates):
    # Features
    features_path = create_csv(features, dummy_interface.dir_path)
    dummy_interface.args.features = features_path
    # Covariates
    covariates_path = create_csv(covariates, dummy_interface.dir_path)
    dummy_interface.args.covariates = covariates_path
    # Covariate name
    dummy_interface.args.covar_name = "gender"
    # Run the modelling pipeline
    dummy_interface.run_age()
    
    # Check for output dir
    assert os.path.exists(dummy_interface.dir_path)

    # Check for output figs
    figs = ["age_bias_correction_f_all",
            "age_bias_correction_m_all",
            "chronological_vs_pred_age_f_all",
            "chronological_vs_pred_age_m_all",
            f"age_distribution_controls_{dummy_interface.args.covar_name}",
            f"features_vs_age_controls_{dummy_interface.args.covar_name}_all"]
    # Print files in path
    svg_paths = [os.path.join(dummy_interface.dir_path, f"model_age/figures/{fig}.png") for fig in figs]
    assert all([os.path.exists(svg_path) for svg_path in svg_paths])
    
    # Check for the existence of the output CSV
    csv_path = os.path.join(dummy_interface.dir_path,
                            f"predicted_age_{dummy_interface.args.covar_name}.csv")
    assert os.path.exists(csv_path)

    # Check that the output CSV has the right columns
    df = pd.read_csv(csv_path, header=0, index_col=0)
    assert all([col in df.columns for col in ["age", "predicted_age_all", "corrected_age_all", "delta_all"]])


def test_run_age_cov_clinical(dummy_interface, features, covariates, clinical):
    # Features
    features_path = create_csv(features, dummy_interface.dir_path)
    dummy_interface.args.features = features_path
    # Covariates
    covariates_path = create_csv(covariates, dummy_interface.dir_path)
    dummy_interface.args.covariates = covariates_path
    # Covariate name
    dummy_interface.args.covar_name = "gender"
    # Clinical file
    clinical_path = create_csv(clinical, dummy_interface.dir_path)
    dummy_interface.args.clinical = clinical_path
    # Run the modelling pipeline
    dummy_interface.run_age()
    
    # Check for output dir
    assert os.path.exists(dummy_interface.dir_path)

    # Check for output figs
    figs = ["age_bias_correction_f_all",
            "age_bias_correction_m_all",
            "chronological_vs_pred_age_f_all",
            "chronological_vs_pred_age_m_all",
            f"age_distribution_controls_{dummy_interface.args.covar_name}",
            f"features_vs_age_controls_{dummy_interface.args.covar_name}_all"]
    svg_paths = [os.path.join(dummy_interface.dir_path, f"model_age/figures/{fig}.png") for fig in figs]
    assert all([os.path.exists(svg_path) for svg_path in svg_paths])
    
    # Check for the existence of the output CSV
    csv_path = os.path.join(dummy_interface.dir_path,
                            f"predicted_age_{dummy_interface.args.covar_name}.csv")
    assert os.path.exists(csv_path)

    # Check that the output CSV has the right columns
    df = pd.read_csv(csv_path, header=0, index_col=0)
    assert all([col in df.columns for col in ["age", "predicted_age_all", "corrected_age_all", "delta_all"]])


def test_run_age_systems(dummy_interface, systems, features):
    # Features
    features_path = create_csv(features, dummy_interface.dir_path)
    dummy_interface.args.features = features_path
    # Systems
    systems_path = create_txt(systems, dummy_interface.dir_path)
    dummy_interface.args.systems = systems_path
    # Run the modelling pipeline
    dummy_interface.run_age()
    
    # Check for output dir
    assert os.path.exists(dummy_interface.dir_path)
    
    # Systems names
    system_names = list(dummy_interface.dict_systems.keys())
    figs = ["age_distribution_controls_multisystem"]
    for system_name in system_names:
        figs.append(f"age_bias_correction_all_{system_name}")
        figs.append(f"chronological_vs_pred_age_all_{system_name}")
        figs.append(f"features_vs_age_controls_multisystem_{system_name}")
    # Check existance of figures
    svg_paths = [os.path.join(dummy_interface.dir_path, f"model_age/figures/{fig}.png") for fig in figs]
    assert all([os.path.exists(svg_path) for svg_path in svg_paths])
    
    # Check existence of output CSV
    csv_path = os.path.join(dummy_interface.dir_path,
                            "predicted_age_multisystem.csv")
    assert os.path.exists(csv_path)
    
    # Check that the output CSV has the right columns
    df = pd.read_csv(csv_path, header=0, index_col=0)
    assert all(any(word in s for s in df.columns) for word in ["age", "predicted_age", "corrected_age", "delta"])


def test_run_age_systems_clinical(dummy_interface, systems, features, clinical):
    # Features
    features_path = create_csv(features, dummy_interface.dir_path)
    dummy_interface.args.features = features_path
    # Systems
    systems_path = create_txt(systems, dummy_interface.dir_path)
    dummy_interface.args.systems = systems_path
    # Clinical
    clinical_path = create_csv(clinical, dummy_interface.dir_path)
    dummy_interface.args.clinical = clinical_path
    # Run the modelling pipeline
    dummy_interface.run_age()
    
    # Check for output dir
    assert os.path.exists(dummy_interface.dir_path)
    
    # Systems names
    system_names = list(dummy_interface.dict_systems.keys())
    figs = ["age_distribution_controls_multisystem"]
    for system_name in system_names:
        figs.append(f"age_bias_correction_all_{system_name}")
        figs.append(f"chronological_vs_pred_age_all_{system_name}")
        figs.append(f"features_vs_age_controls_multisystem_{system_name}")
    # Check existance of figures
    svg_paths = [os.path.join(dummy_interface.dir_path, f"model_age/figures/{fig}.png") for fig in figs]
    assert all([os.path.exists(svg_path) for svg_path in svg_paths])
    
    # Check existence of output CSV
    csv_path = os.path.join(dummy_interface.dir_path,
                            "predicted_age_multisystem.csv")
    assert os.path.exists(csv_path)
    
    # Check that the output CSV has the right columns
    df = pd.read_csv(csv_path, header=0, index_col=0)
    assert all(any(word in s for s in df.columns) for word in ["age", "predicted_age", "corrected_age", "delta"])


def test_run_age_cov_and_systems(dummy_interface, systems, features, covariates):
    # Features
    features_path = create_csv(features, dummy_interface.dir_path)
    dummy_interface.args.features = features_path
    # Covariates
    covariates_path = create_csv(covariates, dummy_interface.dir_path)
    dummy_interface.args.covariates = covariates_path
    # Covariate name
    dummy_interface.args.covar_name = "gender"
    # Systems
    systems_path = create_txt(systems, dummy_interface.dir_path)
    dummy_interface.args.systems = systems_path
    # Run the modelling pipeline
    dummy_interface.run_age()
    
    # Check for output dir
    assert os.path.exists(dummy_interface.dir_path)
    
    # Systems names
    system_names = list(dummy_interface.dict_systems.keys())
    figs = ["age_distribution_controls_gender_multisystem"]
    for system_name in system_names:
        figs.append(f"age_bias_correction_f_{system_name}")
        figs.append(f"age_bias_correction_m_{system_name}")
        figs.append(f"chronological_vs_pred_age_f_{system_name}")
        figs.append(f"chronological_vs_pred_age_m_{system_name}")
        figs.append(f"features_vs_age_controls_gender_multisystem_{system_name}")
    # Check existance of figures
    svg_paths = [os.path.join(dummy_interface.dir_path, f"model_age/figures/{fig}.png") for fig in figs]
    assert all([os.path.exists(svg_path) for svg_path in svg_paths])
    
    # Check existence of output CSV
    csv_path = os.path.join(dummy_interface.dir_path,
                            f"predicted_age_{dummy_interface.args.covar_name}_multisystem.csv")
    assert os.path.exists(csv_path)
    
    # Check that the output CSV has the right columns
    df = pd.read_csv(csv_path, header=0, index_col=0)
    assert all(any(word in s for s in df.columns) for word in ["age", "predicted_age", "corrected_age", "delta"])


def test_run_age_cov_and_systems_clinical(dummy_interface, systems, features, covariates, clinical):
    # Features
    features_path = create_csv(features, dummy_interface.dir_path)
    dummy_interface.args.features = features_path
    # Covariates
    covariates_path = create_csv(covariates, dummy_interface.dir_path)
    dummy_interface.args.covariates = covariates_path
    # Covariate name
    dummy_interface.args.covar_name = "gender"
    # Systems
    systems_path = create_txt(systems, dummy_interface.dir_path)
    dummy_interface.args.systems = systems_path
    # Clinical
    clinical_path = create_csv(clinical, dummy_interface.dir_path)
    dummy_interface.args.clinical = clinical_path
    # Run the modelling pipeline
    dummy_interface.run_age()
    
    # Check for output dir
    assert os.path.exists(dummy_interface.dir_path)
    
    # Systems names
    system_names = list(dummy_interface.dict_systems.keys())
    figs = ["age_distribution_controls_gender_multisystem"]
    for system_name in system_names:
        figs.append(f"age_bias_correction_f_{system_name}")
        figs.append(f"age_bias_correction_m_{system_name}")
        figs.append(f"chronological_vs_pred_age_f_{system_name}")
        figs.append(f"chronological_vs_pred_age_m_{system_name}")
        figs.append(f"features_vs_age_controls_gender_multisystem_{system_name}")
    # Check existance of figures
    svg_paths = [os.path.join(dummy_interface.dir_path, f"model_age/figures/{fig}.png") for fig in figs]
    assert all([os.path.exists(svg_path) for svg_path in svg_paths])
    
    # Check existence of output CSV
    csv_path = os.path.join(dummy_interface.dir_path,
                            f"predicted_age_{dummy_interface.args.covar_name}_multisystem.csv")
    assert os.path.exists(csv_path)
   
    # Check that the output CSV has the right columns
    df = pd.read_csv(csv_path, header=0, index_col=0)
    assert all(any(word in s for s in df.columns) for word in ["age", "predicted_age", "corrected_age", "delta"])


def test_run_factor_correlation(dummy_interface, ages, factors):
    # Run the lifestyle pipeline
    ages_path = create_csv(ages, dummy_interface.dir_path)
    factors_path = create_csv(factors, dummy_interface.dir_path)
    dummy_interface.args.ages = ages_path
    dummy_interface.args.factors = factors_path
    dummy_interface.run_factor_correlation()

    # Check for the existence of the output directory
    assert os.path.exists(dummy_interface.dir_path)

    # Check for the existence of the output figures
    figs = ["factors_vs_deltas_system_cn"]
    svg_paths = [
        os.path.join(dummy_interface.dir_path, f"factor_correlation/figures/{fig}.png") for fig in figs
    ]
    assert all([os.path.exists(svg_path) for svg_path in svg_paths])

    # Check for the existence of the log
    log_path = os.path.join(dummy_interface.dir_path, "log.txt")
    assert os.path.exists(log_path)


def test_run_factor_correlation_systems(dummy_interface, ages_multisystem, factors):
    # Run the lifestyle pipeline
    ages_path = create_csv(ages_multisystem, dummy_interface.dir_path)
    factors_path = create_csv(factors, dummy_interface.dir_path)
    dummy_interface.args.ages = ages_path
    dummy_interface.args.factors = factors_path
    dummy_interface.run_factor_correlation()

    # Check for the existence of the output directory
    assert os.path.exists(dummy_interface.dir_path)

    # Check for the existence of the output figures
    figs = []
    figs.append("factors_vs_deltas_system_cn")
    svg_paths = [
        os.path.join(dummy_interface.dir_path, f"factor_correlation/figures/{fig}.png") for fig in figs
    ]
    assert all([os.path.exists(svg_path) for svg_path in svg_paths])

    # Check for the existence of the log
    log_path = os.path.join(dummy_interface.dir_path, "log.txt")
    assert os.path.exists(log_path)


def test_run_clinical(dummy_interface, ages, clinical):
    # Run the clinical pipeline
    ages_path = create_csv(ages, dummy_interface.dir_path)
    clinical_path = create_csv(clinical, dummy_interface.dir_path)
    dummy_interface.args.ages = ages_path
    dummy_interface.args.clinical = clinical_path
    dummy_interface.run_clinical()

    # Check for the existence of the output directory
    assert os.path.exists(dummy_interface.dir_path)

    # Check for the existence of the output figures
    figs = ["age_distribution_clinical_groups", "clinical_groups_box_plot_all"]
    svg_paths = [
        os.path.join(dummy_interface.dir_path, f"clinical_groups/figures/{fig}.png") for fig in figs
    ]
    assert all([os.path.exists(svg_path) for svg_path in svg_paths])

    # Check for the existence of the log
    log_path = os.path.join(dummy_interface.dir_path, "log.txt")
    assert os.path.exists(log_path)


def test_run_clinical_systems(dummy_interface, ages_multisystem, clinical):
    # Run the clinical pipeline
    ages_path = create_csv(ages_multisystem, dummy_interface.dir_path)
    clinical_path = create_csv(clinical, dummy_interface.dir_path)
    dummy_interface.args.ages = ages_path
    dummy_interface.args.clinical = clinical_path
    dummy_interface.run_clinical()

    # Check for the existence of the output directory
    assert os.path.exists(dummy_interface.dir_path)

    # Check for the existence of the output figures
    system_names = list({col.split("_")[-1] for col in ages_multisystem if "system" in col})
    figs = ["age_distribution_clinical_groups"]
    for system in system_names:
        figs.append(f"clinical_groups_box_plot_{system}")
    svg_paths = [
        os.path.join(dummy_interface.dir_path, f"clinical_groups/figures/{fig}.png") for fig in figs
    ]
    assert all([os.path.exists(svg_path) for svg_path in svg_paths])

    # Check for the existence of the log
    log_path = os.path.join(dummy_interface.dir_path, "log.txt")
    assert os.path.exists(log_path)


def test_run_classification(dummy_interface, ages, clinical):
    # Run the classification pipeline
    ages_path = create_csv(ages, dummy_interface.dir_path)
    clinical_path = create_csv(clinical, dummy_interface.dir_path)
    dummy_interface.args.group1 = 'cn'
    dummy_interface.args.group2 = 'group1'
    dummy_interface.args.ages = ages_path
    dummy_interface.args.clinical = clinical_path
    dummy_interface.run_classification()

    # Check for the existence of the output directory
    assert os.path.exists(dummy_interface.dir_path)

    # Check for the existence of the output figures
    figs = ["roc_curve_cn_vs_group1_all"]
    svg_paths = [
        os.path.join(dummy_interface.dir_path, f"clinical_classify/figures/{fig}.png") for fig in figs
    ]
    assert all([os.path.exists(svg_path) for svg_path in svg_paths])

    # Check for the existence of the log
    log_path = os.path.join(dummy_interface.dir_path, "log.txt")
    assert os.path.exists(log_path)


def test_run_classification_systems(dummy_interface, ages_multisystem, clinical):
    # Run the classification pipeline
    ages_path = create_csv(ages_multisystem, dummy_interface.dir_path)
    clinical_path = create_csv(clinical, dummy_interface.dir_path)
    dummy_interface.args.group1 = 'cn'
    dummy_interface.args.group2 = 'group1'
    dummy_interface.args.ages = ages_path
    dummy_interface.args.clinical = clinical_path
    dummy_interface.run_classification()

    # Check for the existence of the output directory
    assert os.path.exists(dummy_interface.dir_path)

    # Check for the existence of the output figures
    system_names = list({col.split("_")[-1] for col in ages_multisystem if "system" in col})
    figs = []
    for system in system_names:
        figs.append(f"roc_curve_{dummy_interface.args.group1}_vs_{dummy_interface.args.group2}_{system}")
    svg_paths = [
        os.path.join(dummy_interface.dir_path, f"clinical_classify/figures/{fig}.png") for fig in figs
    ]
    assert all([os.path.exists(svg_path) for svg_path in svg_paths])

    # Check for the existence of the log
    log_path = os.path.join(dummy_interface.dir_path, "log.txt")
    assert os.path.exists(log_path)


def test_classification_group_not_given(dummy_interface, ages, clinical):

    # Run create classification pipeline with no groups
    ages_path = create_csv(ages, dummy_interface.dir_path)
    clinical_path = create_csv(clinical, dummy_interface.dir_path)
    dummy_interface.args.ages = ages_path
    dummy_interface.args.clinical = clinical_path

    # Run classification and capture error
    with pytest.raises(ValueError) as exc_info:
        dummy_interface.run_classification()
    assert exc_info.type == ValueError
    assert exc_info.value.args[0] == "Must provide two groups to classify."


def test_classifcation_group_not_in_columns(dummy_interface, ages, clinical):
    # Run create classification pipeline with no groups
    ages_path = create_csv(ages, dummy_interface.dir_path)
    clinical_path = create_csv(clinical, dummy_interface.dir_path)
    dummy_interface.args.ages = ages_path
    dummy_interface.args.clinical = clinical_path
    dummy_interface.args.group1 = 'cn'
    dummy_interface.args.group2 = 'group3'
    
    # Run classification and capture error
    with pytest.raises(ValueError) as exc_info:
        dummy_interface.run_classification()
    assert exc_info.type == ValueError
    error_msg = "Classes must be one of the following: ['%s', '%s']" % ('cn', 'group1')
    assert exc_info.value.args[0] == error_msg


def test_interface_setup_dir_existing_warning(dummy_interface):
    # Setup another dummy_interface in the newly created directory
    with pytest.warns(UserWarning) as warn_record:
        Interface(args=ExampleArguments())
    assert isinstance(warn_record.list[0].message, UserWarning)
    error_message = f"Directory {dummy_interface.dir_path} already exists files may be overwritten."
    assert warn_record.list[0].message.args[0] == error_message


def test_configure_CLI(dummy_cli):
    """Test dummy CLI configured"""
    assert dummy_cli.configFlag is True


def test_get_line_CLI(dummy_cli, monkeypatch, capsys):
    """Test dummy CLI getline"""

    # Test that without required can pass empty string
    monkeypatch.setattr("builtins.input", lambda _: "")
    dummy_cli.get_line(required=False)
    assert dummy_cli.line == ""

    # Test that with required cannot pass empty string
    responses = ["", "mondong"]
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    dummy_cli.get_line(required=True)
    captured = capsys.readouterr()
    assert captured.out == "Must provide a value.\n"
    assert dummy_cli.line == "mondong"


def test_force_command_CLI(dummy_cli, monkeypatch):
    """Test dummy CLI force command"""

    # Test when no input is given and not required
    monkeypatch.setattr("builtins.input", lambda _: "")
    error = dummy_cli.force_command(
        dummy_cli.load_command, "--systems", required=False
    )
    assert error is None
    assert dummy_cli.line == ["--systems", "None"]

    # Test when correct input is error returned is None
    monkeypatch.setattr("builtins.input", lambda _: "linear_reg")
    error = dummy_cli.force_command(dummy_cli.model_command, required=True)
    assert error is None


def test_command_interface_CLI(dummy_cli, monkeypatch, capsys):
    """Test dummy CLI command interface"""

    # Test command that does not exist
    responses = ["mnoang", "q"]
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split("\n")[:-1]
    assert captured[-1] == "Invalid command. Enter 'h' for help."

    # Test running quit command
    responses = ["q"]
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split("\n")[:-1]
    assert captured[-1] == "Enter 'h' for help."


def test_classifier_command_CLI(dummy_cli):
    """Test dummy CLI classifier command"""

    # Test no input
    dummy_cli.line = ""
    error = dummy_cli.classifier_command()
    assert error == "Must provide two arguments or None."

    # Test default
    dummy_cli.line = "None"
    error = dummy_cli.classifier_command()
    assert error is None
    assert dummy_cli.args.classifier_thr == 0.5
    assert dummy_cli.args.classifier_ci == 0.95

    # Test error float
    dummy_cli.line = "mondongo"
    error = dummy_cli.classifier_command()
    assert error == "Parameters must be floats."

    # Test passing two arguments
    dummy_cli.line = "0.1 0.1"
    error = dummy_cli.classifier_command()
    assert error is None
    assert dummy_cli.args.classifier_thr == 0.1
    assert dummy_cli.args.classifier_ci == 0.1

    # Test passing too many arguments
    dummy_cli.line = "0.1 0.1 0.1"
    error = dummy_cli.classifier_command()
    assert error == "Too many values to unpack."


def test_classification_command_CLI(dummy_cli, ages, clinical, monkeypatch, capsys):
    """Test dummy CLI classification command"""

    # Create temporary directory to store data
    tempDir = tempfile.TemporaryDirectory()
    dummy_cli.args.output = tempDir.name
    dummy_cli.set_visualizer(tempDir.name)

    # Create features csv file
    ages_path = create_csv(ages, tempDir.name)
    clinical_path = create_csv(clinical, tempDir.name)

    # Test command
    responses = ["classification", ages_path, clinical_path, "cn group1", "", "", "", "q"]
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split("\n")[:-1]
    assert captured[-1] == 'Finished classification.'

    # Test command with invalid input like incorrect groups
    responses = ["classification", ages_path, clinical_path, "cn group2", "", "", "", "q"]
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split("\n")[:-1]
    print(captured)
    assert captured[-1] == "Error running classification."


def test_clinical_command_CLI(dummy_cli, ages, clinical, monkeypatch, capsys):
    """Test dummy CLI clinical command"""

    # Create temporary directory to store data
    tempDir = tempfile.TemporaryDirectory()
    dummy_cli.args.output = tempDir.name
    dummy_cli.set_visualizer(tempDir.name)

    # Create features csv file
    ages_path = create_csv(ages, tempDir.name)
    clinical_path = create_csv(clinical, tempDir.name)

    # Test command
    responses = ["clinical", ages_path, clinical_path, "", "q"]
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split("\n")[:-1]
    assert captured[-1] == 'Finished clinical analysis.'

    # Test command with invalid input like incorrect file
    responses = ["clinical", ages_path, ages_path, "", "q"]
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split("\n")[:-1]
    print(captured)
    assert captured[-1] == "Error running clinical analysis."


def test_covar_command_CLI(dummy_cli):
    """Test dummy CLI covar command"""

    # Test invalid input
    inputs = ["", "covar1 covar2"]
    for i in inputs:
        dummy_cli.line = i
        error = dummy_cli.covar_command()
        assert error == "Must provide one covariate name."

    # Test passing one covariate
    dummy_cli.line = "covar1"
    error = dummy_cli.covar_command()
    assert error is None
    assert dummy_cli.args.covar_name == "covar1"


def test_cv_command_CLI(dummy_cli):
    """Test dummy CLI cv command"""

    # Test no input
    dummy_cli.line = ""
    error = dummy_cli.cv_command()
    assert error == "Must provide at least one argument."

    # Test default values
    dummy_cli.line = "model None"
    error = dummy_cli.cv_command()
    assert error is None
    assert dummy_cli.args.model_cv_split == 5
    assert dummy_cli.args.model_seed == 0

    # Test non existent flag
    dummy_cli.line = "mondongo"
    error = dummy_cli.cv_command()
    assert error == "Must provide either model or classifier flag."

    # Test non-integer values
    dummy_cli.line = "model 2.5"
    error = dummy_cli.cv_command()
    assert error == "CV parameters must be integers"
    dummy_cli.line = "model 2 3.5"
    error = dummy_cli.cv_command()
    assert error == "CV parameters must be integers"

    # Test passing too many arguments
    dummy_cli.line = "model 1 2 3"
    error = dummy_cli.cv_command()
    assert error == "Too many values to unpack."

    # Test correct parsing
    dummy_cli.line = "model 1"
    error = dummy_cli.cv_command()
    assert error is None
    assert dummy_cli.args.model_cv_split == 1
    assert dummy_cli.args.model_seed == 0
    dummy_cli.line = "model 1 2"
    error = dummy_cli.cv_command()
    assert error is None
    assert dummy_cli.args.model_cv_split == 1
    assert dummy_cli.args.model_seed == 2


def test_factor_correlation_command_CLI(dummy_cli, ages, factors, monkeypatch, capsys):
    """Test dummy CLI factor_correlation command"""

    # Create temporary directory to store data
    tempDir = tempfile.TemporaryDirectory()
    dummy_cli.args.output = tempDir.name
    dummy_cli.set_visualizer(tempDir.name)

    # Create features csv file
    ages_path = create_csv(ages, tempDir.name)
    factors_path = create_csv(factors, tempDir.name)
    empty_path = create_csv(pd.DataFrame([]), tempDir.name)

    # Test command
    responses = ["factor_correlation", ages_path, factors_path, "", "", "q"]
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split("\n")[:-1]
    print(captured)
    assert captured[-1] == 'Finished factor correlation analysis.'

    # Test command with invalid input like incorrect file
    responses = ["factor_correlation", ages_path, empty_path, "", "", "q"]
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split("\n")[:-1]
    print(captured)
    assert captured[-1] == "Error running factor correlation analysis."


def test_group_command_CLI(dummy_cli):
    """Test dummy CLI group command"""

    # Test invalid input
    inputs = ["", "group1", "group1 group2 group3"]
    for i in inputs:
        dummy_cli.line = i
        error = dummy_cli.group_command()
        assert error == "Must provide two groups."

    # Test passing two groups
    dummy_cli.line = "cn group1"
    error = dummy_cli.group_command()
    assert error is None
    assert dummy_cli.args.group1 == "cn"
    assert dummy_cli.args.group2 == "group1"


def test_help_command_CLI(dummy_cli, capsys):
    """Test dummy CLI help command"""
    dummy_cli.help_command()
    captured = capsys.readouterr().out.split("\n")
    assert captured[0] == "User commands:"
    assert captured[1] == messages.classification_command_message
    assert captured[2] == messages.clinical_command_message
    assert captured[3] == messages.factor_correlation_command_message
    assert captured[4] == messages.model_age_command_message
    assert captured[5] == messages.quit_command_message


def test_load_command_CLI(dummy_cli):
    """Test dummy CLI load command"""

    # Test no input
    dummy_cli.line = ""
    error = dummy_cli.load_command()
    assert error == "Must provide a file type and file path."

    # Test passing only one input
    dummy_cli.line = "--features"
    error = dummy_cli.load_command()
    assert error == "Must provide a file path or None when using --file_type."

    # Test passing too many arguments
    dummy_cli.line = "--features file1 file2"
    error = dummy_cli.load_command()
    assert error == "Too many arguments only two arguments --file_type and file path."

    # Test passing non existant file type
    dummy_cli.line = "--features file1"
    error = dummy_cli.load_command()
    assert error == "File file1 not found."

    # Create a temporary file
    tmpcsv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmptxt = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)

    # Test passing incorrect file type
    dummy_cli.line = "--features " + tmptxt.name
    error = dummy_cli.load_command()
    assert error == "File %s must be a .csv file." % tmptxt.name
    dummy_cli.line = "--systems " + tmpcsv.name
    error = dummy_cli.load_command()
    assert error == "File %s must be a .txt file." % tmpcsv.name

    # Test choosing invalid file type
    dummy_cli.line = "--flag " + tmpcsv.name
    error = dummy_cli.load_command()
    error_message = "Choose a valid file type: --features, --covariates, --factors, --clinical, --systems, --ages"
    assert error == error_message

    # Test passing correct arguments
    dummy_cli.line = "--features " + tmpcsv.name
    error = dummy_cli.load_command()
    assert error is None
    assert dummy_cli.args.features == tmpcsv.name


def test_model_age_command_CLI(dummy_cli, features, monkeypatch, capsys):
    """Test dummy CLI model_age command"""

    # Create temporary directory to store data
    tempDir = tempfile.TemporaryDirectory()
    dummy_cli.args.output = tempDir.name
    dummy_cli.set_visualizer(tempDir.name)

    # Create features csv file
    features_path = create_csv(features, tempDir.name)

    # Test command
    responses = ["model_age", features_path, "", "", "", "", "", "", "", "", "", "q"]
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split("\n")[:-1]
    assert captured[-1] == 'Finished running age modelling.'

    # Test command with invalid input like incorrect model parameters
    responses = ["model_age", features_path, "", "", "", "", "", "linear_reg fitIntercept=True", "", "", "", "", "q"]
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split("\n")[:-1]
    print(captured)
    assert "Model parameters are not valid for linear_reg model. Check them in the sklearn documentation." in captured
    
    # Test command with hyperparameter optimization and feature_extension
    responses = ["model_age", features_path, "", "", "", "", "", "linear_svr", "", "2", "3", "", "q"]
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split("\n")[:-1]
    assert any(["feature_extension" in i for i in captured])
    assert "Running Hyperparameter optimization..." in captured
    assert any(["Hyperoptimization best parameters" in i for i in captured])


def test_model_command_CLI(dummy_cli):
    """Test dummy CLI model command"""

    # Test no input
    dummy_cli.line = ""
    error = dummy_cli.model_command()
    assert error == "Must provide at least one argument or None."

    # Test using default
    dummy_cli.line = "None"
    error = dummy_cli.model_command()
    assert error is None
    assert dummy_cli.args.model_type == "linear_reg"
    assert dummy_cli.args.model_params == {}

    # Test passing invalid model type
    dummy_cli.line = "quadratic"
    error = dummy_cli.model_command()
    assert error == f"Choose a valid model type: {list(AgeML.model_dict.keys())}"

    # Test empty model params if none given
    dummy_cli.line = "linear_reg"
    error = dummy_cli.model_command()
    assert error is None
    assert dummy_cli.args.model_type == "linear_reg"
    assert dummy_cli.args.model_params == {}

    # Test passing invalid model params
    message = "Model parameters must be in the format param1=value1 param2=value2 ..."
    dummy_cli.line = "linear_reg intercept"
    error = dummy_cli.model_command()
    assert error == message
    dummy_cli.line = "linear_reg intercept==1"
    error = dummy_cli.model_command()
    assert error == message

    # Test passing correct model params
    dummy_cli.line = "linear_reg fit_intercept=True"
    error = dummy_cli.model_command()
    assert error is None
    assert dummy_cli.args.model_type == "linear_reg"
    assert dummy_cli.args.model_params == {"fit_intercept": True}
    
    # Test passing correctly formated, but invalid sklearn model params
    dummy_cli.line = "linear_reg my_super_fake_intercept=True"
    error = dummy_cli.model_command()
    assert error == "Model parameters are not valid for linear_reg model. Check them in the sklearn documentation."

    # Test passing correctly formated, but invalid sklearn model params in another type of model
    dummy_cli.line = "ridge my_super_fake_intercept=True"
    error = dummy_cli.model_command()
    assert error == "Model parameters are not valid for ridge model. Check them in the sklearn documentation."


def test_output_command_CLI(dummy_cli):
    """Test dummy CLI output command"""

    # Test no input
    dummy_cli.line = ""
    error = dummy_cli.output_command()
    assert error == "Must provide a path."

    # Test passing too many arguments
    dummy_cli.line = "path1 path2"
    error = dummy_cli.output_command()
    assert error == "Too many arguments only one single path."

    # Test path exists
    dummy_cli.line = "path"
    error = dummy_cli.output_command()
    assert error == "Directory path does not exist."

    # Test passing correct arguments
    tempDir = tempfile.TemporaryDirectory()
    dummy_cli.line = tempDir.name
    error = dummy_cli.output_command()
    assert error is None
    assert dummy_cli.args.output == tempDir.name


def test_scaler_command_CLI(dummy_cli):
    """Test dummy CLI scaler command"""

    # Test no input
    dummy_cli.line = ""
    error = dummy_cli.scaler_command()
    assert error == "Must provide at least one argument or None."

    # Test using default
    dummy_cli.line = "None"
    error = dummy_cli.scaler_command()
    assert error is None
    assert dummy_cli.args.scaler_type == "standard"
    assert dummy_cli.args.scaler_params == {}

    # Test passing invalid scaler type
    dummy_cli.line = "mofongo"
    error = dummy_cli.scaler_command()
    assert error == f"Choose a valid scaler type: {list(AgeML.scaler_dict.keys())}"

    # Test empty scaler params if none given
    dummy_cli.line = "standard"
    error = dummy_cli.scaler_command()
    assert error is None
    assert dummy_cli.args.scaler_type == "standard"
    assert dummy_cli.args.scaler_params == {}

    # Test passing invalid scaler params
    message = "Scaler parameters must be in the format param1=value1 param2=value2 ..."
    dummy_cli.line = "standard mean==0"
    error = dummy_cli.scaler_command()
    assert error == message
    dummy_cli.line = "standard mean"
    error = dummy_cli.scaler_command()
    assert error == message

    # Test passing correct scaler params
    dummy_cli.line = "standard with_mean=0"
    error = dummy_cli.scaler_command()
    assert error is None
    assert dummy_cli.args.scaler_type == "standard"
    assert dummy_cli.args.scaler_params == {"with_mean": 0}
    
    # Test passing correctly formated, but invalid sklearn scaler params
    dummy_cli.line = "standard my_super_fake_mean=0"
    error = dummy_cli.scaler_command()
    assert error == "Scaler parameters are not valid for standard scaler. Check them in the sklearn documentation."
    
    # Test passing correctly formated, but invalid sklearn scaler params in another type of scaler
    dummy_cli.line = "minmax my_super_fake_mean=0"
    error = dummy_cli.scaler_command()
    assert error == "Scaler parameters are not valid for minmax scaler. Check them in the sklearn documentation."


def test_hyperparameter_tuning_CLI(dummy_cli):
    dummy_cli.args.model_type = "linear_svr"
    dummy_cli.args.model_params = {"C": 1, "epsilon": 0.1}
    
    # Test no hyperparameters
    dummy_cli.line = ""
    error = dummy_cli.hyperparameter_grid_command()
    assert error is None
    assert dummy_cli.args.hyperparameter_tuning == 0
    
    # Test passing too many arguments
    dummy_cli.line = "1 2 3"
    error = dummy_cli.hyperparameter_grid_command()
    assert error == "Must provide only one integer, or none."
    
    # Test passing non integer arguments
    dummy_cli.line = "1.5"
    error = dummy_cli.hyperparameter_grid_command()
    assert error == "The number of points in the hyperparameter grid must be a positive, nonzero integer."
    dummy_cli.line = "mondong"
    error = dummy_cli.hyperparameter_grid_command()
    assert error == "The number of points in the hyperparameter grid must be a positive, nonzero integer."


def test_feature_extension_CLI(dummy_cli):
    # Test no feature extension
    dummy_cli.line = ""
    error = dummy_cli.feature_extension_command()
    assert error is None
    assert dummy_cli.args.feature_extension == 0
    
    # Test passing too many arguments
    dummy_cli.line = "1 2 3"
    error = dummy_cli.feature_extension_command()
    assert error == "Must provide only one integer, or none."
    
    # Test passing non integer arguments
    dummy_cli.line = "1.5"
    error = dummy_cli.feature_extension_command()
    assert error == "The polynomial feature extension degree must be an integer (0, 1, 2, or 3)"
    dummy_cli.line = "mondong"
    error = dummy_cli.feature_extension_command()
    assert error == "The polynomial feature extension degree must be an integer (0, 1, 2, or 3)"
    
    # Test with a correct argument
    dummy_cli.line = "2"
    error = dummy_cli.feature_extension_command()
    assert error is None
    assert dummy_cli.args.feature_extension == 2
