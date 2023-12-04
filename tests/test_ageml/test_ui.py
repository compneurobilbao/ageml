import os
import pytest
import shutil
import sys
import tempfile
import random
import string
import pandas as pd
import numpy as np

import ageml.messages as messages
from ageml.ui import Interface, CLI, InteractiveCLI


class ExampleArguments(object):
    def __init__(self):
        self.scaler_type = "standard"
        self.scaler_params = {"with_mean": True}
        self.model_type = "linear"
        self.model_params = {"fit_intercept": True}
        self.cv_split = 2
        self.seed = 0
        test_path = os.path.dirname(__file__)
        self.output = test_path
        self.features = None
        self.covariates = None
        self.factors = None
        self.clinical = None
        self.ages = None
        self.group1 = None
        self.group2 = None


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
            "predicted age": [55, 67, 57, 75, 85, 64, 87, 93, 49, 51,
                              58, 73, 80, 89, 55, 67, 57, 75, 85, 64],
            "corrected age": [51, 58, 73, 80, 89, 67, 57, 75, 85, 64,
                              87, 93, 49, 55, 67, 57, 75, 85, 64, 87],
            "delta": [1, -2, 3, 0, -1, 2, 1, 0, -3, 1,
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
    """Dummy InteractiveCLI fixture"""

    # Create temporary directory and file
    temp_dir = tempfile.TemporaryDirectory()
    temp_file = tempfile.NamedTemporaryFile(
        dir=temp_dir.name, suffix=".csv", delete=False
    )

    # Define a list of responses
    responses = [temp_dir.name, temp_file.name, "", "", "", "", "", "", "", "", "q"]

    # Patch the input function
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    interface = InteractiveCLI()
    return interface


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


def test_load_csv(dummy_interface, features):
    features_path = create_csv(features, dummy_interface.dir_path)
    data = dummy_interface.load_csv(features_path)

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
    error_message = "Clinical file must contian a column name 'CN' or any other case-insensitive variation."
    assert exc_info.value.args[0] == error_message


def test_load_data_ages_missing_column(dummy_interface, ages):
    # Test removal of columns
    cols = ["age", "predicted age", "corrected age", "delta"]
    for col in cols:
        # Remove column
        df = ages.copy()
        df.drop(col, axis=1, inplace=True)
        file_path = create_csv(df, dummy_interface.dir_path)
        dummy_interface.args.ages = file_path

        # Test error risen
        with pytest.raises(KeyError) as exc_info:
            dummy_interface.load_data()
        assert exc_info.type == KeyError
        error_message = "Ages file must contain a column name %s" % col
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
    assert exc_info.value.args[0] == "Clinical columns must be boolean type."


def test_load_data_ages_warning(dummy_interface, ages):
    ages_path = create_csv(ages, dummy_interface.dir_path)
    dummy_interface.args.ages = ages_path
    with pytest.warns(UserWarning) as warn_record:
        dummy_interface.load_data()
        dummy_interface.load_data()
    assert isinstance(warn_record.list[0].message, UserWarning)
    expected = (
        "Ages file already loaded, overwriting with  %s provided file."
        % dummy_interface.args.ages
    )
    assert warn_record.list[0].message.args[0] == expected


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
    df1 = pd.DataFrame({"age": dist1})
    df2 = pd.DataFrame({"age": dist2})
    with pytest.warns(UserWarning) as warn_record:
        dummy_interface.age_distribution([df1, df2], labels=["dist1", "dist2"])
    assert isinstance(warn_record.list[0].message, UserWarning)
    expected = "Age distributions %s and %s are not similar." % ("dist1", "dist2")
    assert warn_record.list[0].message.args[0] == expected


def test_run_age(dummy_interface, features):
    # Run the modelling pipeline
    features_path = create_csv(features, dummy_interface.dir_path)
    dummy_interface.args.features = features_path
    dummy_interface.run_age()

    # Check for the existence of the output directory
    assert os.path.exists(dummy_interface.dir_path)

    # Check for the existence of the output figures
    figs = [
        "age_bias_correction_all",
        "age_distribution_controls",
        "features_vs_age_controls",
        "chronological_vs_pred_age_all",
    ]
    svg_paths = [
        os.path.join(dummy_interface.dir_path, f"figures/{fig}.svg") for fig in figs
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
    assert all(
        [
            col in df.columns
            for col in ["age", "predicted age", "corrected age", "delta"]
        ]
    )


# TODO: def test_run_age_with_covars(dummy_interface, ages, features, covariates):
def test_run_lifestyle(dummy_interface, ages, factors):
    # Run the lifestyle pipeline
    ages_path = create_csv(ages, dummy_interface.dir_path)
    factors_path = create_csv(factors, dummy_interface.dir_path)
    dummy_interface.args.ages = ages_path
    dummy_interface.args.factors = factors_path
    dummy_interface.run_lifestyle()

    # Check for the existence of the output directory
    assert os.path.exists(dummy_interface.dir_path)

    # Check for the existence of the output figures
    figs = ["factors_vs_deltas"]
    svg_paths = [
        os.path.join(dummy_interface.dir_path, f"figures/{fig}.svg") for fig in figs
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
    figs = ["age_distribution_clinical_groups", "clinical_groups_box_plot"]
    svg_paths = [
        os.path.join(dummy_interface.dir_path, f"figures/{fig}.svg") for fig in figs
    ]
    assert all([os.path.exists(svg_path) for svg_path in svg_paths])

    # Check for the existence of the log
    log_path = os.path.join(dummy_interface.dir_path, "log.txt")
    assert os.path.exists(log_path)


def test_run_lifestyle(dummy_interface, ages, factors):
    # Run the lifestyle pipeline
    ages_path = create_csv(ages, dummy_interface.dir_path)
    factors_path = create_csv(factors, dummy_interface.dir_path)
    dummy_interface.args.ages = ages_path
    dummy_interface.args.factors = factors_path
    dummy_interface.run_lifestyle()

    # Check for the existence of the output directory
    assert os.path.exists(dummy_interface.dir_path)

    # Check for the existence of the output figures
    figs = ["factors_vs_deltas"]
    svg_paths = [
        os.path.join(dummy_interface.dir_path, f"figures/{fig}.svg") for fig in figs
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
    figs = ["age_distribution_clinical_groups", "clinical_groups_box_plot"]
    svg_paths = [
        os.path.join(dummy_interface.dir_path, f"figures/{fig}.svg") for fig in figs
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
    figs = ["roc_curve_cn_vs_group1"]
    svg_paths = [
        os.path.join(dummy_interface.dir_path, f"figures/{fig}.svg") for fig in figs
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


def test_cli_initialization(features, dummy_interface):
    # create features file
    features_data_path = create_csv(features, dummy_interface.dir_path)

    output_path = os.path.dirname(__file__)
    # Path sys.argv (command line arguments)
    # sys.argv[0] should be empty, so we set it to ''
    # TODO: Cleaner way to test CLI?
    sys.argv = [
        "",
        "-f",
        features_data_path,
        "-o",
        output_path,
        "-r",
        "age",
        "--cv",
        "2",
        "1",
    ]
    cli = CLI()

    # Check correct default initialization
    assert cli.args.features == features_data_path
    assert cli.args.model == ["linear"]
    assert cli.args.scaler_type == "standard"
    assert cli.args.cv == [2, 1]


def test_configure_interactiveCLI(dummy_cli):
    """Test dummy InteractiveCLI configured"""
    assert dummy_cli.configFlag is True


def test_get_line_interactiveCLI(dummy_cli, monkeypatch, capsys):
    """Test dummy InteractiveCLI getline"""

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


def test_force_command_interactiveCLI(dummy_cli, monkeypatch):
    """Test dummy InteractiveCLI force command"""

    # Test when no input is given and not required
    monkeypatch.setattr("builtins.input", lambda _: "")
    error = dummy_cli.force_command(
        dummy_cli.load_command, "l --systems", required=False
    )
    assert error is None
    assert dummy_cli.line == ["--systems", "None"]

    # Test when correct input is error returned is None
    monkeypatch.setattr("builtins.input", lambda _: "linear")
    error = dummy_cli.force_command(dummy_cli.model_command, "m", required=True)
    assert error is None


def test_command_interface_interactiveCLI(dummy_cli, monkeypatch, capsys):
    """Test dummy InteractiveCLI command interface"""

    # Test command that does not exist
    responses = ["mnoang", "q"]
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split("\n")[:-1]
    assert captured[-1] == "Invalid command. Enter 'h' for help."

    # Test running run command that shouldn't wokr because dummy not well configured
    responses = ["r age", "q"]
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split("\n")[:-1]
    assert captured[-1] == "Error running modelling."

    # Test running output command
    tempDir = tempfile.TemporaryDirectory()
    responses = ["o " + tempDir.name, "q"]
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split("\n")[:-1]
    assert captured[-1] == "Enter 'h' for help."

    # Test running model command with invalid sklearn inputs
    responses = ["m linear intercept=True", "q"]
    monkeypatch.setattr("builtins.input", lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split("\n")[:-1]
    assert captured[-1] == "Error setting up model."


def test_cv_command_interactiveCLI(dummy_cli):
    """Test dummy InteractiveCLI cv command"""

    # Test no input
    dummy_cli.line = "cv "
    error = dummy_cli.cv_command()
    assert error == "Must provide at least one argument or None."

    # Test default values
    dummy_cli.line = "cv None"
    error = dummy_cli.cv_command()
    assert error is None
    assert dummy_cli.args.cv_split == 5
    assert dummy_cli.args.seed == 0

    # Test non-integer values
    dummy_cli.line = "cv 2.5"
    error = dummy_cli.cv_command()
    assert error == "CV parameters must be integers"
    dummy_cli.line = "cv 2 3.5"
    error = dummy_cli.cv_command()
    assert error == "CV parameters must be integers"

    # Test passing too many arguments
    dummy_cli.line = "cv 1 2 3"
    error = dummy_cli.cv_command()
    assert error == "Too many values to unpack."

    # Test correct parsing
    dummy_cli.line = "cv 1"
    error = dummy_cli.cv_command()
    assert error is None
    assert dummy_cli.args.cv_split == 1
    assert dummy_cli.args.seed == 0
    dummy_cli.line = "cv 1 2"
    error = dummy_cli.cv_command()
    assert error is None
    assert dummy_cli.args.cv_split == 1
    assert dummy_cli.args.seed == 2


def test_help_command_interactiveCLI(dummy_cli, capsys):
    """Test dummy InteractiveCLI help command"""
    dummy_cli.help_command()
    captured = capsys.readouterr().out.split("\n")
    assert captured[0] == "User commands:"
    assert captured[1] == messages.cv_command_message
    assert captured[2] == messages.help_command_message
    assert captured[3] == messages.load_command_message
    assert captured[4] == messages.model_command_message
    assert captured[5] == messages.output_command_message
    assert captured[6] == messages.quit_command_message
    assert captured[7] == messages.run_command_message
    assert captured[8] == messages.scaler_command_message


def test_load_command_interactiveCLI(dummy_cli):
    """Test dummy InteractiveCLI load command"""

    # Test no input
    dummy_cli.line = "l"
    error = dummy_cli.load_command()
    assert error == "Must provide a file type and file path."

    # Test passing only one input
    dummy_cli.line = "l --features"
    error = dummy_cli.load_command()
    assert error == "Must provide a file path or None when using --file_type."

    # Test passing too many arguments
    dummy_cli.line = "l --features file1 file2"
    error = dummy_cli.load_command()
    assert error == "Too many arguments only two arguments --file_type and file path."

    # Test passing non existant file type
    dummy_cli.line = "l --features file1"
    error = dummy_cli.load_command()
    assert error == "File file1 not found."

    # Create a temporary file
    tmpcsv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmptxt = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)

    # Test passing incorrect file type
    dummy_cli.line = "l --features " + tmptxt.name
    error = dummy_cli.load_command()
    assert error == "File %s must be a .csv file." % tmptxt.name
    dummy_cli.line = "l --systems " + tmpcsv.name
    error = dummy_cli.load_command()
    assert error == "File %s must be a .txt file." % tmpcsv.name

    # Test choosing invalid file type
    dummy_cli.line = "l --flag " + tmpcsv.name
    error = dummy_cli.load_command()
    error_message = "Choose a valid file type: --features, --covariates, --factors, --clinical, --systems, --ages"
    assert error == error_message

    # Test passing correct arguments
    dummy_cli.line = "l --features " + tmpcsv.name
    error = dummy_cli.load_command()
    assert error is None
    assert dummy_cli.args.features == tmpcsv.name


def test_model_command_interactiveCLI(dummy_cli):
    """Test dummy InteractiveCLI model command"""

    # Test no input
    dummy_cli.line = "m"
    error = dummy_cli.model_command()
    assert error == "Must provide at least one argument or None."

    # Test using default
    dummy_cli.line = "m None"
    error = dummy_cli.model_command()
    assert error is None
    assert dummy_cli.args.model_type == "linear"
    assert dummy_cli.args.model_params == {}

    # Test passing invalid model type
    dummy_cli.line = "m quadratic"
    error = dummy_cli.model_command()
    assert error == "Choose a valid model type: {}".format(["linear"])

    # Test empty model params if none given
    dummy_cli.line = "m linear"
    error = dummy_cli.model_command()
    assert error is None
    assert dummy_cli.args.model_type == "linear"
    assert dummy_cli.args.model_params == {}

    # Test passing invalid model params
    message = "Model parameters must be in the format param1=value1 param2=value2 ..."
    dummy_cli.line = "m linear intercept"
    error = dummy_cli.model_command()
    assert error == message
    dummy_cli.line = "m linear intercept==1"
    error = dummy_cli.model_command()
    assert error == message

    # Test passing correct model params
    dummy_cli.line = "m linear fit_intercept=True"
    error = dummy_cli.model_command()
    assert error is None
    assert dummy_cli.args.model_type == "linear"
    assert dummy_cli.args.model_params == {"fit_intercept": True}


def test_output_command_interactiveCLI(dummy_cli):
    """Test dummy InteractiveCLI output command"""

    # Test no input
    dummy_cli.line = "o"
    error = dummy_cli.output_command()
    assert error == "Must provide a path."

    # Test passing too many arguments
    dummy_cli.line = "o path1 path2"
    error = dummy_cli.output_command()
    assert error == "Too many arguments only one single path."

    # Test path exists
    dummy_cli.line = "o path"
    error = dummy_cli.output_command()
    assert error == "Directory path does not exist."

    # Test passing correct arguments
    tempDir = tempfile.TemporaryDirectory()
    dummy_cli.line = "o " + tempDir.name
    error = dummy_cli.output_command()
    assert error is None
    assert dummy_cli.args.output == tempDir.name


def test_run_command_interactiveCLI(dummy_cli):
    """Test dummy InteractiveCLI run command"""

    # Test no input or mutiple arguments
    dummy_cli.line = "r"
    error = dummy_cli.run_command()
    assert error == "Must provide one argument only."
    dummy_cli.line = "r type1 type1"
    error = dummy_cli.run_command()
    assert error == "Must provide one argument only."

    # Test passing invalid run type
    dummy_cli.line = "r type1"
    error = dummy_cli.run_command()
    assert error == "Choose a valid run type: age, lifestyle, clinical, classification"

    # Test passing run type with more arguments than required
    dummy_cli.line = "r age 1"
    error = dummy_cli.run_command()
    assert error == "Too many arguments given for run type age"

    # Test passing run type with less arguments than required
    dummy_cli.line = "r classification 1 2 3"
    error = dummy_cli.run_command()
    assert error == "For run type classification two arguments should be given"

    # Test passing run type with more arguments than required
    dummy_cli.line = "r age 1"
    error = dummy_cli.run_command()
    assert error == "Too many arguments given for run type age"

    # Test passing run type with less arguments than required
    dummy_cli.line = "r classification 1 2 3"
    error = dummy_cli.run_command()
    assert error == "For run type classification two arguments should be given"

    # Test passing correct arguments
    dummy_cli.line = "r age"
    error = dummy_cli.run_command()
    assert error is None
    assert dummy_cli.run == dummy_cli.run_age


def test_scaler_command_interactiveCLI(dummy_cli):
    """Test dummy InteractiveCLI scaler command"""

    # Test no input
    dummy_cli.line = "s"
    error = dummy_cli.scaler_command()
    assert error == "Must provide at least one argument or None."

    # Test using default
    dummy_cli.line = "s None"
    error = dummy_cli.scaler_command()
    assert error is None
    assert dummy_cli.args.scaler_type == "standard"
    assert dummy_cli.args.scaler_params == {}

    # Test passing invalid scaler type
    dummy_cli.line = "s minmax"
    error = dummy_cli.scaler_command()
    assert error == "Choose a valid scaler type: {}".format(["standard"])

    # Test empty scaler params if none given
    dummy_cli.line = "s standard"
    error = dummy_cli.scaler_command()
    assert error is None
    assert dummy_cli.args.scaler_type == "standard"
    assert dummy_cli.args.scaler_params == {}

    # Test passing invalid scaler params
    message = "Scaler parameters must be in the format param1=value1 param2=value2 ..."
    dummy_cli.line = "s standard mean==0"
    error = dummy_cli.scaler_command()
    assert error == message
    dummy_cli.line = "s standard mean"
    error = dummy_cli.scaler_command()
    assert error == message

    # Test passing correct scaler params
    dummy_cli.line = "s standard mean=0"
    error = dummy_cli.scaler_command()
    assert error is None
    assert dummy_cli.args.scaler_type == "standard"
    assert dummy_cli.args.scaler_params == {"mean": 0}
