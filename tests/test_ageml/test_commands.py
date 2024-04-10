import os
import pandas as pd
import pytest
import random
import string
import sys
import tempfile

from ageml.commands import (
    model_age,
    factor_correlation,
    clinical_groups,
    clinical_classify,
)


# Fake data for testing
@pytest.fixture
def features():
    df = pd.DataFrame(
        {
            "id": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
            ],
            "age": [
                50,
                55,
                60,
                65,
                70,
                75,
                80,
                85,
                90,
                57,
                53,
                57,
                61,
                65,
                69,
                73,
                77,
                81,
                85,
                89,
            ],
            "feature1": [
                1.3,
                2.2,
                3.9,
                4.1,
                5.7,
                6.4,
                7.5,
                8.2,
                9.4,
                1.7,
                1.4,
                2.2,
                3.8,
                4.5,
                5.4,
                6.2,
                7.8,
                8.2,
                9.2,
                2.6,
            ],
            "feature2": [
                9.4,
                8.2,
                7.5,
                6.4,
                5.3,
                4.1,
                3.9,
                2.2,
                1.3,
                9.4,
                9.3,
                8.1,
                7.9,
                6.5,
                5.0,
                4.0,
                3.7,
                2.1,
                1.4,
                8.3,
            ],
        }
    )
    df.set_index("id", inplace=True)
    return df


@pytest.fixture
def factors():
    df = pd.DataFrame(
        {
            "id": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
            ],
            "factor1": [
                1.3,
                2.2,
                3.9,
                4.1,
                5.7,
                6.4,
                7.5,
                8.2,
                9.4,
                1.3,
                1.3,
                2.2,
                3.9,
                4.1,
                5.7,
                6.4,
                7.5,
                8.2,
                9.4,
                2.2,
            ],
            "factor2": [
                0.1,
                1.3,
                2.2,
                3.9,
                4.1,
                5.7,
                6.4,
                7.5,
                8.2,
                9.4,
                4.7,
                3.7,
                2.3,
                1.2,
                0.9,
                0.3,
                0.2,
                0.1,
                0.1,
                0.1,
            ],
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
            "cn": [True, False, True, False, True, False, True, False, True, False,
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


def create_csv(df, path):
    # Generate random name for the csv file
    letters = string.ascii_lowercase
    csv_name = "".join(random.choice(letters) for i in range(20)) + ".csv"
    file_path = os.path.join(path, csv_name)
    df.to_csv(path_or_buf=file_path, index=True)
    return file_path


# Create temporary directory
@pytest.fixture
def temp_dir():
    return tempfile.TemporaryDirectory()


def test_model_age(temp_dir, features):
    """Test model_age function."""

    # Create features file
    features_data_path = create_csv(features, temp_dir.name)

    # Create systems arguments
    sys.argv = [
        "",
        "-o",
        temp_dir.name,
        "-f",
        features_data_path,
        "-m",
        "linear_reg",
        "fit_intercept=True",
        "-s",
        "standard",
        "--cv",
        "5",
        "0",
    ]

    # Run function
    model_age()


def test_factor_correlation(temp_dir, ages, factors):
    """Test factor_correlation function."""

    # Create features file
    age_data_path = create_csv(ages, temp_dir.name)
    factors_data_path = create_csv(factors, temp_dir.name)

    # Create systems arguments
    sys.argv = ["", "-o", temp_dir.name, "-a", age_data_path, "-f", factors_data_path]

    # Run function
    factor_correlation()


def test_clinical_groups(temp_dir, ages, clinical):
    """Test clinical_groups function."""

    # Create features file
    age_data_path = create_csv(ages, temp_dir.name)
    clinical_data_path = create_csv(clinical, temp_dir.name)

    # Create systems arguments
    sys.argv = [
        "",
        "-o",
        temp_dir.name,
        "-a",
        age_data_path,
        "--clinical",
        clinical_data_path,
    ]

    # Run function
    clinical_groups()


def test_clinical_classify(temp_dir, ages, clinical):
    """Test age_deltas function."""

    # Create features file
    age_data_path = create_csv(ages, temp_dir.name)
    clinical_data_path = create_csv(clinical, temp_dir.name)

    # Create systems arguments
    sys.argv = [
        "",
        "-o",
        temp_dir.name,
        "-a",
        age_data_path,
        "--clinical",
        clinical_data_path,
        "--groups",
        "CN",
        "group1",
    ]

    # Run function
    clinical_classify()
