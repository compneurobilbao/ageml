import pytest
import os
import shutil
import ageml.utils as utils


@pytest.mark.parametrize("input, expected", [("true", True), ("false", False), ("420.69", 420.69), ("mondongo", "mondongo")])
def test_convert_value(input, expected):
    assert utils.convert(input) == expected


def test_create_directory():
    path = os.path.join(os.getcwd(), "temp_folder")
    print(path)
    # Create the directory
    utils.create_directory(path)
    # Existance Assertion
    assert os.path.exists(path)
    # Delete after
    shutil.rmtree(path)
