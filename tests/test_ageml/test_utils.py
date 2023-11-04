import pytest
import os
import shutil
import ageml.utils as utils


def func_to_log():
    print('Mondong!')


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


def test_log_decorator():
    pass


def test_insert_newlines():
    text = "Insert the newlines before this and before this!"
    expected = "Insert the newlines\nbefore this and\nbefore this!"
    nwords = 3

    assert utils.insert_newlines(text, nwords) == expected


def test_insert_newlines_zerospacing():
    text = "This will not even be processed."
    with pytest.raises(ValueError) as e:
        utils.insert_newlines(text, 0)
    assert e.type == ValueError
