import pytest
import os
import shutil
import ageml.utils as utils


class InstanceClass(object):
    """
    Dummy class to test the ageml.utils.log decorator.
    The decorator needs an 'instance' input object, and this
    input object must have a .log_path attribute, ideally a
    path.
    """
    def __init__(self, log_path):
        self.log_path = log_path


def test_log_decorator():
    @utils.log
    def func_to_log(instance, to_print):
        print(to_print)

    # Define instance and print message
    dummy_path = os.path.join(os.getcwd(), 'test_log_mondongo')
    instance_dummy = InstanceClass(dummy_path)
    to_print = 'Mondongo for the win'
    
    # Just in case, remove the log file from possible unsuccessful test run that might have left it undeleted.
    if os.path.exists(dummy_path):
        os.remove(dummy_path)
    func_to_log(instance_dummy, to_print)
    
    # Assert that the log exists
    assert os.path.exists(instance_dummy.log_path)
    # Assert that the log content is what it is expected
    with open(instance_dummy.log_path, 'r') as f:
        assert f.readline() == to_print + "\n"
    # Cleanup of log file
    os.remove(dummy_path)


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
