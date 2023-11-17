import pytest
import tempfile

from ageml.ui import InteractiveCLI
from ageml.messages import *

@pytest.fixture
def monkeypatch():
    """Monkeypatch fixture"""
    return pytest.MonkeyPatch()

@pytest.fixture
def dummy_cli(monkeypatch):
    """Dummy InteractiveCLI fixture"""

    # Create temporary directory and file
    temp_dir = tempfile.TemporaryDirectory()
    temp_file = tempfile.NamedTemporaryFile(dir=temp_dir.name, suffix='.csv', delete=False)

    # Define a list of responses
    responses = [temp_dir.name, temp_file.name, '', '', '', '', '', '', '', 'q']
    
    # Patch the input function
    monkeypatch.setattr('builtins.input', lambda _: responses.pop(0))
    interface = InteractiveCLI()
    return interface

def test_configure_interactiveCLI(dummy_cli):
    """Test dummy InteractiveCLI configured"""
    assert dummy_cli.configFlag == True

def test_get_line_interactiveCLI(dummy_cli, monkeypatch, capsys):
    """Test dummy InteractiveCLI getline"""

    # Test that without required can pass empty string
    monkeypatch.setattr('builtins.input', lambda _:'')
    dummy_cli.get_line(required=False)
    assert dummy_cli.line == ''

    # Test that with required cannot pass empty string
    responses = ['', 'asdf']
    monkeypatch.setattr('builtins.input', lambda _: responses.pop(0))
    dummy_cli.get_line(required=True)
    captured = capsys.readouterr()
    assert captured.out == 'Must provide a value.\n'
    assert dummy_cli.line == 'asdf'

def test_force_command_interactiveCLI(dummy_cli, monkeypatch):
    """Test dummy InteractiveCLI force command"""

    # Test when no input is given and not required
    monkeypatch.setattr('builtins.input', lambda _:'')
    error = dummy_cli.force_command(dummy_cli.load_command, 'l --systems', required=False)
    assert error == None
    assert dummy_cli.line == ['--systems', 'None']

    # Test when correct input is error returned is None
    monkeypatch.setattr('builtins.input', lambda _:'linear')
    error = dummy_cli.force_command(dummy_cli.model_command, 'm', required=True)
    assert error == None

def test_command_interface_interactiveCLI(dummy_cli, monkeypatch, capsys):
    """Test dummy InteractiveCLI command interface"""

    # Test command that does not exist
    responses = ['asdf', 'q']
    monkeypatch.setattr('builtins.input', lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split('\n')[:-1]
    assert captured[-1] == "Invalid command. Enter 'h' for help."

    # Test running run command that shouldn't wokr because dummy not well configured
    responses = ['r age', 'q']
    monkeypatch.setattr('builtins.input', lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split('\n')[:-1]
    assert captured[-1] == 'Error running modelling.'

    # Test running output command
    tempDir = tempfile.TemporaryDirectory()
    responses = ['o ' + tempDir.name, 'q']
    monkeypatch.setattr('builtins.input', lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split('\n')[:-1]
    assert captured[-1] == "Enter 'h' for help."

    # Test running model command with invalid sklearn inputs
    responses = ['m linear intercept=True', 'q']
    monkeypatch.setattr('builtins.input', lambda _: responses.pop(0))
    dummy_cli.command_interface()
    captured = capsys.readouterr().out.split('\n')[:-1]
    assert captured[-1] == 'Error setting up model.'


def test_cv_command_interactiveCLI(dummy_cli):
    """Test dummy InteractiveCLI cv command"""

    # Test no input
    dummy_cli.line = 'cv '
    error = dummy_cli.cv_command()
    assert error == 'Must provide at least one argument or None.'

    # Test default values
    dummy_cli.line = 'cv None'
    error = dummy_cli.cv_command()
    assert error == None
    assert dummy_cli.args.cv_split == 5
    assert dummy_cli.args.seed == 0

    # Test non-integer values
    dummy_cli.line = 'cv 2.5'
    error = dummy_cli.cv_command()
    assert error == 'CV parameters must be integers'
    dummy_cli.line = 'cv 2 3.5'
    error = dummy_cli.cv_command()
    assert error == 'CV parameters must be integers'

    # Test passing too many arguments
    dummy_cli.line = 'cv 1 2 3'
    error = dummy_cli.cv_command()
    assert error == 'Too many values to unpack.'

    # Test correct parsing
    dummy_cli.line = 'cv 1'
    error = dummy_cli.cv_command()
    assert error == None
    assert dummy_cli.args.cv_split == 1
    assert dummy_cli.args.seed == 0
    dummy_cli.line = 'cv 1 2'
    error = dummy_cli.cv_command()
    assert error == None
    assert dummy_cli.args.cv_split == 1
    assert dummy_cli.args.seed == 2

def test_help_command_interactiveCLI(dummy_cli, capsys):
    """Test dummy InteractiveCLI help command"""
    dummy_cli.help_command()
    captured = capsys.readouterr().out.split('\n')
    assert captured[0] == 'User commands:'
    assert captured[1] == cv_command_message
    assert captured[2] == help_command_message
    assert captured[3] == load_command_message
    assert captured[4] == model_command_message
    assert captured[5] == output_command_message
    assert captured[6] == quit_command_message
    assert captured[7] == run_command_message
    assert captured[8] == scaler_command_message

def test_load_command_interactiveCLI(dummy_cli):
    """Test dummy InteractiveCLI load command"""

    # Test no input
    dummy_cli.line = 'l'
    error = dummy_cli.load_command()
    assert error == 'Must provide a file type and file path.'

    # Test passing only one input
    dummy_cli.line = 'l --features'
    error = dummy_cli.load_command()
    assert error == 'Must provide a file path or None when using --file_type.'

    # Test passing too many arguments
    dummy_cli.line = 'l --features file1 file2'
    error = dummy_cli.load_command()
    assert error == 'Too many arguments only two arguments --file_type and file path.'

    # Test passing non existant file type
    dummy_cli.line = 'l --features file1'
    error = dummy_cli.load_command()
    assert error == 'File file1 not found.'

    # Create a temporary file
    tmpcsv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    tmptxt = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)

    # Test passing incorrect file type
    dummy_cli.line = 'l --features ' + tmptxt.name
    error = dummy_cli.load_command()
    assert error == 'File %s must be a .csv file.' % tmptxt.name
    dummy_cli.line = 'l --systems ' + tmpcsv.name
    error = dummy_cli.load_command()
    assert error == 'File %s must be a .txt file.' % tmpcsv.name

    # Test passing None to required file
    dummy_cli.line = 'l --features None'
    error = dummy_cli.load_command()
    assert error == 'A features file must be provided must not be None.'

    # Test choosing invalid file type
    dummy_cli.line = 'l --flag ' + tmpcsv.name
    error = dummy_cli.load_command()
    assert error == 'Choose a valid file type: --features, --covariates, --factors, --clinical, --systems'

    # Test passing correct arguments
    dummy_cli.line = 'l --features ' + tmpcsv.name
    error = dummy_cli.load_command()
    assert error == None
    assert dummy_cli.args.features == tmpcsv.name

def test_model_command_interactiveCLI(dummy_cli):
    """Test dummy InteractiveCLI model command"""

    # Test no input
    dummy_cli.line = 'm'
    error = dummy_cli.model_command()
    assert error == 'Must provide at least one argument or None.'

    # Test using default
    dummy_cli.line = 'm None'
    error = dummy_cli.model_command()
    assert error == None
    assert dummy_cli.args.model_type == 'linear'
    assert dummy_cli.args.model_params == {}

    # Test passing invalid model type
    dummy_cli.line = 'm quadratic'
    error = dummy_cli.model_command()
    assert error == 'Choose a valid model type: {}'.format(['linear'])

    # Test empty model params if none given
    dummy_cli.line = 'm linear'
    error = dummy_cli.model_command()
    assert error == None
    assert dummy_cli.args.model_type == 'linear'
    assert dummy_cli.args.model_params == {}

    # Test passing invalid model params
    message = 'Model parameters must be in the format param1=value1 param2=value2 ...'
    dummy_cli.line = 'm linear intercept'
    error = dummy_cli.model_command()
    assert error == message
    dummy_cli.line = 'm linear intercept==1'
    error = dummy_cli.model_command()
    assert error == message

    # Test passing correct model params
    dummy_cli.line = 'm linear fit_intercept=True'
    error = dummy_cli.model_command()
    assert error == None
    assert dummy_cli.args.model_type == 'linear'
    assert dummy_cli.args.model_params == {'fit_intercept': True}

def test_output_command_interactiveCLI(dummy_cli):
    """Test dummy InteractiveCLI output command"""

    # Test no input
    dummy_cli.line = 'o'
    error = dummy_cli.output_command()
    assert error == 'Must provide a path.'

    # Test passing too many arguments
    dummy_cli.line = 'o path1 path2'
    error = dummy_cli.output_command()
    assert error == 'Too many arguments only one single path.'

    # Test path exists
    dummy_cli.line = 'o path'
    error = dummy_cli.output_command()
    assert error == 'Directory path does not exist.'

    # Test passing correct arguments
    tempDir = tempfile.TemporaryDirectory()
    dummy_cli.line = 'o ' + tempDir.name
    error = dummy_cli.output_command()
    assert error == None
    assert dummy_cli.args.output == tempDir.name

def test_run_command_interactiveCLI(dummy_cli):
    """Test dummy InteractiveCLI run command"""

    # Test no input or mutiple arguments
    dummy_cli.line = 'r'
    error = dummy_cli.run_command()
    assert error == 'Must provide one argument only.' 
    dummy_cli.line = 'r type1 type1'
    error = dummy_cli.run_command()
    assert error == 'Must provide one argument only.'

    # Test passing invalid run type
    dummy_cli.line = 'r type1'
    error = dummy_cli.run_command()
    assert error == 'Choose a valid run type: age, lifestyle, clinical, classification'

    # Test passing correct arguments
    dummy_cli.line = 'r age'
    error = dummy_cli.run_command()
    assert error == None
    assert dummy_cli.run == dummy_cli.run_age


def test_scaler_command_interactiveCLI(dummy_cli):
    """Test dummy InteractiveCLI scaler command"""

    # Test no input
    dummy_cli.line = 's'
    error = dummy_cli.scaler_command()
    assert error == 'Must provide at least one argument or None.'

    # Test using default
    dummy_cli.line = 's None'
    error = dummy_cli.scaler_command()
    assert error == None
    assert dummy_cli.args.scaler_type == 'standard'
    assert dummy_cli.args.scaler_params == {}

    # Test passing invalid scaler type
    dummy_cli.line = 's minmax'
    error = dummy_cli.scaler_command()
    assert error == 'Choose a valid scaler type: {}'.format(['standard'])

    # Test empty scaler params if none given
    dummy_cli.line = 's standard'
    error = dummy_cli.scaler_command()
    assert error == None
    assert dummy_cli.args.scaler_type == 'standard'
    assert dummy_cli.args.scaler_params == {}

    # Test passing invalid scaler params
    message = 'Scaler parameters must be in the format param1=value1 param2=value2 ...'
    dummy_cli.line = 's standard mean==0'
    error = dummy_cli.scaler_command()
    assert error == message
    dummy_cli.line = 's standard mean'
    error = dummy_cli.scaler_command()
    assert error == message

    # Test passing correct scaler params
    dummy_cli.line = 's standard mean=0'
    error = dummy_cli.scaler_command()
    assert error == None
    assert dummy_cli.args.scaler_type == 'standard'
    assert dummy_cli.args.scaler_params == {'mean': 0}