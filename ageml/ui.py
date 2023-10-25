"""Implement the user interface.

Used in the AgeML project to enable the user to enter commands
to run the modelling with desired inputs.

Classes:
--------
Interface - reads, parses and executes user commands.
CLI - reads and parsers user commands via command line.
"""

import argparse
import numpy as np
import pandas as pd
import os
import warnings

from datetime import datetime

from .visualizer import Visualizer
from .utils import create_directory, convert, log
from .modelling import AgeML

class Interface:

    """Reads, parses and executes user commands.

    This class allows the user to enter certain commands.
    These commands enable the user to run the modellin with the selected
    input files and parameters.

    Parameters
    -----------
    args: arguments with which to run the modelling

    Public methods:
    ---------------
    run(self): Runs the age modelling interface.
    """

    def __init__(self, args):
        """Initialise variables."""

        # Arguments with which to run modelling
        self.args = args

        # Set up directory for storage of results
        self._setup()

        # Initialise objects form library
        self.visualizer = Visualizer(self.dir_path)
        self.ageml = AgeML(self.args.scaler_type, self.args.scaler_params,
                           self.args.model_type, self.args.model_params,
                           self.args.cv_split, self.args.seed)

    def _setup(self):
        """Create required directories and files to store results."""

        # Create directories
        self.dir_path = os.path.join(self.args.output, 'ageml')
        if os.path.exists(self.dir_path):
            warnings.warn("Directory %s already exists files may be overwritten." %
                          self.dir_path)
        create_directory(self.dir_path)
        create_directory(os.path.join(self.dir_path,'figures'))

        # Create .txt log file and log time
        self.log_path = os.path.join(self.dir_path, 'log.txt')
        with open(self.log_path, "a") as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(current_time + '\n')

    def _load_csv(self, file):
        """Use panda to load csv into dataframe.

        Parameters
        ----------
        file: path to file; must be .csv
        """

        if file is not None:
            df = pd.read_csv(file, header=0, index_col=0)
            df.columns = df.columns.str.lower() # ensure lower case
            return df
        else:
            return None

    @log
    def _load_data(self):
        """Load data from csv files."""

        # Load data 
        self.df_features = self._load_csv(self.args.features)
        self.df_covariates = self._load_csv(self.args.covariates)
        self.df_factors = self._load_csv(self.args.factors)
        self.df_clinical = self._load_csv(self.args.clinical)

        # Remove subjects with missing features
        subjects_missing_data = self.df_features[self.df_features.isnull().any(axis=1)].index
        if subjects_missing_data is not None:
            print('-----------------------------------')
            print('Subjects with missing data: %s' % subjects_missing_data.to_list())
            warnings.warn('Subjects with missing data: %s' % subjects_missing_data)
        self.df_features.dropna(inplace=True)

    @log
    def _age_distribution(self):
        """Use visualizer to show age distribution."""

        # Select age information
        ages = self.df_features['age'].to_numpy()

        # Use visualizer to show age distribution
        self.visualizer.age_distribution(ages)

    @log
    def _features_vs_age(self):
        """Use visualizer to explore relationship between features and age."""

        # Select data to visualize
        feature_names = [name for name in self.df_features.columns
                         if name != 'age']
        X = self.df_features[feature_names].to_numpy()
        Y = self.df_features['age'].to_numpy()

        # Use visualizer to show 
        self.visualizer.features_vs_age(X, Y, feature_names)

    @log
    def _model_age(self):
        """Use AgeML to fit age model with data."""

        # Show training pipeline
        print('-----------------------------------')
        print('Training Age Model')
        print(self.ageml.pipeline)

        # Select data to model
        feature_names = [name for name in self.df_features.columns
                         if name != 'age']
        X = self.df_features[feature_names].to_numpy()
        y = self.df_features['age'].to_numpy()

        # Fit model and plot results
        y_pred, y_corrected = self.ageml.fit_age(X, y)
        self.visualizer.true_vs_pred_age(y, y_pred)
        self.visualizer.age_bias_correction(y, y_pred, y_corrected)

        # Save to dataframe and csv
        data = np.stack((y, y_pred, y_corrected), axis=1)
        cols = ['Age', 'Predicted Age', 'Corrected Age']
        self.df_age = df = pd.DataFrame(data, index=self.df_features.index, columns=cols)
        self.df_age.to_csv(os.path.join(self.dir_path, 'predicted_age.csv'))

    def run(self):
        """Read the command entered and call the corresponding functions"""

        # Load data
        self._load_data()

        # Distribution of ages
        self._age_distribution()

        # Relationship between features and age
        self._features_vs_age()

        # Model age
        self._model_age()

class CLI(Interface):

    """Read and parses user commands via command line."""

    def __init__(self):
        """Initialise variables."""
        self.parser = argparse.ArgumentParser(description="Age Modelling using python.",
                                              formatter_class=argparse.RawTextHelpFormatter)
        self._configure_parser()
        args = self.parser.parse_args()
        args = self._configure_args(args)
        super().__init__(args)

    def _configure_parser(self):
        """Configure parser with required arguments for processing."""
        self.parser.add_argument('-o', '--output', metavar='DIR', required=True,
                                 help="Path to output directory where to save results. (Required)")
        self.parser.add_argument('-f', "--features", metavar='FILE',
                                 help="Path to input CSV file containing features. (Required) \n"
                                      "In the file the first column should be the ID, the second column should be the AGE, \n"
                                      "and the following columns the features. The first row should be the header for \n"
                                      "column names.", required=True)
        self.parser.add_argument('-m', '--model', nargs='*', default=['linear'],
                                 help='Model type and model parameters to use. First argument is the type and the following \n'
                                      'arguments are input as keyword arguments into the model. They must be seperated by an =.\n'
                                      'Example: -m linear fit_intercept=False\n'
                                      'Available Types: linear (Default: linear)')
        self.parser.add_argument('-s', '--scaler', nargs='*', default=['standard'],
                                 help='Scaler type and scaler parameters to use. First argument is the type and the following \n'
                                      'arguments are input as keyword arguments into the scaler. They must be seperated by an =.\n'
                                      'Example: -m standard\n'
                                      'Available Types: standard (Default: standard)')
        self.parser.add_argument('--cv', nargs='+', type=int, default=[5, 0],
                                 help='Number of CV splits with which to run the Cross Validation Scheme. Expect 1 or 2 integers. \n'
                                      'First integer is the number of splits and the second is the seed for randomization. \n'
                                      'Default: 5 0')
        self.parser.add_argument("--covariates", metavar='FILE',
                                 help="Path to input CSV file containing covariates. \n"
                                      "In the file the first column should be the ID, the followins columns should be the \n"
                                      "covariates. The first row should be the header for column names.")
        self.parser.add_argument("--factors", metavar='FILE',
                                 help="Path to input CSV file containing factors (e.g. liefstyle and environmental factors). \n"
                                      "In the file the first column should be the ID, the followins columns should be the \n"
                                      "factors. The first row should be the header for column names.")
        self.parser.add_argument("--clinical", metavar='FILE',
                                 help="Path to input CSV file containing health conditions. \n"
                                      "In the file the first column should be the ID, the second column should be wether the \n"
                                      "subject is a CONTROL and the following columns are binary variables for different \n"
                                      "conditions. The first row should be the header for column names.")
        self.parser.add_argument("--systems", metavar='FILE',
                                 help="Path to input .txt file containing the features to use to model each system. \n"
                                      "Each new line corresponds to a different system. The parser follows a formatting \n"
                                      "where the first words in the line is the system name followed by a colon and then the \n"
                                      "names of the features seperated by commas. [SystemName]: [Feature1], [Feature2], ... \n"
                                      "(e.g. Brain Structure: White Matter Volume, Grey Matter Volume, VCSF Volume)")

    def _configure_args(self, args):
        """Configure argumens with required fromatting for modelling.

        Parameters
        ----------
        args: arguments object from parser
        """

        # Set CV params first item is the number of CV splits
        if len(args.cv) == 1:
            args.cv_split = args.cv[0]
            args.seed = self.parser.get_default('cv')[1]
        elif len(args.cv) == 2:
            args.cv_split, args.seed = args.cv
        else:
            raise ValueError('Too many values to unpack')

        # Set Scaler parameters first item is the scaler type
        # The rest of the arguments conform a dictionary for **kwargs
        args.scaler_type = args.scaler[0]
        if len(args.scaler) > 1:
            scaler_params = {}
            for item in args.scaler[1:]:
                key, value = item.split('=')
                value = convert(value)
                scaler_params[key] = value
            args.scaler_params = scaler_params
        else:
            args.scaler_params = {}

        # Set Model parameters first item is the model type
        # The rest of the arguments conform a dictionary for **kwargs
        args.model_type = args.model[0]
        if len(args.model) > 1:
            model_params = {}
            for item in args.model[1:]:
                key, value = item.split('=')
                value = convert(value)
                model_params[key] = value
            args.model_params = model_params
        else:
            args.model_params = {}

        return args

class InteractiveCLI(Interface):

    """Read and parses user commands via command line via an interactive interface
    
    Public methods:
    ---------------
    command_interface(self): Reads in the commands and calls the corresponding
                             functions.

    get_line(self): Prints a prompt for the user and updates the user entry.

    read_command(self): Returns the first non-whitespace character.

    get_character(self): Moves the cursor forward by one character in the user
                         entry.

    skip_spaces(self): Skips whitespace characters until a non-whitespace
                       character is reached.

    cv_command(self): Loads CV parameters.

    help_command(self): Prints a list of valid commands.

    load_command(self): Loads file paths.

    model_command(self): Loads model parameters.

    output_command(self): Loads output directory.

    run_command(self): Runs the modelling.

    scaler_command(self): Loads scaler parameters.
    """

    def __init__(self):
        """Initialise variables."""

        # Interactive setup
        self.character = ""  # current character
        self.line = ""  # current string entered by the user
        self.cursor = 0  # cursor position

        # Ask for required inputs
        self.args = argparse.Namespace()
        print("Age Modelling (AgeML): interactive command line user interface.")
        print('-----------------------------------')
        print('Setup (for Optional or Default values enter: None)')
        
        # Askf for output directory
        print('Output directory path (Required):')
        self.get_line()
        self.output_command()

        # Ask for input files
        print('Input features file path (Required):')
        self.get_line()
        self.line = '--features ' + self.line
        self.load_command()
        print('Input covariates file path (Optional):')
        self.get_line()
        self.line = '--covariates ' + self.line
        self.load_command()
        print('Input factors file path (Optional):')
        self.get_line()
        self.line = '--factors ' + self.line
        self.load_command()
        print('Input clinical file path (Optional):')
        self.get_line()
        self.line = '--clinical ' + self.line
        self.load_command()
        print('Input systems file path (Optional):')
        self.get_line()
        self.line = '--clinical ' + self.line
        self.load_command()

        # Ask for scaler, model and CV parameters
        print('Scaler type and parameters (Default:Standard):')
        self.get_line() 
        self.scaler_command()
        print('Model type and parameters (Default:linear):')
        self.get_line()
        self.model_command()
        print('CV parameters (Default: nº splits=5 and seed=0):')
        self.get_line()
        self.cv_command()

        # Configure Interface
        super().__init__(self.args)
    
    def command_interface(self):
        """Read the command entered and call the corresponding function."""

        # Interactive mode after setup
        print("Initialization finished. Enter 'h' for help.")
        self.get_line()  # get the user entry
        command = self.read_command()  # read the first character
        while command != "q":
            if command == "h":
                self.help_command()
            elif command == "l":
                self.load_command()
            elif command == "m":
                self.model_command()
            elif command == "o":
                self.output_command()
            elif command == "r":
                self.run_command()
            elif command == "s":
                self.scaler_command()
            elif command == "v":
                self.cv_command()
            else:
                print("Invalid command. Enter 'h' for help.")
            self.get_line()  # get the user entry
            command = self.read_command()  # read the first character

    def get_line(self):
        """Print prompt for the user and update the user entry."""
        self.cursor = 0
        self.line = input("#: ")
        while self.line == "":  # if the user enters a blank line
            self.line = input("#: ")

    def read_command(self):
        """Return the first non-whitespace character."""
        self.skip_spaces()
        return self.character

    def get_character(self):
        """Move the cursor forward by one character in the user entry."""
        if self.cursor < len(self.line):
            self.character = self.line[self.cursor]
            self.cursor += 1
        else:  # end of the line
            self.character = ""

    def skip_spaces(self):
        """Skip whitespace until a non-whitespace character is reached."""
        self.get_character()
        while self.character.isspace():
            self.get_character()

    def cv_command(self):
        """Load CV parameters."""
        self.line = self.line.split()
        # TODO check that i/ps are integers
        if self.line[0] == 'None':
            self.args.cv_split = 5
            self.args.seed = 0
        elif len(self.line) == 1:
            self.args.cv_split = self.line[0]
            self.args.seed = 0
        elif len(self.line) == 2:
            self.args.cv_split, self.args.seed = self.line[0], self.line[1]
        else:
            # TODO ask for i/p again
            print('Too many values to unpack')

    def help_command(self):
        """Print a list of valid commands."""
        print("User commands:")
        print("h                                   - help (this command)")
        print("l --flag [file]                     - load file with the specified flag")
        print("m model_type [param1, param2, ...]  - set model type and parameters (Default: linear)")
        print("o [directory]                       - set output directory")
        print("q                                   - quit the program")
        print("r                                   - run the modelling")
        print("s scaler_type [param1, param2, ...] - set scaler type and parameters (Default: standard)")
        print("v [nº splits] [seed]                - set CV parameters (Default: 5, 0)")

    def load_command(self):
        """Load file paths."""
        self.line = self.line.split()
        command = self.line[0]
        file = self.line[1]
        if file == 'None':
            file = None
        if command == '--features':
            self.args.features = file
        elif command == '--covariates':
            self.args.covariates = file
        elif command == '--factors':
            self.args.factors = file
        elif command == '--clinical':
            self.args.clinical = file
        elif command == '--systems':
            self.args.systems = file
        else:
            print('Choose a valid file type: --features, --covariates, --factors, --clinical, --systems')
    
    def model_command(self):
        """Load model parameters."""
        self.line = self.line.split()
        # Set deafult
        if self.line[0] == 'None':
            self.args.model_type = 'linear'
        else:
            # TODO check that model type is valid
            self.args.model_type = self.line[0]
        if len(self.line[0]) > 1:
            model_params = {}
            for item in self.line[1:]:
                key, value = item.split('=')
                value = convert(value)
                model_params[key] = value
            self.args.model_params = model_params
        else:
            self.args.model_params = {}

    def output_command(self):
        """Load output directory."""
        self.args.output = self.line

    def run_command(self):
        """Run the modelling."""
        self.run()

    def scaler_command(self):
        self.line = self.line.split()
        if self.line[0] == 'None':
            self.args.scaler_type = 'standard'
        else:
            # TODO check that scaler type is valid
            self.args.scaler_type = self.line[0]
        if len(self.line[0]) > 1:
            scaler_params = {}
            for item in self.line[1:]:
                key, value = item.split('=')
                value = convert(value)
                scaler_params[key] = value
            self.args.scaler_params = scaler_params
        else:
            self.args.scaler_params = {}