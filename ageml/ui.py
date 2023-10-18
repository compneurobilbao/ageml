"""Implement the user interface.

Used in the AgeML project to enable the user to enter commands
to run the modelling with desired inputs.

Classes:
--------
Interface - reads, parses and executes user commands.
CLI - reads and parsers user commands via command line.
"""

import argparse
import pandas as pd
import os
import warnings

from datetime import datetime

from .visualizer import Visualizer
from .utils import create_directory, log

class Interface:

    """Reads, parses and executes user commands.

    This class allows the user to enter certain commands.
    These commands enable the user to run the modellin with the selected
    input files and parameters.

    Parameters
    -----------

    Public methods:
    ---------------
    run(self): Runs the age modelling.
    """

    def __init__(self):
        """Initialise variables."""

        # Arguments with which to run modelling
        self.args = None

        # Initialise objects form library
        self.visualizer = Visualizer()

    def _setup(self):
        """Create required directories and files to store results."""

        # Create directories
        dir_path = os.path.join(self.args.output, 'ageml')
        if os.path.exists(dir_path):
            warnings.warn("Directory %s already exists files may be overwritten." %
                          dir_path)
        create_directory(dir_path)
        create_directory(os.path.join(dir_path,'figures'))

        # Set visualizer directory path
        self.visualizer.set_directory(dir_path)

        # Create .txt log file and log time
        self.log_path = os.path.join(dir_path, 'log.txt')
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
            df = pd.read_csv(file, header=0, index_col=None)
            df.columns = df.columns.str.lower() # ensure lower case
            return df
        else:
            return None

    @log
    def _features_vs_age(self):
        """Use visualizer to explore relationship between features and age."""

        # Select data to visualize
        feature_names = self.df_features.columns[2:]
        X = self.df_features[feature_names].to_numpy()
        Y = self.df_features['age'].to_numpy()

        # Use visualizer to show 
        self.visualizer.featuresvsage(X, Y, feature_names)

    def run(self):
        """Read the command entered and call the corresponding functions"""

        # Set up directory for storage of results
        self._setup()

        # Load data 
        self.df_features = self._load_csv(self.args.features)
        self.df_covariates = self._load_csv(self.args.covariates)
        self.df_factors = self._load_csv(self.args.factors)
        self.df_clinical = self._load_csv(self.args.clinical)

        # Relationship between features and age
        self._features_vs_age()

class CLI(Interface):

    """Read and parses user commands via command line."""

    def __init__(self):
        """Initialise variables."""
        super().__init__()
        self.parser = argparse.ArgumentParser(description="Age Modelling using python.",
                                              formatter_class=argparse.RawTextHelpFormatter)
        self._configure_parser()
        self.args = self.parser.parse_args()

    def _configure_parser(self):
        """Configure parser with required arguments for processing."""
        self.parser.add_argument('-o', '--output', metavar='DIR', required=True,
                                 help="Path to output directory where to save results. (Required)")
        self.parser.add_argument("--features", metavar='FILE',
                                 help="Path to input CSV file containing features. (Required) \n"
                                      "In the file the first column should be the ID, the second column should be the AGE, \n"
                                      "and the following columns the features. The first row should be the header for \n"
                                      "column names.", required=True)
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

