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

    def _load_csv(self, file):
        """Use panda to load csv into dataframe."""
        if file is not None:
            return pd.read_csv(file, header=0, index_col=None)
        else:
            return None

    def run(self):
        """Read the command enterend and call the corresponding functions"""

        # Load data
        self.df_features = self._load_csv(self.args.features)
        self.df_covariates = self._load_csv(self.args.covariates)
        self.df_factors = self._load_csv(self.args.factors)
        self.df_clinical = self._load_csv(self.args.clinical)

class CLI(Interface):

    """Read and parses user commands via command line."""

    def __init__(self):
        """Initialise variables."""
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

