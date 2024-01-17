"""Command line commands.

Used in the AgeML project with poetry to create command line commands.

Public classes:
---------------
ModelAge: Run age modelling with required parameters.
FactorAnalysis: Run factor analysis with required parameters.
ClinicalGroups: Run clinical analysis with age deltas with required parameters.
ClinicalClassification: Run classification of groups based on age deltas with required parameters.


Public functions:
-----------------
model_age(): Run ModelAge class.
factor_analysis(): Run FactorAnalysis class.
clinical_groups(): Run ClinicalGroups class.
clinical_classify(): Run ClinicalClassification class.
"""

import argparse

import ageml.messages as messages

from ageml.ui import Interface
from ageml.utils import convert


class ModelAge(Interface):
    """Run age modelling with required parameters."""

    def __init__(self):
        """Initialise variables."""

        # Initialise parser
        self.parser = argparse.ArgumentParser(
            description="Age Modelling using python.",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Configure parser
        self.configure_parser()
        args = self.parser.parse_args()
        args = self.configure_args(args)

        # Initialise parent class
        super().__init__(args)

        # Run age modelling
        self.run_wrapper(self.run_age)

    def configure_parser(self):
        """Configure parser with required arguments for processing."""

        # Required arguments
        self.parser.add_argument("-o", "--output", metavar="DIR", required=True,
                                 help=messages.output_long_description,)
        self.parser.add_argument("-f", "--features", metavar="FILE", required=True,
                                 help=messages.features_long_description)
        
        # Parameter arguments with defaults
        self.parser.add_argument("-m", "--model", nargs="*", default=["linear"],
                                 help=messages.model_long_description)
        self.parser.add_argument("-s", "--scaler", nargs="*", default=["standard"],
                                 help=messages.scaler_long_description)
        self.parser.add_argument("--cv", nargs="+", type=int, default=[5, 0],
                                 help=messages.cv_long_description)
        
        # Optional arguments
        self.parser.add_argument("--covariates", metavar="FILE",
                                 help=messages.covar_long_description)
        self.parser.add_argument("--covar_name", metavar="COVAR_NAME",
                                 help=messages.covar_name_long_description)
        self.parser.add_argument("--clinical", metavar="FILE",
                                 help=messages.clinical_long_description)
        self.parser.add_argument("--systems", metavar="FILE",
                                 help=messages.systems_long_description)

    def configure_args(self, args):
        """Configure argumens with required fromatting for modelling.

        Parameters
        ----------
        args: arguments object from parser
        """

        # Set CV params first item is the number of CV splits
        if len(args.cv) == 1:
            args.cv_split = args.cv[0]
            args.seed = self.parser.get_default("cv")[1]
        elif len(args.cv) == 2:
            args.cv_split, args.seed = args.cv
        else:
            raise ValueError("Too many values to unpack")

        # Set Scaler parameters first item is the scaler type
        # The rest of the arguments conform a dictionary for **kwargs
        args.scaler_type = args.scaler[0]
        if len(args.scaler) > 1:
            scaler_params = {}
            for item in args.scaler[1:]:
                # Check that item has one = to split
                if item.count("=") != 1:
                    raise ValueError(
                        "Scaler parameters must be in the format param1=value1 param2=value2 ..."
                    )
                key, value = item.split("=")
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
                # Check that item has one = to split
                if item.count("=") != 1:
                    raise ValueError(
                        "Model parameters must be in the format param1=value1 param2=value2 ..."
                    )
                key, value = item.split("=")
                value = convert(value)
                model_params[key] = value
            args.model_params = model_params
        else:
            args.model_params = {}

        return args


class FactorAnalysis(Interface):
    """Run factor analysis with required parameters."""

    def __init__(self):
        """Initialise variables."""

        # Initialise parser
        self.parser = argparse.ArgumentParser(
            description="Factor analysis of age deltas.",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Configure parser
        self.configure_parser()
        args = self.parser.parse_args()

        # Initialise parent class
        super().__init__(args)

        # Run factor analysis
        self.run_wrapper(self.run_factor_analysis)

    def configure_parser(self):
        """Configure parser with required arguments for processing."""

        # Required arguments
        self.parser.add_argument("-o", "--output", metavar="DIR", required=True,
                                 help=messages.output_long_description,)
        self.parser.add_argument("-a", "--ages", metavar="FILE", required=True,
                                 help=messages.ages_long_description)
        self.parser.add_argument("-f", "--factors", metavar="FILE", required=True,
                                 help=messages.factors_long_description)
        
        # Optional arguments
        self.parser.add_argument("--covariates", metavar="FILE",
                                 help=messages.covar_long_description)
        self.parser.add_argument("--clinical", metavar="FILE",
                                 help=messages.clinical_long_description)


class ClinicalGroups(Interface):
    """Run clinical analysis with age deltas with required parameters."""

    def __init__(self):
        """Initialise variables."""

        # Initialise parser
        self.parser = argparse.ArgumentParser(
            description="Age deltas for each clinical group.",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Configure parser
        self.configure_parser()
        args = self.parser.parse_args()

        # Initialise parent class
        super().__init__(args)

        # Run factor analysis
        self.run_wrapper(self.run_clinical)

    def configure_parser(self):
        """Configure parser with required arguments for processing."""

        # Required arguments
        self.parser.add_argument("-o", "--output", metavar="DIR", required=True,
                                 help=messages.output_long_description,)
        self.parser.add_argument("-a", "--ages", metavar="FILE", required=True,
                                 help=messages.ages_long_description)
        self.parser.add_argument("--clinical", metavar="FILE", required=True,
                                 help=messages.clinical_long_description)


class ClinicalClassification(Interface):
    """Run classification of groups based on age deltas with required parameters."""

    def __init__(self):
        """Initialise variables."""

        # Initialise parser
        self.parser = argparse.ArgumentParser(
            description="Classify two groups based on age deltas.",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Configure parser
        self.configure_parser()
        args = self.parser.parse_args()

        # Parse groups
        args.group1, args.group2 = args.groups

        # Initialise parent class
        super().__init__(args)

        # Run factor analysis
        self.run_wrapper(self.run_classification)

    def configure_parser(self):
        """Configure parser with required arguments for processing."""

        # Required arguments
        self.parser.add_argument("-o", "--output", metavar="DIR", required=True,
                                 help=messages.output_long_description,)
        self.parser.add_argument("-a", "--ages", metavar="FILE", required=True,
                                 help=messages.ages_long_description)
        self.parser.add_argument("--clinical", metavar="FILE", required=True,
                                 help=messages.clinical_long_description)
        self.parser.add_argument("--groups", nargs=2, metavar="GROUP", required=True,
                                 help=messages.groups_long_description)


# Object wrappers

def model_age():
    """Run ModelAge class."""

    ModelAge()


def factor_analysis():
    """Run FactorAnalysis class."""

    FactorAnalysis()


def clinical_groups():
    """Run ClinicalGroups class."""

    ClinicalGroups()


def clinical_classify():
    """Run ClinicalClassification class."""

    ClinicalClassification()