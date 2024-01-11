"""Implement the user interface.

Used in the AgeML project to enable the user to enter commands
to run the modelling with desired inputs.

Classes:
--------
Interface - reads, parses and executes user commands.
CLI - reads and parsers user commands via command line.
InteractiveCLI - reads and parsers user commands via command line via an interactive interface.
"""

import argparse
import numpy as np
import pandas as pd
import os
import warnings

from datetime import datetime
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats

import ageml.messages as messages
from ageml.visualizer import Visualizer
from ageml.utils import create_directory, feature_extractor, significant_markers, convert, log
from ageml.modelling import AgeML, Classifier
from ageml.processing import find_correlations


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
    setup(self): Creates required directories and files to store results.

    set_flags(self): Set flags.

    set_visualizer(self): Set visualizer with output directory.

    set_model(self): Set model with parameters.

    set_classifier(self): Set classifier with parameters.

    check_file(self, file): Check that file exists.

    load_csv(self, file): Use panda to load csv into dataframe.

    load_data(self, required): Load data from csv files.

    age_distribution(self, dfs, labels=None): Use visualizer to show age distribution.

    features_vs_age(self, df, significance=0.05): Use visualizer to explore relationship between features and age.

    model_age(self, df, model): Use AgeML to fit age model with data.

    predict_age(self, df, model): Use AgeML to predict age with data.

    factors_vs_deltas(self, dfs_ages, dfs_factors, groups, significance=0.05): Calculate correlations between factors and deltas.

    deltas_by_group(self, df, labels): Calculate summary metrics of deltas by group.

    classify(self, df1, df2, groups): Classify two groups based on deltas.

    run_wrapper(self, run): Wrapper for running modelling with log.

    run_age(self): Run age modelling.

    run_factor_analysis(self): Factor analysis between deltas and factors.

    run_clinical(self): Analyse differences between deltas in clinical groups.

    run_classification(self): Classify groups based on deltas.
    """

    def __init__(self, args):
        """Initialise variables."""

        # Arguments with which to run modelling
        self.args = args

        # Flags
        self.set_flags()

        # Set up directory for storage of results
        self.setup()

        # Initialise objects form library
        self.set_visualizer()

    def setup(self):
        """Create required directories and files to store results."""

        # Create directories
        self.dir_path = os.path.join(self.args.output, "ageml")
        if os.path.exists(self.dir_path):
            warnings.warn(
                "Directory %s already exists files may be overwritten." % self.dir_path,
                category=UserWarning,
            )
        create_directory(self.dir_path)
        create_directory(os.path.join(self.dir_path, "figures"))

        # Create .txt log file and log time
        self.log_path = os.path.join(self.dir_path, "log.txt")
        with open(self.log_path, "a") as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(current_time + "\n")

    def set_flags(self):
        """Set flags."""

        self.flags = {"clinical": False, "covariates": False}

    def set_visualizer(self):
        """Set visualizer with output directory."""

        self.visualizer = Visualizer(self.dir_path)

    def set_model(self):
        """Set model with parameters."""

        self.ageml = AgeML(
            self.args.scaler_type,
            self.args.scaler_params,
            self.args.model_type,
            self.args.model_params,
            self.args.cv_split,
            self.args.seed,
        )

    def set_classifier(self):
        """Set classifier with parameters."""

        self.classifier = Classifier()

    def check_file(self, file):
        """Check that file exists."""
        if not os.path.exists(file):
            return False
        else:
            return True

    def load_csv(self, file_type):
        """Use panda to load csv into dataframe.

        Parameters
        ----------
        file_type: type of file to load
        """

        # Obtain file name
        if hasattr(self.args, file_type):
            file = getattr(self.args, file_type)
        else:
            file = None

        if file is not None:
            # Check file exists
            if not self.check_file(file):
                raise FileNotFoundError("File %s not found." % file)
            df = pd.read_csv(file, header=0, index_col=0)
            df.columns = df.columns.str.lower()  # ensure lower case
            return df
        else:
            return None

    def load_data(self, required=None):
        """Load data from csv files.

        Parameters
        ----------
        required: list of required files"""

        # Load files
        print("-----------------------------------")
        print("Loading data...")

        # Required files default
        if required is None:
            required = []

        # Load features
        self.df_features = self.load_csv('features')
        if self.df_features is not None:
            if "age" not in self.df_features.columns:
                raise KeyError(
                    "Features file must contain a column name 'age', or any other case-insensitive variation."
                )
        elif "features" in required:
            raise ValueError("Features file must be provided.")

        # Load covariates
        self.df_covariates = self.load_csv('covariates')
        # Check that covar name is given
        if self.df_covariates is not None and hasattr(self.args, 'covar_name'):
            self.flags['covariates'] = True

        # Load factors
        self.df_factors = self.load_csv('factors')
        if self.df_factors is None and "factors" in required:
            raise ValueError("Factors file must be provided.")

        # Load clinical
        self.df_clinical = self.load_csv('clinical')
        if self.df_clinical is not None:
            # Check that CN in columns
            if "cn" not in self.df_clinical.columns:
                raise KeyError(
                    "Clinical file must contian a column name 'CN' or any other case-insensitive variation."
                )
            # Check datatypes of columns are all boolean
            elif [
                self.df_clinical[col].dtype == bool for col in self.df_clinical.columns
            ].count(False) != 0:
                raise TypeError("Clinical columns must be boolean type.")
            else:
                self.flags['clinical'] = True
                self.cn_subjects = self.df_clinical[self.df_clinical["cn"]].index
        elif "clinical" in required:
            raise ValueError("Clinical file must be provided.")

        # Load ages
        self.df_ages = self.load_csv('ages')
        # Check that ages file has required columns
        if self.df_ages is not None:
            cols = ["age", "predicted age", "corrected age", "delta"]
            for col in cols:
                if col not in self.df_ages.columns:
                    raise KeyError("Ages file must contain a column name %s" % col)
        elif "ages" in required:
            raise ValueError("Ages file must be provided.")

        # Remove subjects with missing values
        dfs = [
            self.df_features,
            self.df_covariates,
            self.df_factors,
            self.df_clinical,
            self.df_ages,
        ]
        labels = ['features', 'covariates', 'factors', 'clinical', 'ages']
        self.subjects_missing_data = []
        for label, df in zip(labels, dfs):
            if df is not None:
                missing_subjects = df[df.isnull().any(axis=1)].index.to_list()
                self.subjects_missing_data = (
                    self.subjects_missing_data + missing_subjects
                )
                if missing_subjects.__len__() != 0:
                    warn_message = "Subjects with missing data in %s: %s" % (label, missing_subjects)
                    print(warn_message)
                    warnings.warn(warn_message, category=UserWarning)

        # Check that all dataframes have the same subjects
        for i in range(len(dfs)):
            for j in range(len(dfs)):
                if i != j:
                    if dfs[i] is not None and dfs[j] is not None:
                        # Find subjects in one dataframe but not the other
                        non_shared_subjects = [s for s in dfs[i].index.to_list()
                                               if s not in dfs[j].index.to_list()]
                        if non_shared_subjects.__len__() != 0:
                            warn_message = ("Subjects in dataframe %s not in dataframe %s: %s"
                                            % (labels[i], labels[j], non_shared_subjects))
                            print(warn_message)
                            warnings.warn(warn_message, category=UserWarning)
                            self.subjects_missing_data = (
                                self.subjects_missing_data + non_shared_subjects
                            )

        # Remove subjects with missing values
        self.subjects_missing_data = set(self.subjects_missing_data)
        for df in dfs:
            if df is not None:
                df.drop(self.subjects_missing_data, inplace=True, errors="ignore")

    def age_distribution(self, dfs: list, labels=None, name=""):
        """Use visualizer to show age distribution.

        Parameters
        ----------
        dfs: list of dataframes with age information; shape=(n,m)
        labels: categories of separation criterion
        name: name to give to visualizer to save file"""

        # Select age information
        print("-----------------------------------")
        print("Age distribution %s" % name)
        list_ages = []
        for i, df in enumerate(dfs):
            if labels is not None:
                print(labels[i])
            ages = df["age"].to_numpy()
            print("Mean age: %.2f" % np.mean(ages))
            print("Std age: %.2f" % np.std(ages))
            print("Age range: [%d,%d]" % (np.min(ages), np.max(ages)))
            list_ages.append(ages)

        # Check that distributions of ages are similar
        print("Checking that age distributions are similar...")
        for i in range(len(list_ages)):
            for j in range(i + 1, len(list_ages)):
                _, p_val = stats.ttest_ind(list_ages[i], list_ages[j])
                if p_val < 0.05:
                    warn_message = "Age distributions %s and %s are not similar." % (
                        labels[i],
                        labels[j],
                    )
                    print(warn_message)
                    warnings.warn(warn_message, category=UserWarning)

        # Use visualiser
        self.visualizer.age_distribution(list_ages, labels, name)

    def features_vs_age(self, dfs: list, labels: list = None, significance: float = 0.05, name: str = ""):
        """Use visualizer to explore relationship between features and age.

        Parameters
        ----------
        dfs: list of dataframes with features and age; shape=(n,m+1)
        labels: # TODO
        significance: significance level for correlation"""

        # Select data to visualize
        print("-----------------------------------")
        print("Features by correlation with Age")
        print("significance: %.2g * -> FDR, ** -> bonferroni" % significance)
        if not isinstance(dfs, list):
            raise TypeError("Input to 'Interface.features_vs_age' must be a list of dataframes.")

        if labels is not None:
            print(labels)

        # Make lists to store covariate info for each dataframe
        X_list = []
        y_list = []
        corr_list = []
        order_list = []
        significance_list = []
        for df, label in zip(dfs, labels):
            # Extract features
            X, y, feature_names = feature_extractor(df)
            # Calculate correlation between features and age
            corr, order, p_values = find_correlations(X, y)
            # Reject null hypothesis of no correlation
            reject_bon, _, _, _ = multipletests(p_values, alpha=significance, method='bonferroni')
            reject_fdr, _, _, _ = multipletests(p_values, alpha=significance, method='fdr_bh')
            significant = significant_markers(reject_bon, reject_fdr)
            # Print results
            for idx, order_element in enumerate(order):
                print("%d.%s %s %s: %.2f" % (idx + 1, label, significant[order_element],
                                             feature_names[order_element], corr[order_element]))
            # Append all the values
            X_list.append(X)
            y_list.append(y)
            corr_list.append(corr)
            order_list.append(order)
            significance_list.append(significant)

        # Use visualizer to show results
        self.visualizer.features_vs_age(X_list, y_list, corr_list, order_list,
                                        significance_list, feature_names, labels, name)

    def model_age(self, df, model, name: str = ""):
        """Use AgeML to fit age model with data.

        Parameters
        ----------
        df: dataframe with features and age; shape=(n,m+1)
        model: AgeML object"""

        # Show training pipeline
        print("-----------------------------------")
        if name == "":
            print("Training Age Model")
        else:
            print("Training Model for covariate %s" % name)
        print(self.ageml.pipeline)

        # Select data to model
        X, y, _ = feature_extractor(df)

        # Fit model and plot results
        y_pred, y_corrected = model.fit_age(X, y)
        self.visualizer.true_vs_pred_age(y, y_pred, name)
        self.visualizer.age_bias_correction(y, y_pred, y_corrected, name)

        # Calculate deltas
        deltas = y_corrected - y

        # Save to dataframe and csv
        data = np.stack((y, y_pred, y_corrected, deltas), axis=1)
        cols = ["age", "predicted age", "corrected age", "delta"]
        df_ages = pd.DataFrame(data, index=df.index, columns=cols)

        return model, df_ages

    def predict_age(self, df, model):
        """Use AgeML to predict age with data."""

        # Show prediction pipeline
        print("-----------------------------------")
        print("Predicting with Age Model")
        print(self.ageml.pipeline)

        # Select data to model
        X, y, _ = feature_extractor(df)

        # Predict age
        y_pred, y_corrected = model.predict_age(X, y)

        # Calculate deltas
        deltas = y_corrected - y

        # Save to dataframe and csv
        data = np.stack((y, y_pred, y_corrected, deltas), axis=1)
        cols = ["age", "predicted age", "corrected age", "delta"]
        df_ages = pd.DataFrame(data, index=df.index, columns=cols)

        return df_ages

    def factors_vs_deltas(self, dfs_ages, dfs_factors, groups, factor_names, significance=0.05):
        """Calculate correlations between factors and deltas.

        Parameters
        ----------
        dfs_ages: list of dataframes with delta information; shape=(n,m)
        dfs_factors: list of dataframes with factor information; shape=(n,m)
        groups: list of labels for each dataframe; shape=(n,),
        factor_names: list of factor names; shape=(m,)
        significance: significance level for correlation"""

        # Select age information
        print("-----------------------------------")
        print("Correlations between lifestyle factors by group")
        print("significance: %.2g * -> FDR, ** -> bonferroni" % significance)

        # Iterate over groups
        corrs, significants = [], []
        for group, df_ages, df_factors in zip(groups, dfs_ages, dfs_factors):
            print(group)

            # Select data to visualize
            deltas = df_ages["delta"].to_numpy()
            factors = df_factors.to_numpy()

            # Calculate correlation between features and age
            corr, order, p_values = find_correlations(factors, deltas)
            corrs.append(corr)

            # Reject null hypothesis of no correlation
            reject_bon, _, _, _ = multipletests(p_values, alpha=significance, method='bonferroni')
            reject_fdr, _, _, _ = multipletests(p_values, alpha=significance, method='fdr_bh')
            significant = significant_markers(reject_bon, reject_fdr)
            significants.append(significant)

            # Print results
            for i, o in enumerate(order):
                print("%d. %s %s: %.2f" % (i + 1, significant[o], factor_names[o], corr[o]))

        # Use visualizer to show bar graph
        self.visualizer.factors_vs_deltas(corrs, groups, factor_names, significants)

    def deltas_by_group(self, df, labels):
        """Calculate summary metrics of deltas by group.
        
        Parameters
        ----------
        df: list of dataframes with delta information; shape=(n,m)
        labels: list of labels for each dataframe; shape=(n,)"""

        # Select age information
        print("-----------------------------------")
        print("Delta distribution by group")

        # Obtain deltas means and stds
        deltas = []
        for i, df_group in enumerate(df):
            deltas.append(df_group["delta"].to_numpy())
            print(labels[i])
            print("Mean delta: %.2f" % np.mean(deltas[i]))
            print("Std delta: %.2f" % np.std(deltas[i]))
            print("Delta range: [%d, %d]" % (np.min(deltas[i]), np.max(deltas[i])))

        # Obtain statistically significant difference between deltas
        print("Checking for statistically significant differences between deltas...")
        print("*: p-value < 0.01, **: p-value < 0.001")
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                _, p_val = stats.ttest_ind(deltas[i], deltas[j])
                pval_message = "p-value between %s and %s: %.2g" % (
                    labels[i],
                    labels[j],
                    p_val,
                )
                if p_val < 0.001:
                    pval_message = "*" + pval_message
                elif p_val < 0.01:
                    pval_message = "**" + pval_message
                print(pval_message)

        # Use visualizer
        self.visualizer.deltas_by_groups(deltas, labels)

    def classify(self, df1, df2, groups):
        """Classify two groups based on deltas.

        Parameters
        ----------
        df1: dataframe with delta information; shape=(n,m)
        df2: dataframe with delta information; shape=(n,m)
        groups: list of labels for each dataframe; shape=(2,)"""

        # Classification
        print("-----------------------------------")
        print("Classification between groups %s and %s" % (groups[0], groups[1]))

        # Select delta information
        deltas1 = df1["delta"].to_numpy()
        deltas2 = df2["delta"].to_numpy()

        # Create X and y for classification
        X = np.concatenate((deltas1, deltas2)).reshape(-1, 1)
        y = np.concatenate((np.zeros(deltas1.shape), np.ones(deltas2.shape)))

        # Calculate classification
        y_pred = self.classifier.fit_model(X, y)

        # Visualize AUC
        self.visualizer.classification_auc(y, y_pred, groups)

    @log
    def run_wrapper(self, run):
        """Wrapper for running modelling with log."""
        run()

    def run_age(self):
        """Run age modelling."""

        # Run age modelling
        print("Running age modelling...")

        # Reset flags
        self.set_flags()

        # Load data
        self.load_data(required=["features"])

        # Select controls
        if self.flags["clinical"]:
            df_cn = self.df_features.loc[self.df_features.index.isin(self.cn_subjects)]
            df_clinical = self.df_features.loc[~self.df_features.index.isin(self.cn_subjects)]
        else:
            df_cn = self.df_features
            df_clinical = None

        if self.flags["covariates"] and self.args.covar_name is not None:
            # Check that covariate column exists
            if self.args.covar_name not in self.df_covariates.columns:
                raise KeyError("Covariate column %s not found in covariates file." % self.args.covar_name)

            # Create dataframe list of controls by covariate
            labels_covar = pd.unique(self.df_covariates[self.args.covar_name]).tolist()
            df_covar_cn = self.df_covariates.loc[df_cn.index]
            dfs_cn = []
            for label_covar in labels_covar:
                dfs_cn.append(df_cn[df_covar_cn[self.args.covar_name] == label_covar])

            if self.flags["clinical"]:
                # Create dataframe list of clinical cases by covariate
                dfs_clinical = []
                for label_covar in labels_covar:
                    df_covar_clinical = self.df_covariates.loc[df_clinical.index]
                    dfs_clinical.append(df_clinical[df_covar_clinical[self.args.covar_name] == label_covar])

        else:  # No covariates, so df list of controls is [df_cn] and [df_clinical]
            dfs_cn = [df_cn]
            dfs_clinical = [df_clinical]
            labels_covar = ["all"]
            self.args.covar_name = "all"

        # Relationship between features and age
        if self.flags["covariates"]:
            initial_plots_names = f"controls_{self.args.covar_name}"
        else:
            initial_plots_names = "controls"
            
        # Use visualizer to show age distribution
        self.age_distribution(dfs_cn, labels=labels_covar, name=initial_plots_names)
    
        self.features_vs_age(dfs_cn, labels=labels_covar, name=initial_plots_names)
        
        # Model age
        self.models = {}
        dfs_ages = {}
        for label_covar, df_cn in zip(labels_covar, dfs_cn):
            model_name = f"{self.args.covar_name}_{label_covar}"
            # Reinitialise model for each covariate to ensure same initial state
            self.set_model()
            self.models[model_name], dfs_ages[model_name] = self.model_age(df_cn, self.ageml, label_covar)
            df_ages_cn = pd.concat(dfs_ages.values(), axis=0)

        # NOTE: Matching dataframes that cannot be indexed by their name and models could be dangerous and prone to mismatches.
        # TODO: Discuss about alternatives. Use dicts for all dataframes and models?

        # Apply to clinical data
        dict_predicted_ages = {}
        if self.flags["clinical"]:
            for df_age_clinical, label_covar in zip(dfs_clinical, labels_covar):
                model_name = f"{self.args.covar_name}_{label_covar}"
                dict_predicted_ages[model_name] = self.predict_age(df_age_clinical, self.models[model_name])
            # Concatenate all the predicted ages
            self.df_ages = pd.concat(list(dict_predicted_ages.values()))
            self.df_ages = pd.concat([self.df_ages, df_ages_cn])
        else:
            self.df_ages = df_ages_cn

        # Save dataframe
        if self.flags["covariates"]:
            filename = f"predicted_age_{self.args.covar_name}.csv"
        else:
            filename = "predicted_age.csv"
        self.df_ages.to_csv(os.path.join(self.dir_path, filename))

    def run_factor_analysis(self):
        """Run factor analysis between deltas and factors."""

        print("Running lifestyle factors...")

        # Reset flags
        self.set_flags()

        # Load data
        self.load_data(required=["ages", "factors"])

        # Check wether to split by clinical groups
        if self.flags["clinical"]:
            groups = self.df_clinical.columns.to_list()
            dfs_ages, dfs_factors = [], []
            for g in groups:
                dfs_ages.append(self.df_ages.loc[self.df_clinical[g]])
                dfs_factors.append(self.df_factors.loc[self.df_clinical[g]])
        else:
            dfs_ages = [self.df_ages]
            dfs_factors = [self.df_factors]
            groups = ["all"]

        # Calculate correlations between factors and deltas
        self.factors_vs_deltas(dfs_ages, dfs_factors, groups, self.df_factors.columns.to_list())

    def run_clinical(self):
        """Analyse differences between deltas in clinical groups."""

        print("Running clinical outcomes...")

        # Reset flags
        self.set_flags()

        # Load data
        self.load_data(required=["ages", "clinical"])

        # Obtain dataframes for each clinical group
        groups = self.df_clinical.columns.to_list()
        group_ages = []
        for g in groups:
            group_ages.append(self.df_ages.loc[self.df_clinical[g]])

        # Use visualizer to show age distribution
        self.age_distribution(group_ages, groups, name="clinical_groups")

        # Use visualizer to show box plots of deltas by group
        self.deltas_by_group(group_ages, groups)

    def run_classification(self):
        """Run classification between two different clinical groups."""

        print("Running classification...")

        # Reset flags
        self.set_flags()

        # Load data
        self.load_data(required=["ages", "clinical"])

        # Check that arguments given for each group
        if self.args.group1 is None or self.args.group2 is None:
            raise ValueError("Must provide two groups to classify.")
        
        # Check that those groups exist
        groups = [self.args.group1.lower(), self.args.group2.lower()]
        if groups[0] not in self.df_clinical.columns or groups[1] not in self.df_clinical.columns:
            raise ValueError("Classes must be one of the following: %s" % self.df_clinical.columns.to_list())

        # Obtain dataframes for each clinical group
        df_group1 = self.df_ages.loc[self.df_clinical[groups[0]]]
        df_group2 = self.df_ages.loc[self.df_clinical[groups[1]]]

        # Classify between groups
        self.set_classifier()
        self.classify(df_group1, df_group2, groups)


class CLI(Interface):

    """Read and parses user commands via command line via an interactive interface

    Public methods:
    ---------------

    initial_command(self): Ask for initial inputs for initial setup.

    get_line(self): Prints a prompt for the user and updates the user entry.

    force_command(self, flag="", command = None): Force the user to enter a valid command.

    command_interface(self): Reads in the commands and calls the corresponding
                             functions.

    classification_command(self): Runs classification.

    clinical_command(self): Runs clinical analysis.

    covar_command(self): Loads covariate group.

    cv_command(self): Loads CV parameters.

    factor_analysis_command(self): Runs factor analysis.

    group_command(self): Loads groups.

    help_command(self): Prints a list of valid commands.

    load_command(self): Loads file paths.

    model_command(self): Loads model parameters.

    model_age_command(self): Runs age modelling.

    output_command(self): Loads output directory.

    scaler_command(self): Loads scaler parameters.
    """

    def __init__(self):
        """Initialise variables."""

        # Initialization
        self.args = argparse.Namespace()

        # Print welcome message
        print(messages.emblem)
        print("Age Modelling (AgeML): interactive command line user interface.")

        # Setup
        print(messages.setup_banner)
        print("For Optional or Default values leave empty. \n")
        self.initial_command()

        # Configure Interface
        self.configFlag = False
        while not self.configFlag:
            try:
                print("\n Configuring Interface...")
                super().__init__(self.args)
                self.configFlag = True
            except Exception as e:
                print(e)
                print("Error configuring interface, please try again.")
                self.initial_command()

        # Run command interface
        print("\n Initialization finished.")
        print(messages.modelling_banner)
        self.command_interface()

    def initial_command(self):
        """Ask for initial inputs for initial setup."""

        # Askf for output directory
        print("Output directory path (Required):")
        self.force_command(self.output_command, required=True)

    def get_line(self, required=True):
        """Print prompt for the user and update the user entry."""
        self.line = input("#: ")
        while self.line == "" and required:
            print("Must provide a value.")
            self.line = input("#: ")

    def force_command(self, func, flag="", required=False):
        """Force the user to enter a valid command."""
        while True:
            self.get_line(required=required)
            if self.line == "":
                self.line = "None"
            self.line = flag + " " + self.line
            error = func()
            if error is None:
                return None
            else:
                print(error)

    def reset_args(self):
        """Reset arguments to None except output directory."""

        for attr_name in vars(self.args):
            if attr_name != 'output':
                setattr(self.args, attr_name, None)

    def command_interface(self):
        """Read the command entered and call the corresponding function."""

        # Interactive mode after setup
        print("Enter 'h' for help.")
        self.get_line()  # get the user entry
        command = self.line.split()[0]  # read the first item
        while command != "q":
            # Reset arguments
            self.reset_args()

            # Run command
            error = None
            if command == "classification":
                error = self.classification_command()
            elif command == "clinical":
                error = self.clinical_command()
            elif command == "factor_analysis":
                error = self.factor_analysis_command()
            elif command == "model_age":
                error = self.model_age_command()
            elif command == "h":
                self.help_command()
            else:
                print("Invalid command. Enter 'h' for help.")

            # Check error and if not make updates
            if error is not None:
                print(error)

            # Get next command
            self.get_line()  # get the user entry
            command = self.line.split()[0]  # read the first item

    def classification_command(self):
        """Run classification."""

        error = None

        # Ask for input files
        print("Input ages file path (Required):")
        self.force_command(self.load_command, "--ages", required=True)
        print("Input clinical file path (Required):")
        self.force_command(self.load_command, "--clinical", required=True)

        # Ask for groups
        print("Input groups (Required):")
        self.force_command(self.group_command, required=True)

        # Run classification capture any error raised and print
        try:
            self.run_wrapper(self.run_classification)
            print("Finished classification.")
        except Exception as e:
            print(e)
            error = "Error running classification."
        
        return error

    def clinical_command(self):
        """Run clinical analysis."""

        error = None

        # Ask for input files
        print("Input ages file path (Required):")
        self.force_command(self.load_command, "--ages", required=True)
        print("Input clinical file path (Required):")
        self.force_command(self.load_command, "--clinical", required=True)

        # Run clinical analysis capture any error raised and print
        try:
            self.run_wrapper(self.run_clinical)
            print("Finished clinical analysis.")
        except Exception as e:
            print(e)
            error = "Error running clinical analysis."
        
        return error

    def covar_command(self):
        """Load covariate group."""

        # Split into items and remove  command
        self.line = self.line.split()
        error = None

        # Check that one argument given
        if len(self.line) != 1:
            error = "Must provide one covariate name."
            return error

        # Set covariate name
        if self.line[0] == "None":
            pass
        else:
            self.args.covar_name = self.line[0]

        return error

    def cv_command(self):
        """Load CV parameters."""

        # Split into items and remove  command
        self.line = self.line.split()
        error = None

        # Check that at least one argument input
        if len(self.line) == 0:
            error = "Must provide at least one argument or None."
            return error

        # Set default values
        if self.line[0] == "None":
            self.args.cv_split = 5
            self.args.seed = 0
            return error

        # Check wether items are integers
        for item in self.line:
            if not item.isdigit():
                error = "CV parameters must be integers"
                return error

        # Set CV parameters
        if len(self.line) == 1:
            self.args.cv_split = int(self.line[0])
            self.args.seed = 0
        elif len(self.line) == 2:
            self.args.cv_split, self.args.seed = int(self.line[0]), int(self.line[1])
        else:
            error = "Too many values to unpack."

        return error

    def factor_analysis_command(self):
        """Run factor analysis."""

        error = None

        # Ask for input files
        print("Input ages file path (Required):")
        self.force_command(self.load_command, "--ages", required=True)
        print("Input factors file path (Required):")
        self.force_command(self.load_command, "--factors", required=True)
        print("Input clinical file path (Optional):")
        self.force_command(self.load_command, "--clinical")
        print("Input covariates file path (Optional):")
        self.force_command(self.load_command, "--covariates")

        # Run factor analysis capture any error raised and print
        try:
            self.run_wrapper(self.run_factor_analysis)
            print("Finished factor analysis.")
        except Exception as e:
            print(e)
            error = "Error running factor analysis."
        
        return error

    def group_command(self):
        """Load groups."""

        # Split into items and remove  command
        self.line = self.line.split()
        error = None

        # Check that two groups are given
        if len(self.line) != 2:
            error = "Must provide two groups."
            return error

        # Set groups
        self.args.group1, self.args.group2 = self.line[0], self.line[1]

        return error

    def help_command(self):
        """Print a list of valid commmands."""

        # Print possible commands
        print("User commands:")
        print(messages.classification_command_message)
        print(messages.clinical_command_message)
        print(messages.factor_analysis_command_message)
        print(messages.model_age_command_message)
        print(messages.quit_command_message)

    def load_command(self):
        """Load file paths."""

        # Split into items and remove  command
        self.line = self.line.split()
        error = None

        # Determine if correct number of arguments and check file valid
        if len(self.line) > 2:
            error = "Too many arguments only two arguments --file_type and file path."
        elif len(self.line) == 1:
            error = "Must provide a file path or None when using --file_type."
        elif len(self.line) == 0:
            error = "Must provide a file type and file path."
        else:
            file_type = self.line[0]
            file = self.line[1]
            # Set file path
            if file == "None":
                file = None
            else:
                if not self.check_file(file):
                    error = "File %s not found." % file
                elif file_type in [
                    "--features",
                    "--covariates",
                    "--factors",
                    "--clinical",
                    "--ages",
                ]:
                    if not file.endswith(".csv"):
                        error = "File %s must be a .csv file." % file
                elif file_type == "--systems":
                    if not file.endswith(".txt"):
                        error = "File %s must be a .txt file." % file

        # Throw error if detected
        if error is not None:
            return error

        # Set file path
        if file_type == "--features":
            self.args.features = file
        elif file_type == "--covariates":
            self.args.covariates = file
        elif file_type == "--factors":
            self.args.factors = file
        elif file_type == "--clinical":
            self.args.clinical = file
        elif file_type == "--systems":
            self.args.systems = file
        elif file_type == "--ages":
            self.args.ages = file
        else:
            error = "Choose a valid file type: --features, --covariates, --factors, --clinical, --systems, --ages"

        return error

    def model_age_command(self):
        """Run age modelling."""

        error = None
        
        # Ask for input files
        print("Input features file path (Required):")
        self.force_command(self.load_command, "--features", required=True)
        print("Input covariates file path (Optional):")
        self.force_command(self.load_command, "--covariates")
        print("Input covariate type to train seperate models (Optional):")
        self.force_command(self.covar_command)
        print("Input clinical file path (Optional):")
        self.force_command(self.load_command, "--clinical")
        print("Input systems file path (Optional):")
        self.force_command(self.load_command, "--systems")

        # Ask for scaler, model and CV parameters
        print("Scaler type and parameters (Default:standard)")
        print("Available: standard (from sklearn)")
        print("Example: standard with_mean=True with_std=False")
        self.force_command(self.scaler_command)
        print("Model type and parameters (Default:linear)")
        print("Available: linear (from sklearn)")
        print("Example: linear fit_intercept=True positive=False")
        self.force_command(self.model_command)
        print("CV parameters (Default: nº splits=5 and seed=0):")
        self.force_command(self.cv_command)

        # Run modelling capture any error raised and print
        try:
            self.run_wrapper(self.run_age)
            print('Finished running age modelling.')
        except Exception as e:
            print(e)
            error = "Error running age modelling."
        
        return error

    def model_command(self):
        """Load model parameters."""

        # Split into items and remove  command
        self.line = self.line.split()
        valid_types = ["linear"]
        error = None

        # Check that at least one argument input
        if len(self.line) == 0:
            error = "Must provide at least one argument or None."
            return error
        else:
            model_type = self.line[0]

        # Set model type or default
        if model_type == "None":
            self.args.model_type = "linear"
        else:
            if model_type not in valid_types:
                error = "Choose a valid model type: {}".format(valid_types)
            else:
                self.args.model_type = model_type

        # Set model parameters
        if len(self.line) > 1 and model_type != "None":
            model_params = {}
            for item in self.line[1:]:
                # Check that item has one = to split
                if item.count("=") != 1:
                    error = "Model parameters must be in the format param1=value1 param2=value2 ..."
                    return error
                key, value = item.split("=")
                value = convert(value)
                model_params[key] = value
            self.args.model_params = model_params
        else:
            self.args.model_params = {}

        return error

    def output_command(self):
        """Load output directory."""

        # Split into items and remove  command
        self.line = self.line.split()
        error = None

        # Check wether there is a path
        if len(self.line) == 0:
            error = "Must provide a path."
            return error

        # Check that path exists
        path = self.line[0]
        if len(self.line) > 1:
            error = "Too many arguments only one single path."
        elif os.path.isdir(path):
            self.args.output = path
        else:
            error = "Directory %s does not exist." % path

        return error

    def scaler_command(self):
        """Load scaler parameters."""

        # Split into items and remove  command
        self.line = self.line.split()
        error = None
        valid_types = ["standard"]

        # Check that at least one argument input
        if len(self.line) == 0:
            error = "Must provide at least one argument or None."
            return error
        else:
            scaler_type = self.line[0]

        # Set scaler type or default
        if scaler_type == "None":
            self.args.scaler_type = "standard"
        else:
            if scaler_type not in valid_types:
                error = "Choose a valid scaler type: {}".format(valid_types)
            else:
                self.args.scaler_type = scaler_type

        # Set scaler parameters
        if len(self.line) > 1 and scaler_type != "None":
            scaler_params = {}
            for item in self.line[1:]:
                if item.count("=") != 1:
                    error = "Scaler parameters must be in the format param1=value1 param2=value2 ..."
                    return error
                key, value = item.split("=")
                value = convert(value)
                scaler_params[key] = value
            self.args.scaler_params = scaler_params
        else:
            self.args.scaler_params = {}

        return error
