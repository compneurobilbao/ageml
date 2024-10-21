"""Implement the user interface.

Used in the AgeML project to enable the user to enter commands
to run the modelling with desired inputs.

Classes:
--------
Interface - reads, parses and executes user commands.
CLI - reads and parses user commands via command line.
InteractiveCLI - reads and parses user commands via command line via an interactive interface.
"""

import argparse
import numpy as np
import pandas as pd
import os
import warnings

from datetime import datetime
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
import scipy.stats as stats

import ageml.messages as messages
from ageml.visualizer import Visualizer
from ageml.utils import create_directory, feature_extractor, significant_markers, convert, log, NameTag
from ageml.modelling import AgeML, Classifier
from ageml.datasets.synthetic_data import rng
from ageml.processing import find_correlations, features_mutual_info, covariate_correction, cohen_d


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

    command_setup(self, dir_path): Create required directories and files to store results for command.

    set_flags(self): Set flags.

    set_visualizer(self): Set visualizer with output directory.

    set_dict(self): Initialise dictionaries for data storage.

    set_features_dataframes(self): Set features dataframes for each subject type, covariate and system.

    generate_model(self): Set model with parameters.

    set_classifier(self): Set classifier with parameters.

    update_params(self): Update initial parameters after load.

    check_file(self, file): Check that file exists.

    load_csv(self, file): Use panda to load csv into dataframe.

    load_features(self, required): Load features from csv file.

    load_covariates(self, required): Load covariates from csv file.

    load_clinical(self, required): Load clinical from csv file.

    load_factors(self, required): Load factors from csv file.

    load_ages(self, required): Load ages from csv file.

    system_parser(self, file): Parse systems file.

    load_systems(self, required): Load systems file.

    remove_missing_data(self): Remove subjects with missing values.

    load_data(self, required): Load data from csv files.

    age_distribution(self, dfs, labels=None): Use visualizer to show age distribution.

    features_vs_age(self, df, significance=0.05): Use visualizer to explore relationship between features and age.

    model_age(self, df, model): Use AgeML to fit age model with data.

    model_all(self): Model age for each system and covariate on controls.

    predict_age(self, df, model, model_name): Use AgeML to predict age with data.

    predict_all(self): Predict age for each system and covariate on all subject types excpet cn.

    save_predictions(self): Save age predictions to csv.

    model_age_and_clasify(self, features, model, tag): Use AgeML to fit model with data and classify.

    classify_with_features(self, features, tag): Classify two groups based on features.

    model_feature_anaylsis(self, df, tag): Use AgeML to fit model with data.

    classification_feature_analysis(self, df, tag): Use AgeML to fit model with data and classify.

    factors_vs_deltas(self, dfs_ages, dfs_factors, groups, significance=0.05): Calculate correlations between factors and deltas.

    deltas_by_group(self, df, labels): Calculate summary metrics of deltas by group.

    classify(self, df1, df2, groups): Classify two groups based on deltas.

    run_wrapper(self, run): Wrapper for running modelling with log.

    run_age(self): Run age modelling.

    run_model_feautre_influence(self): Run model feature influence analysis.

    run_age_model_vs_logistic_regression(self): Run age model vs logistic regression.

    run_factor_correlation(self): Factor correlation analysis between deltas and factors.

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

    def setup(self):
        """Create main directory."""

        # Create directory
        self.dir_path = os.path.join(self.args.output, "ageml")
        if os.path.exists(self.dir_path):
            warnings.warn("Directory %s already exists files may be overwritten." % self.dir_path, category=UserWarning)
        else:
            create_directory(self.dir_path)

        # Create .txt log file for which command run
        self.log_path = os.path.join(self.dir_path, "log.txt")
        with open(self.log_path, "a") as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(current_time + "\n")

    def command_setup(self, dir_path):
        """Create required directories and files to store results for command.

        Parameters
        ----------
        dir_path: directory path to create"""

        # Create directory
        self.command_dir = os.path.join(self.dir_path, dir_path)
        if os.path.exists(self.command_dir):
            warnings.warn("Directory %s already exists files may be overwritten." % self.command_dir, category=UserWarning)
        else:
            create_directory(self.command_dir)

        # Create .txt log file to save results and log time
        self.log_path = os.path.join(self.command_dir, "log.txt")
        with open(self.log_path, "a") as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(current_time + "\n")

        # Set visualizer as command directory
        self.set_visualizer(self.command_dir)

        # Reset flags
        self.set_flags()

        # Set initial parameters for model to defaults
        self.naming = ""
        self.subject_types = ["cn"]
        self.covars = ["all"]
        self.systems = ["all"]

    def set_flags(self):
        """Set flags."""

        self.flags = {
            "clinical": False,
            "covariates": False,
            "covarname": False,
            "systems": False,
            "ages": False,
            "features": False,
            "covcorr_mode": "cn",
        }

    def set_visualizer(self, dir):
        """Set visualizer with output directory."""

        self.visualizer = Visualizer(dir)

    def set_dict(self):
        """Initialise dictionaries for data storage."""

        self.dfs = {
            subject_type: {covar: {system: {} for system in self.systems} for covar in self.covars} for subject_type in self.subject_types
        }
        self.preds = {
            subject_type: {covar: {system: {} for system in self.systems} for covar in self.covars} for subject_type in self.subject_types
        }
        self.models = {covar: {system: {} for system in self.systems} for covar in self.covars}
        self.betas = {covar: {system: {} for system in self.systems} for covar in self.covars}

    def set_features_dataframes(self):
        """Set features dataframes for each subject type, covariate and system."""

        # Obtain dataframes for each subject type, covariate and system
        for subject_type in self.subject_types:
            # Keep only the subjects of the specified type
            df_sub = self.df_features[self.df_clinical[subject_type]]
            for covar in self.covars:
                # Keep subjects with the specified covariate
                if self.flags["covarname"]:
                    covar_index = set(self.df_covariates[self.df_covariates[self.args.covar_name] == covar].index)
                    df_cov = df_sub[df_sub.index.isin(covar_index)]
                else:
                    df_cov = df_sub
                for system in self.systems:
                    # Keep only the features of the system
                    df_sys = df_cov[["age"] + self.dict_systems[system]]
                    # Save the dataframe
                    self.dfs[subject_type][covar][system] = df_sys

    def generate_model(self, verbose=False):
        """Set model with parameters."""

        model = AgeML(
            self.args.scaler_type,
            self.args.scaler_params,
            self.args.model_type,
            self.args.model_params,
            self.args.model_cv_split,
            self.args.model_seed,
            self.args.hyperparameter_tuning,
            self.args.hyperparameter_params,
            self.args.feature_extension,
            verbose=verbose,
        )
        return model

    def generate_classifier(self, verbose=False):
        """Set classifier with parameters."""

        classifier = Classifier(self.args.classifier_cv_split, self.args.classifier_seed, self.args.classifier_thr, self.args.classifier_ci, verbose=verbose)

        return classifier

    def update_params(self):
        """Update initial parameters after load."""

        # Check possible flags of interest
        if self.flags["clinical"]:
            self.subject_types = self.df_clinical.columns.to_list()
        if self.flags["covarname"]:
            self.covars = pd.unique(self.df_covariates[self.args.covar_name]).tolist()
            self.naming += f"_{self.args.covar_name}"
        if self.flags["systems"]:
            self.systems = list(self.dict_systems.keys())
            self.naming += "_multisystem"
        elif self.flags["features"]:
            self.dict_systems["all"] = self.df_features.columns.drop("age").to_list()
        if self.flags["ages"]:
            self.systems = [col[6:] for col in self.df_ages.columns if "delta" in col]

    def check_file(self, file):
        """Check that file exists."""
        if not os.path.exists(file):
            return False
        else:
            return True

    def load_csv(self, file_type):
        """Use pandas to load csv into dataframe, making all columns lowercase.

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

    def load_features(self, required=False):
        """Load features from csv file.

        Parameters
        ----------
        required: boolean to check if file is required

        Returns
        -------
        df: dataframe with features"""

        # Load data
        df = self.load_csv("features")

        # Check that file
        if required and df is None:
            raise ValueError("Features file must be provided.")
        elif df is None:
            return df

        # Check that age column is present
        if "age" not in df:
            raise KeyError("Features file must contain a column name 'age', or any other case-insensitive variation.")

        # Check that columns are dtypes float or int
        error_cols = []
        for col in df.columns:
            if df[col].dtype not in [float, int]:
                error_cols.append(col)
        if error_cols != []:
            raise TypeError("Features file columns must be float or int type: %s" % (error_cols))

        # Set features flag
        self.flags["features"] = True

        return df

    def load_covariates(self, required=False):
        """Load covariates from csv file.

        Parameters
        ----------
        required: boolean to check if file is required

        Returns
        -------
        df: dataframe with covariates"""

        # Load data
        df = self.load_csv("covariates")

        # Check required
        if required and df is None:
            raise ValueError("Covariates file must be provided.")
        elif df is None:
            return df

        # Set covariate flag
        self.flags["covariates"] = True

        # Check that columns are dtypes float or int
        error_cols = []
        for col in df.columns:
            if df[col].dtype not in [float, int]:
                error_cols.append(col)
        if error_cols != []:
            raise TypeError("Covariates file columns must be float or int type: %s" % (error_cols))

        # Set covariate for analysis
        if hasattr(self.args, "covar_name") and self.args.covar_name is not None:
            self.flags["covarname"] = True
            self.args.covar_name = self.args.covar_name.lower()
            if self.args.covar_name not in df:
                raise KeyError("Covariate column %s not found in covariates file." % self.args.covar_name)

        # Check if covariate correction mode was specified in the arguments and set the flag accordingly
        if hasattr(self.args, "covcorr_mode") and self.args.covcorr_mode is not None:
            self.flags["covcorr_mode"] = self.args.covcorr_mode

        return df

    def load_clinical(self, required=False):
        """Load clinical from csv file.

        Parameters
        ----------
        required: boolean to check if file is required

        Returns
        -------
        df: dataframe with clinical information"""

        # Load data
        df = self.load_csv("clinical")

        # Check required
        if required and df is None:
            raise ValueError("Clinical file must be provided.")
        elif df is None:
            return df

        # Check that CN in columns and boolean type
        if "cn" not in df:
            raise KeyError("Clinical file must contain a column name 'CN' or any other case-insensitive variation.")

        # Iterate over each column in the DataFrame
        error_cols = []
        for column in df.columns:
            # Check if all values in the column are either 0 or 1
            if df[column].isin([0, 1]).all():
                # Convert the column to boolean type
                df[column] = df[column].astype(bool)
            else:
                error_cols.append(column)

        # Raise an error if the column contains values other than 0 or 1
        if error_cols != []:
            raise TypeError(f"Clinical file columns: {error_cols} contains values other than 0 and 1.")

        # Check that all columns have at least two subjects and show which column
        for col in df.columns:
            if df[col].sum() < 2:
                raise ValueError("Clinical column %s has less than two subjects." % col)

        # Find rows with all False
        if not df.any(axis=1).all():
            # Show which rows have all False
            rows = df[~df.any(axis=1)].index.to_list()
            raise ValueError("Clinical file contains rows with all False values. Please check the file. Rows: %s" % rows)

        # Set clinical flag
        self.flags["clinical"] = True

        return df

    def load_factors(self, required=False):
        """Load factors from csv file.

        Parameters
        ----------
        required: boolean to check if file is required

        Returns
        -------
        df: dataframe with factors"""

        # Load data
        df = self.load_csv("factors")

        # Check required
        if required and df is None:
            raise ValueError("Factors file must be provided.")
        elif df is None:
            return df

        # Check that columns are dtypes float or int
        error_cols = []
        for col in df.columns:
            if df[col].dtype not in [float, int]:
                error_cols.append(col)
        if error_cols != []:
            raise TypeError("Factors file columns must be float or int type: %s" % (error_cols))

        return df

    def load_ages(self, required=False):
        """Load ages from csv file.

        Parameters
        ----------
        required: boolean to check if file is required

        Returns
        -------
        df: dataframe with ages"""

        # Load data
        df = self.load_csv("ages")

        # Check required
        if required and df is None:
            raise ValueError("Ages file must be provided.")
        elif df is None:
            return df

        # Check that columns are dtypes float or int
        error_cols = []
        for col in df.columns:
            if df[col].dtype not in [float, int]:
                error_cols.append(col)
        if error_cols != []:
            raise TypeError("Ages file columns must be float or int type: %s" % (error_cols))

        # Required columns
        self.flags["ages"] = True
        req_cols = ["age", "predicted_age", "corrected_age", "delta"]
        cols = [col.lower() for col in df.columns.to_list()]

        # Check that columns are present
        for col in req_cols:
            if not any(c.startswith(col) for c in cols):
                raise KeyError("Ages file missing the following column %s, or derived names." % col)

        # Check that columns present are of the correct type
        for col in cols:
            if not any(col.startswith(c) for c in req_cols):
                raise KeyError("Ages file contains unknwon column %s" % col)

        return df

    def system_parser(self, file):
        """Parse systems file.

        Parameters
        ----------
        file: file to parse

        Returns
        -------
        dict_systems: dictionary with systems and their features"""

        # Initialise dictionary
        systems = {}

        # Load feature names
        features = {f.lower() for f in self.df_features.columns.to_list()}

        # Parse file
        for line in open(file, "r"):
            line = line.split("\n")[0]  # Remove newline character
            line = line.split(":")  # Split by the separator

            # Must have at exactly two elements (system, features)
            if len(line) != 2:
                raise ValueError("Systems file must be in the format 'system_name_1:feature1,feature2,...'")

            # Obtain system features ignoring trailing white spaces
            system = line[0].strip()
            systems_features = [f.lower().strip() for f in line[1].split(",")]

            # Check features exist and not repeated
            for f in systems_features:
                if f not in features:
                    raise ValueError("Feature '%s' not found in features file." % f)
                elif systems_features.count(f) > 1:
                    raise ValueError("Feature '%s' is repeated in the system: %s." % (f, system))

            # Save the system name and its features
            systems[system] = systems_features

        return systems

    def load_systems(self, required=False):
        """Load systems file.

        .txt expected. Format: system_name:feature1,feature2,...

        Parameters
        ----------
        required: boolean to check if file is required

        Returns
        -------
        systems: dictionary with systems and their features"""

        # Initialise dictionary
        systems = {}

        # Check if systems file is provided
        if hasattr(self.args, "systems"):
            if self.args.systems is not None and not self.check_file(self.args.systems):
                raise ValueError("Systems file '%s' not found." % self.args.systems)
            elif required and self.args.systems is None:
                raise ValueError("Systems file must be provided.")
            elif self.args.systems is None:
                return systems
        else:
            return systems

        # Set flag
        self.flags["systems"] = True

        # Parse file
        systems = self.system_parser(self.args.systems)

        # Check that the dictionary has at least one entry
        if len(systems) == 0:
            raise ValueError("Systems file is probably incorrectly formatted. Check it please.")

        return systems

    def remove_missing_data(self):
        """Remove subjects with missing values."""

        # Dataframes
        dfs = {
            "features": self.df_features,
            "covariates": self.df_covariates,
            "factors": self.df_factors,
            "clinical": self.df_clinical,
            "ages": self.df_ages,
        }
        dfs = {label: df for label, df in dfs.items() if df is not None}

        # Subjects before removing missing data
        init_count = {label: len(df) for label, df in dfs.items()}
        for label in dfs.keys():
            print("Number of subjects in dataframe %s: %d" % (label, init_count[label]))

        # Check for missing data
        print("Removing subjects with missing data...")
        for label, df in dfs.items():
            missing_subjects = df[df.isnull().any(axis=1)].index.to_list()
            if missing_subjects.__len__() != 0:
                warn_message = "Number of subjects with missing data in %s dataframe: %s" % (label, len(missing_subjects))
                print(warn_message)
                warnings.warn(warn_message, category=UserWarning)
                with open(os.path.join(self.command_dir, f"{label}_subjects_with_missing_data.txt"), "w") as f:
                    f.write("\n".join(map(str, missing_subjects)))
                    pass
                dfs[label] = df.drop(missing_subjects)

        # Compute the intersection of the indices of the dataframes
        indices_collection = [set(df.index) for df in dfs.values()]
        shared_idx = list(indices_collection[0].intersection(*indices_collection[1:]))
        # Remove subjects not shared among dataframes, report subjects left for analysis, and set dataframes
        print("Removing subjects not shared among dataframes...")
        for label in dfs.keys():
            removed_subjects = set(dfs[label].index) - set(shared_idx)
            if removed_subjects.__len__() != 0:
                warn_message = f"{len(removed_subjects)} subjects removed from {label} dataframe."
                dfs[label] = dfs[label].loc[shared_idx]
                print(warn_message)
                warnings.warn(warn_message, category=UserWarning)
                # Save removed subjects in a txt file
                with open(os.path.join(self.command_dir, f"{label}_nonshared_subjects.txt"), "w") as f:
                    f.write("\n".join(map(str, removed_subjects)))
            setattr(self, f"df_{label}", dfs[label])
            msg = (
                f"Final number of subjects in dataframe {label}: {len(shared_idx)} "
                f"({len(shared_idx) / init_count[label] * 100:.2f} % of initial)"
            )
            print(msg)

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

        # Load Features
        self.df_features = self.load_features(required="features" in required)

        # Load covariates
        self.df_covariates = self.load_covariates(required="covariates" in required)

        # Load clinical
        self.df_clinical = self.load_clinical(required="clinical" in required)

        # Load factors
        self.df_factors = self.load_factors(required="factors" in required)

        # Load ages
        self.df_ages = self.load_ages(required="ages" in required)

        # Load systems
        self.dict_systems = self.load_systems(required="systems" in required)

        # Removes subjects with missing values
        self.remove_missing_data()

        # Show number of subjects per clinical category
        if self.flags["clinical"]:
            for col in self.df_clinical.columns:
                print("Number of %s subjects: %d" % (col, self.df_clinical[col].sum()))
        else:
            print("No clinical information provided using all subjects as CN.")
            if self.df_features is not None:
                index = self.df_features.index
            elif self.df_ages is not None:
                index = self.df_ages.index
            self.df_clinical = pd.DataFrame(index=index, columns=["cn"], data=True)

        # Update initial parameters after load
        self.update_params()

    def age_distribution(self, ages_dict: dict, name=""):
        """Use visualizer to show age distribution.

        Parameters
        ----------
        ages_dict: dictionary containing ages for each group
        name: name to give to visualizer to save file"""

        # Select age information
        print("-----------------------------------")
        print("Age distribution of %s" % name)
        for key, vals in ages_dict.items():
            print("[Group: %s]" % key)
            print("Mean age: %.2f" % np.mean(vals))
            print("Std age: %.2f" % np.std(vals))
            print("Age range: [%d,%d]" % (np.min(vals), np.max(vals)))

        # Obtain labels and ages
        labels = list(ages_dict.keys())
        ages = list(ages_dict.values())

        # Check that distributions of ages are similar if more than one
        if len(ages_dict) > 1:
            print("Checking that age distributions are similar using T-test: T-stat (p_value)")
            print("If p_value > 0.05 distributions are considered simiilar and not displayed...")
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    t_stat, p_val = stats.ttest_ind(ages[i], ages[j])
                    if p_val < 0.05:
                        warn_message = "Age distributions %s and %s are not similar: %.2f (%.2g) " % (labels[i], labels[j], t_stat, p_val)
                        print(warn_message)
                        warnings.warn(warn_message, category=UserWarning)

        # Use visualiser
        self.visualizer.age_distribution(ages, labels, name)

    def features_vs_age(
        self,
        features_dict: dict,
        tag,
        significance: float = 0.05,
    ):
        """Use visualizer to explore relationship between features and age.

        Parameters
        ----------
        features_dict: dictionary containing features for each group
        significance: significance level for correlation"""

        # Select data to visualize
        print("-----------------------------------")
        print("Features by correlation with Age of Controls [System: %s]" % tag.system)
        print("significance: %.2g * -> FDR, ** -> bonferroni" % significance)

        # Make lists to store covariate info for each dataframe
        X_list, y_list, corr_list, order_list, significance_list = [], [], [], [], []
        for label, df in features_dict.items():
            print("Covariate %s" % label)
            # Extract features
            X, y, feature_names = feature_extractor(df)
            # Covariate correction
            if self.flags["covariates"] and not self.flags["covarname"]:
                print("Covariate effects will be subtracted from features.")
                X, _ = covariate_correction(X, self.df_covariates.loc[df.index].to_numpy())
            # Calculate correlation between features and age
            corr, order, p_values = find_correlations(X, y)
            # Reject null hypothesis of no correlation
            reject_bon, _, _, _ = multipletests(p_values, alpha=significance, method="bonferroni")
            reject_fdr, _, _, _ = multipletests(p_values, alpha=significance, method="fdr_bh")
            significant = significant_markers(reject_bon, reject_fdr)
            # Print results
            for idx, order_element in enumerate(order):
                print(
                    "%d.%s %s %s: %.2f (%.2g)"
                    % (
                        idx + 1,
                        label,
                        significant[order_element],
                        feature_names[order_element],
                        corr[order_element],
                        p_values[order_element],
                    )
                )
            # Append all the values
            X_list.append(X), y_list.append(y), corr_list.append(corr), order_list.append(order), significance_list.append(significant)

        # Use visualizer to show results
        self.visualizer.features_vs_age(
            X_list, y_list, corr_list, order_list, significance_list, feature_names, tag, list(features_dict.keys())
        )

    def feature_ordering(self, tag):
        """Order features by correlation with age.

        Parameters
        ----------
        df: dataframe with features and age; shape=(n,m+1)
        tag: NameTag object"""

        # Show feature ordering
        print("-----------------------------------")
        print("Feature ordering")

        # Relationship between features and age of controls
        print("Features mutual information with Age of Controls [System: %s]" % tag.system)

        # Extract features for controls
        X, age, feature_names = feature_extractor(self.dfs["cn"][tag.covar][tag.system])

        # Use mutual information to order acording to age
        order_age, mi_scores_age = features_mutual_info(X, age)

        # Print results
        for idx, order_element in enumerate(order_age):
            print("%d. %s: %.2f" % (idx + 1, feature_names[order_element], mi_scores_age[order_element]))
    
        # Relationiship between features and discrimination between groups
        print("Features mutual information for Discrimination between Groups %s and %s [System: %s]" % (self.args.group1, self.args.group2, tag.system))

        # Extract feature for each group
        df1 = self.dfs[self.args.group1][tag.covar][tag.system]
        df2 = self.dfs[self.args.group2][tag.covar][tag.system]
        X1, _, _ = feature_extractor(df1)
        X2, _, _ = feature_extractor(df2)

        # Use mutual information to order according to discrimination between groups
        labels = np.concatenate((np.zeros(X1.shape[0]), np.ones(X2.shape[0])))
        X = np.concatenate((X1, X2))
        order_discrimination, mi_scores_discrimination = features_mutual_info(X, labels)

        # Print results
        for idx, order_element in enumerate(order_discrimination):
            print("%d. %s: %.2f" % (idx + 1, feature_names[order_element], mi_scores_discrimination[order_element]))
        

    def model_age(self, df, model, tag):
        """Use AgeML to fit age model with data.

        Parameters
        ----------
        df: dataframe with features and age; shape=(n,m+1)
        model: AgeML object
        name: name of the model"""

        # Show training pipeline
        print("-----------------------------------")
        print(f"Training Age Model [Covariate:{tag.covar}, System:{tag.system}]")
        print(model.pipeline)

        # Select data to model
        X, y, _ = feature_extractor(df)

        # Throw error if we do not have enough controls for modelling
        if X.shape[0] / self.args.model_cv_split < 2:
            raise ValueError("Not enough controls for modelling for each CV split.")

        # Covariate correction
        if self.flags["covariates"] and not self.flags["covarname"]:
            print("Covariate effects will be subtracted from features.")
            X, beta = covariate_correction(X, self.df_covariates.loc[df.index].to_numpy())
        else:
            beta = None

        # Fit model and plot results
        y_pred, y_corrected = model.fit_age(X, y)
        self.visualizer.true_vs_pred_age(y[model.test_indices], y_pred[model.test_indices], tag)
        self.visualizer.age_bias_correction(y[model.test_indices], y_pred[model.test_indices], y_corrected[model.test_indices], tag)

        # Calculate deltas
        deltas = np.full(np.shape(y), np.nan)
        deltas[model.test_indices] = y_corrected[model.test_indices] - y[model.test_indices]

        # Save to dataframe and csv
        data = np.stack((y, y_pred, y_corrected, deltas), axis=1)
        cols = ["age", "predicted_age", "corrected_age", "delta"]
        df_ages = pd.DataFrame(data, index=df.index, columns=cols)

        return model, df_ages, beta

    def model_all(self):
        """Model age for each system and covariate on controls."""
        # Get the test indices for the models
        # Iterate over covariate and system
        for covar in self.covars:
            for system in self.systems:
                train_indices, test_indices = self.__get_test_indices(covar, system)
                covar_tag = self.args.covar_name + "_" + str(covar) if self.flags["covarname"] else covar
                tag = NameTag(covar=covar_tag, system=system)
                # Generate model
                ageml_model = self.generate_model(verbose=True)
                data = self.dfs["cn"][covar][system]
                ageml_model.train_indices = train_indices
                ageml_model.test_indices = test_indices
                model, df_pred, betas = self.model_age(data, ageml_model, tag=tag)
                self.models[covar][system] = model
                self.betas[covar][system] = betas
                # Save predictions
                df_pred = df_pred.drop(columns=["age"])
                df_pred.rename(columns=lambda x: f"{x}_{system}", inplace=True)
                self.preds["cn"][covar][system] = df_pred

    def __get_test_indices(self, covar, system):
        """Gets and sets the train and test indices to keep track of them when running
        the model_age pipeline. Necessary to inform the user that when hyperopt was
        selected as the model, no CV was used, and instead a train test split was done.
        """
        data = self.dfs["cn"][covar][system]
        X, y, _ = feature_extractor(data)

        if self.args.model_type == "hyperopt":
            # Split the data in training and test sets
            # TODO: Substitute 0.2 factor by the user CLI argument
            test_size = int(0.2 * y.shape[0])
            indices = rng.permutation(X.shape[0])
            train_indices = indices[:-test_size]
            test_indices = indices[-test_size:]

        else:
            train_indices = list(range(X.shape[0]))
            test_indices = list(range(X.shape[0]))

        return train_indices, test_indices

    def predict_age(self, df, model, tag: NameTag, beta: np.ndarray = None):
        """Use AgeML to predict age with data."""

        # Show prediction pipeline
        print("-----------------------------------")
        print(f"Predicting for {tag.group}")
        print(f"with Age Model [Covariate:{tag.covar}, System:{tag.system}]")
        print(model.pipeline)

        # Select data to model
        X, y, _ = feature_extractor(df)

        # Covariate correction
        if self.flags["covariates"] and not self.flags["covarname"]:
            print("Covariate effects will be subtracted from features.")
            X, _ = covariate_correction(X, self.df_covariates.loc[df.index].to_numpy(), beta)

        # Predict age
        y_pred, y_corrected = model.predict_age(X, y)

        # Calculate deltas
        deltas = y_corrected - y

        # Save to dataframe and csv
        data = np.stack((y, y_pred, y_corrected, deltas), axis=1)
        cols = ["age", "predicted_age", "corrected_age", "delta"]
        df_ages = pd.DataFrame(data, index=df.index, columns=cols)

        return df_ages

    def predict_all(self):
        """Predict age for each system and covariate on all subject types excpet cn."""

        for subject_type in self.subject_types:
            # Do not apply to controls
            if subject_type == "cn":
                continue
            for covar in self.covars:
                for system in self.systems:
                    covar_tag = self.args.covar_name + "_" + str(covar) if self.flags["covarname"] else covar
                    tag = NameTag(group=subject_type, covar=covar_tag, system=system)
                    df_pred = self.predict_age(
                        self.dfs[subject_type][covar][system], self.models[covar][system], tag, self.betas[covar][system]
                    )
                    df_pred = df_pred.drop(columns=["age"])
                    df_pred.rename(columns=lambda x: f"{x}_{system}", inplace=True)
                    self.preds[subject_type][covar][system] = df_pred

    def save_predictions(self):
        """Save age predictions to csv."""

        # Concatenate predictions into a DataFrame
        stack = []
        for subject_type in self.subject_types:
            for covar in self.covars:
                df_systems = pd.concat([self.preds[subject_type][covar][system] for system in self.systems], axis=1)
                stack.append(df_systems)
        df_ages = pd.concat(stack, axis=0)

        # Drop duplicates keep first (some subjects may be in more than one subject type)
        df_ages = df_ages[~df_ages.index.duplicated(keep="first")]

        # Add age information
        df_ages = pd.concat([self.df_features["age"], df_ages], axis=1)

        # Handle NaNs
        df_ages = df_ages.fillna("")

        # Save dataframe to csv
        filename = "predicted_age" + self.naming + ".csv"
        df_ages.to_csv(os.path.join(self.command_dir, filename))

        # Save test indices if available
        if self.args.model_type == "hyperopt":
            for covar in self.covars:
                for system in self.systems:
                    filename = f"test_indices_{self.args.covar_name}_{covar}_system_{system}" + ".csv"
                    indices_path = os.path.abspath(os.path.join(self.command_dir, filename))
                    indices = self.models[covar][system].test_indices
                    df_indices = pd.DataFrame(indices, columns=["test_indices"])
                    df_indices.to_csv(indices_path, index=False)

            msg = (
                f"Saved test indices to {os.path.abspath(self.command_dir)} because a train test "
                "split was used forced by the election of the 'hyperopt' model."
            )
            print(msg)
            
    def model_age_and_classify(self, features, model, tag):
        """Train model to predict age and classify between groups using deltas."""

        # Get subset for controls, group1 and group2
        df_subset_cn = self.dfs["cn"][tag.covar][tag.system][['age'] + features]
        df_subset_group1 = self.dfs[self.args.group1][tag.covar][tag.system][['age'] + features]
        df_subset_group2 = self.dfs[self.args.group2][tag.covar][tag.system][['age'] + features]
        
        # Train model based on controls
        X_cn, y_cn, _ = feature_extractor(df_subset_cn) 
        y_pred_cn, y_corrected_cn = model.fit_age(X_cn, y_cn)

        # Calculate metrics
        age_metrics = model.metrics.get_summary()
        mae, mae_std  = age_metrics['test']['mae']['mean'], age_metrics['test']['mae']['std']
    
        # For each group now predict except if they are controls use CV predicted values
        if self.args.group1 != 'cn':
            X_group1, y_group1, _ = feature_extractor(df_subset_group1)
            _, y_corrected_group1 = model.predict_age(X_group1, y_group1)
            deltas_group1 = y_corrected_group1 - y_group1
        else:
            deltas_group1 = y_corrected_cn - y_cn
        if self.args.group2 != 'cn':
            X_group2, y_group2, _ = feature_extractor(df_subset_group2)
            _, y_corrected_group2 = model.predict_age(X_group2, y_group2)
            deltas_group2 = y_corrected_group2 - y_group2
        else:
            deltas_group2 = y_corrected_cn - y_cn

        # Subsampling bigger group to make same size as smaller group
        min_samples = min(len(deltas_group1), len(deltas_group2))
        deltas_group1_subsampled = np.random.choice(deltas_group1, min_samples, replace=False)
        deltas_group2_subsampled = np.random.choice(deltas_group2, min_samples, replace=False)

        # Classify between groups using deltas
        deltas = np.concatenate((deltas_group1_subsampled, deltas_group2_subsampled)).reshape(-1, 1)
        labels = np.concatenate((np.zeros(deltas_group1_subsampled.shape), np.ones(deltas_group2_subsampled.shape)))
        self.classifier = self.generate_classifier()
        _ = self.classifier.fit_model(deltas, labels)

        # Calculate AUC
        summary_metrics = self.classifier.metrics.get_summary()
        auc = summary_metrics['test']['auc']['mean']
        auc_std = summary_metrics['test']['auc']['std']

        return mae, mae_std, auc, auc_std, deltas_group1, deltas_group2

    def classify_with_features(self, X_group1, X_group2):
        """Classify between groups using features."""

        # Subsample bigger group to make same size as smaller group
        min_samples = min(X_group1.shape[0], X_group2.shape[0])
        X_group1 = X_group1[np.random.permutation(X_group1.shape[0])[:min_samples]]
        X_group2 = X_group2[np.random.permutation(X_group2.shape[0])[:min_samples]]

        # Concatenate features
        X = np.concatenate((X_group1, X_group2))
        y = np.concatenate((np.zeros(X_group1.shape[0]), np.ones(X_group2.shape[0])))

        # Classify between groups using features
        self.classifier = self.generate_classifier()
        _ = self.classifier.fit_model(X, y, scale=True)

        # Calculate AUC
        summary_metrics = self.classifier.metrics.get_summary()
        auc = summary_metrics['test']['auc']['mean']
        auc_std = summary_metrics['test']['auc']['std']

        return auc, auc_std

    def model_feature_analysis(self, order_age, order_discrimination, tag):
        """Train different models using the specificed order of features."""

        print("-----------------------------------")
        print("Training different models with different set of features")
        print('Using groups: "%s" and "%s"' % (self.args.group1, self.args.group2))

        # Obtain features of controls
        _, _, feature_names = feature_extractor(self.dfs["cn"][tag.covar][tag.system]) 

        # For each order iterate over all sets of features in order
        order_type = ['age', 'discrimination']

        for order, type in zip([order_age, order_discrimination], order_type):

            print("Ordering by: %s" % type)
            print("Feature added: MAE ± std, AUC ± std")

            # Metrics
            maes, maes_std, aucs, aucs_std = [], [], [], []

            # Iterate over each order to create subsets
            for i in range(len(order)):

                # Get subset of features
                features = [feature_names[o] for o in order[:i+1]]
                
                # Model age and classify
                model = self.generate_model()
                mae, mae_std, auc, auc_std, _, _ = self.model_age_and_classify(features, model, tag)
            
                # Store results
                maes.append(mae)
                maes_std.append(mae_std)
                aucs.append(auc)
                aucs_std.append(auc_std)

                # Print results nicely with feature added and number its on mae +- mae_std and auc +- auc_std
                print("%d. %s: %.2f ± %.2f, %.2f ± %.2f" % (i + 1, feature_names[order[i]], mae, mae_std, auc, auc_std))

            # Convert maes, aucs to numpy arrays
            maes, maes_std, aucs, auc_std  = np.array(maes), np.array(maes_std), np.array(aucs), np.array(aucs_std)
            
            # Visualize results
            self.visualizer.metrics_vs_num_features(maes, maes_std, aucs, aucs_std, type)

    def classification_feature_analysis(self, order, tag):
        """Train classifiers with different number of features and deltas from different models."""

        print("-----------------------------------")
        print("Training different models with different number of features")
        print('Using groups: "%s" and "%s"' % (self.args.group1, self.args.group2))

        # Obtain features of controls
        _, _, feature_names = feature_extractor(self.dfs["cn"][tag.covar][tag.system]) 

        # Models to train
        models = ['linear_reg', 'ridge', 'logistic_regression']

        # Metrics
        aucs, aucs_std = {}, {}

        # Iterate over each model type
        for model_type in models:
            print("Model: %s" % model_type)

            # Arrays for storage
            aucs[model_type], aucs_std[model_type] = [], []

            # Iterate over each order to create subsets
            for i in range(len(order)):

                # Get subset of features
                features = [feature_names[o] for o in order[:i+1]]
                
                # Model age and classify
                if model_type != 'logistic_regression':
                    model = AgeML('standard', {}, model_type, {}, self.args.model_cv_split, self.args.model_seed)
                    _, _, auc, auc_std, _, _ = self.model_age_and_classify(features, model, tag)
                else:
                    
                    # Get subset for controls, group1 and group2
                    df_subset_group1 = self.dfs[self.args.group1][tag.covar][tag.system][['age'] + features]
                    df_subset_group2 = self.dfs[self.args.group2][tag.covar][tag.system][['age'] + features]

                    # Extract features
                    X_group1, _, _ = feature_extractor(df_subset_group1)
                    X_group2, _, _ = feature_extractor(df_subset_group2)
                    auc, auc_std = self.classify_with_features(X_group1, X_group2)
            
                # Store results
                aucs[model_type].append(auc)
                aucs_std[model_type].append(auc_std)

        # TODO print resutls

        # Visualize results
        self.visualizer.auc_vs_num_features(aucs, aucs_std, 'linear_reg')

    def classifcication_analysis(self, tag):
        """Train classifiers with different sets of features."""

        print("-----------------------------------")
        print("Training different models with different sets of features")
        print('Using groups: "%s" and "%s"' % (self.args.group1, self.args.group2))
        print("AUC ± std")

        # Get subset for controls, group1 and group2
        df_subset_group1 = self.dfs[self.args.group1][tag.covar][tag.system]
        df_subset_group2 = self.dfs[self.args.group2][tag.covar][tag.system]

        # Classify using features directly
        X_group1, age_group1, feature_names = feature_extractor(df_subset_group1)
        X_group2, age_group2, _ = feature_extractor(df_subset_group2)
        auc_features, auc_std_features = self.classify_with_features(X_group1, X_group2)
        print("Features: %.2f ± %.2f" % (auc_features, auc_std_features))

        # Classify using features and age directly
        X_group1_f_age = np.concatenate((X_group1, age_group1.reshape(-1, 1)), axis=1)
        X_group2_f_age = np.concatenate((X_group2, age_group2.reshape(-1, 1)), axis=1)
        auc_features_age, auc_std_features_age = self.classify_with_features(X_group1_f_age, X_group2_f_age)
        print("Features + Age: %.2f ± %.2f" % (auc_features_age, auc_std_features_age))

        # Classify using deltas
        model = AgeML('standard', {}, 'linear_reg', {}, self.args.model_cv_split, self.args.model_seed)
        _, _, auc_deltas, auc_std_deltas, deltas_group1, deltas_group2 = self.model_age_and_classify(feature_names, model, tag)
        print("Deltas: %.2f ± %.2f" % (auc_deltas, auc_std_deltas))


        # Classify using features and deltas
        X_group1_f_delta = np.concatenate((X_group1, deltas_group1.reshape(-1, 1)), axis=1)
        X_group2_f_delta = np.concatenate((X_group2, deltas_group2.reshape(-1, 1)), axis=1)
        auc_features_delta, auc_std_features_delta = self.classify_with_features(X_group1_f_delta, X_group2_f_delta)
        print("Features + Deltas: %.2f ± %.2f" % (auc_features_delta, auc_std_features_delta))        
        

    def factors_vs_deltas(self, dict_ages, df_factors, tag, covars=None, beta=None, significance=0.05):
        """Calculate correlations between factors and deltas.

        Parameters
        ----------
        # TODO: Update docstring for the current input arguments
        dict_ages: dictionary for each system with deltas; shape=(n,m)
        df_factors: dataframe with factor information; shape=(n,m)
        significance: significance level for correlation"""

        # Select age information
        print("-----------------------------------")
        print("Correlations between lifestyle factors for %s" % tag.group)
        print("significance: %.2g * -> FDR, ** -> bonferroni" % significance)

        # Iterate over systems
        corrs, significants = [], []

        # Factor information
        factors = df_factors.to_numpy()
        factor_names = df_factors.columns.to_list()

        # Apply covariate correction
        if self.flags["covariates"]:
            factors, _ = covariate_correction(factors, covars, beta)

        for system, df in dict_ages.items():
            print(f"System: {system}")

            # Select data to visualize
            deltas = df["delta_%s" % system].to_numpy()

            # Calculate correlation between features and age
            corr, order, p_values = find_correlations(factors, deltas)
            corrs.append(corr)

            # Reject null hypothesis of no correlation
            reject_bon, _, _, _ = multipletests(p_values, alpha=significance, method="bonferroni")
            reject_fdr, _, _, _ = multipletests(p_values, alpha=significance, method="fdr_bh")
            significant = significant_markers(reject_bon, reject_fdr)
            significants.append(significant)

            # Print results
            for i, o in enumerate(order):
                print("%d. %s %s: %.2f (%.2g)" % (i + 1, significant[o], factor_names[o], corr[o], p_values[o]))

        # Use visualizer to show bar graph
        self.visualizer.factors_vs_deltas(corrs, list(dict_ages.keys()), factor_names, significants, tag)

    def deltas_by_group(self, dfs, tag, significance: float = 0.05):
        """Calculate summary metrics of deltas by group.

        Parameters
        ----------
        # TODO: Update docstring for the current input arguments
        dfs: list of dataframes with delta information; shape=(n,m)
        labels: list of labels for each dataframe; shape=(n,)
        system: name of the system from which the variables come from"""

        # Select age information
        print("-----------------------------------")
        print(f"Delta distribution for System: {tag.system}")

        # Compute covariate correction coefficients
        if self.flags["covariates"]:
            beta = {}
            # When "cn" -> Only use cn group for computing beta
            if self.flags["covcorr_mode"] == "cn":
                print("Correcting for covariates using CN group as reference.")
                df_group = dfs["cn"]
                group_idx = df_group.index
                covars = self.df_covariates.loc[group_idx].to_numpy()
                deltas = df_group[f"delta_{tag.system}"].to_numpy()
                _, beta["cn"] = covariate_correction(deltas, covars)
            # When "all" -> Use whole dataset for computing beta
            elif self.flags["covcorr_mode"] == "all":
                print("Correcting for covariates using whole population as reference.")
                # Concatenate all groups to get the whole dataset
                df_group = pd.concat(dfs, axis=0)
                # Because all indices are used, get them from the covariates dataframe
                group_idx = self.df_covariates.index
                covars = self.df_covariates.to_numpy()
                deltas = df_group[f"delta_{tag.system}"].to_numpy()
                _, beta["all"] = covariate_correction(deltas, covars)
            # When "each" -> Use each group for computing betas
            elif self.flags["covcorr_mode"] == "each":
                print("Correcting for covariates of each group separately.")
                for group, df_group in dfs.items():
                    group_idx = df_group.index
                    covars = self.df_covariates.loc[group_idx].to_numpy()
                    deltas = df_group[f"delta_{tag.system}"].to_numpy()
                    _, beta[group] = covariate_correction(deltas, covars)

        # Obtain deltas means and stds
        deltas = []
        for group, df_group in dfs.items():
            vals = df_group[f"delta_{tag.system}"].to_numpy()
            # Apply covariate correction coefficients
            if self.flags["covariates"]:
                covars = self.df_covariates.loc[df_group.index].to_numpy()
                # When "cn" or "all" -> Use the same beta for all groups
                if self.flags["covcorr_mode"] in ["cn", "all"]:
                    vals, _ = covariate_correction(vals, covars, beta[self.flags["covcorr_mode"]])
                # When "each" -> Use the specific beta for each group
                elif self.flags["covcorr_mode"] == "each":
                    vals, _ = covariate_correction(vals, covars, beta[group])
            deltas.append(vals)
            print(f"[Group: {group}]")
            print("Mean delta: %.2f" % np.mean(vals))
            print("Std delta: %.2f" % np.std(vals))
            print("Delta range: [%d, %d]" % (np.min(vals), np.max(vals)))

        # Obtain statistically significant difference between deltas
        print("Checking for statistically significant differences between deltas...")
        header = (
            "p-val (group1 vs. group2)            "
            "Cohen's d              "
            "CI (95%):   [low, upp]          "
            "Delta difference(mean1 - mean2)"
        )
        print(len(str(header)) * "=")
        print("significance: %.2g * -> FDR, ** -> bonferroni" % significance)
        print(str(header))
        print(len(str(header)) * "-")

        # Calculate p-values
        p_vals_matrix = np.zeros((len(deltas), len(deltas)))
        effect_size_matrix = np.zeros((len(deltas), len(deltas)))
        conf_intervals = np.zeros((len(deltas), len(deltas), 2))  # Lower & upper bounds

        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                _, p_val = stats.ttest_ind(deltas[i], deltas[j])
                p_vals_matrix[i, j] = p_val
                effect_size_matrix[i, j] = cohen_d(deltas[i], deltas[j])
                # Compute confidene interval for the mean difference
                cm = CompareMeans(DescrStatsW(deltas[i]), DescrStatsW(deltas[j]))
                # We use unequal variance, Welch's test and Satterthwaite's dofs
                ci_low, ci_upp = cm.tconfint_diff(alpha=significance, usevar="unequal")
                conf_intervals[i, j] = [ci_low, ci_upp]

        # Reject null hypothesis of no correlation
        reject_bon, _, _, _ = multipletests(p_vals_matrix.flatten(), alpha=significance, method="bonferroni")
        reject_fdr, _, _, _ = multipletests(p_vals_matrix.flatten(), alpha=significance, method="fdr_bh")
        reject_bon = reject_bon.reshape((len(deltas), len(deltas)))
        reject_fdr = reject_fdr.reshape((len(deltas), len(deltas)))

        # Print results
        labels = list(dfs.keys())
        for i in range(len(deltas)):
            significant = significant_markers(reject_bon[i], reject_fdr[i])
            for j in range(i + 1, len(deltas)):
                pval_message = "p-val (%s vs. %s): %.2g" % (labels[i], labels[j], p_vals_matrix[i, j])
                if significant[j] != "":
                    pval_message = significant[j] + " " + pval_message
                cohen_d_value = effect_size_matrix[i, j]
                delta_difference = np.mean(deltas[i]) - np.mean(deltas[j])
                ci_low, ci_upp = conf_intervals[i, j]
                ci_p = 100 * (1 - significance)  # Confidence interval percentage
                pval_message = f"{pval_message:<35}  Cohen's d: {cohen_d_value:<10.2f}  CI ({ci_p}%): [{ci_low:<6.2f}, {ci_upp:<6.2f}]"
                pval_message += f"    Delta difference: {delta_difference:<15.2f}"
                print(pval_message)

        # Use visualizer
        self.visualizer.deltas_by_groups(deltas, labels, tag)

    def classify(self, df1, df2, groups, tag, beta: np.ndarray = None):
        """Classify two groups based on deltas.

        Parameters
        ----------
        df1: dataframe with delta information; shape=(n,m)
        df2: dataframe with delta information; shape=(n,m)
        groups: list of labels for each dataframe; shape=(2,)
        system: name of the system from which the variables come from
        beta: coefficients for covariate correction"""

        # Classification
        print("-----------------------------------")
        print(f"Classification between groups {groups[0]} and {groups[1]} [System: {tag.system}]")

        # Select delta information
        delta_cols = [col for col in df1.columns if "delta" in col]
        deltas1 = df1[delta_cols].to_numpy()
        deltas2 = df2[delta_cols].to_numpy()

        # Covariate correction
        if self.flags["covariates"]:
            beta = {}
            if self.flags["covcorr_mode"] == "cn":
                print("Applying covariate correction for deltas using CN group as reference.")
                group_index = self.df_clinical["cn"]
                df_cn = self.df_ages[group_index]
                covars = self.df_covariates.loc[group_index].to_numpy()
                deltas = df_cn[delta_cols].to_numpy()
                _, beta["cn"] = covariate_correction(deltas, covars)
            elif self.flags["covcorr_mode"] == "all":
                print("Applying covariate correction for deltas using whole population as reference.")
                covars = self.df_covariates.to_numpy()
                deltas = self.df_ages[delta_cols].to_numpy()
                _, beta["all"] = covariate_correction(deltas, covars)
            elif self.flags["covcorr_mode"] == "each":
                print("Applying covariate correction for deltas using each clinical group as reference.")
                for group in self.subject_types:
                    group_index = self.df_clinical[group]
                    deltas = self.df_ages[group_index][delta_cols].to_numpy()
                    covars = self.df_covariates.loc[group_index].to_numpy()
                    _, beta[group] = covariate_correction(deltas, covars)

            covars1 = self.df_covariates.loc[df1.index].to_numpy()
            covars2 = self.df_covariates.loc[df2.index].to_numpy()

            if self.flags["covcorr_mode"] in ["cn", "all"]:
                deltas1, _ = covariate_correction(deltas1, covars1, beta[self.flags["covcorr_mode"]])
                deltas2, _ = covariate_correction(deltas2, covars2, beta[self.flags["covcorr_mode"]])
            elif self.flags["covcorr_mode"] == "each":
                deltas1, _ = covariate_correction(deltas1, covars1, beta[groups[0]])
                deltas2, _ = covariate_correction(deltas2, covars2, beta[groups[1]])

        # Create X and y for classification
        if len(delta_cols) == 1:
            X = np.concatenate((deltas1, deltas2)).reshape(-1, 1)
            y = np.concatenate((np.zeros(deltas1.shape), np.ones(deltas2.shape)))
        else:
            X = np.concatenate((deltas1, deltas2))
            y = np.concatenate((np.zeros(deltas1.shape[0]), np.ones(deltas2.shape[0])))

        # Throw error if we do not have enough controls for modelling
        if X.shape[0] / self.args.classifier_cv_split < 2:
            raise ValueError("Not enough subjects for classification for each CV split.")

        # Generate classifier
        self.classifier = self.generate_classifier(verbose=True)

        # Apply covariate correction
        # TODO: AGAIN? Commented for now.
        # if self.flags["covariates"]:
        #     Z = np.concatenate((self.df_covariates.loc[df1.index].to_numpy(),
        #                         self.df_covariates.loc[df2.index].to_numpy()))
        #     X, _ = covariate_correction(X, Z, beta)

        # Calculate classification
        y_pred = self.classifier.fit_model(X, y)

        # Print regressor weights
        if len(delta_cols) > 1:
            print("Logistic regressor weigths coef (norm_coef):")
            coefs = self.classifier.model.coef_[0]
            max_coef = np.max(np.abs(coefs))
            for coef, delta in zip(coefs, delta_cols):
                print(f"{delta} = {coef:.3f} ({np.abs(coef)/max_coef:.3f})")

        # Visualize AUC
        self.visualizer.classification_auc(y, y_pred, groups, tag)

    @log
    def run_wrapper(self, run):
        """Wrapper for running modelling with log."""
        run()

    def run_age(self):
        """Run age modelling."""

        # Run age modelling
        print("Running age modelling...")

        # Set up directory
        self.command_setup("model_age")

        # Load data
        self.load_data(required=["features"])

        # Initialized dictionaries
        self.set_dict()

        # Set dataframes
        self.set_features_dataframes()

        # Use visualizer to show age distribution of controls per covariate (all systems share the age distribution)
        cn_ages = {covar: self.dfs["cn"][covar][self.systems[0]]["age"].to_list() for covar in self.covars}
        self.age_distribution(cn_ages, name="Controls")

        # Show features vs age for controls for each system
        for system in self.systems:
            cn_features = {covar: self.dfs["cn"][covar][system] for covar in self.covars}
            self.features_vs_age(cn_features, tag=NameTag(system=system))

        # Model age for each system on controls
        self.model_all()

        # Predict to all other subject types
        self.predict_all()

        # Save prediction
        self.save_predictions()
    
    def run_model_feature_influence(self):
        """Run feature influence on age model and classification."""

       # Run model feature influence
        print("Running model feature influence...")

        # Run setup
        self.command_setup("model_feature_influence")

        # Load data
        self.load_data(required=["features", "clinical"])

        # Initialize dictionaries
        self.set_dict()

        # Check that arguments given for each group and that they exist
        if self.args.group1 is None or self.args.group2 is None:
            raise ValueError("Must provide two groups to classify.")
        elif self.args.group1 not in self.df_clinical.columns or self.args.group2 not in self.df_clinical.columns:
            raise ValueError("Classes must be one of the following: %s" % self.df_clinical.columns.to_list())

        # Set dataframes
        self.set_features_dataframes()

        # Show ordering of features with importance to age vs discrimination of clinical groups
        for system in self.systems:
            for covar in self.covars:
                tag = NameTag(covar=covar, system=system)
                self.feature_ordering(tag)

    def run_age_model_vs_logistic_regression(self):
        """Run age model vs logistic regression for classification."""

        print("Running age model vs logistic regression...")

        # Set up
        self.command_setup("age_model_vs_logistic_regression")

        # Load data
        self.load_data(required=["features", "clinical"])

        # Initialize dictionaries
        self.set_dict()

        # Check that arguments given for each group and that they exist
        if self.args.group1 is None or self.args.group2 is None:
            raise ValueError("Must provide two groups to classify.")
        elif self.args.group1 not in self.df_clinical.columns or self.args.group2 not in self.df_clinical.columns:
            raise ValueError("Classes must be one of the following: %s" % self.df_clinical.columns.to_list())
        
        # Set dataframes
        self.set_features_dataframes()

        # Show ordering of features with importance to age vs discrimination of clinical groups
        for system in self.systems:
            for covar in self.covars:
                tag = NameTag(covar=covar, system=system)
                order_age, order_discrimination = self.feature_ordering(tag)
                self.classification_feature_analysis(order_age, tag)
                self.classifcication_analysis(tag)


    def run_factor_correlation(self):
        """Run factor correlation analysis between deltas and factors."""

        print("Running factors correlation analysis...")

        # Set up
        self.command_setup("factor_correlation")

        # Load data
        self.load_data(required=["ages", "factors"])

        # Calculate covariate correction for factors
        if self.flags["covariates"]:
            beta = {}
            if self.flags["covcorr_mode"] == "cn":
                print("Applying covariate correction for factors using CN group as reference.")
                group_index = self.df_clinical["cn"]
                factors = self.df_factors.loc[group_index].to_numpy()
                covars = self.df_covariates.loc[group_index].to_numpy()
                _, beta["cn"] = covariate_correction(factors, covars)
            elif self.flags["covcorr_mode"] == "all":
                print("Applying covariate correction for factors using whole population as reference.")
                # We take all subjects for covariate correction
                factors = self.df_factors.to_numpy()
                covars = self.df_covariates.to_numpy()
                _, beta["all"] = covariate_correction(factors, covars)
            elif self.flags["covcorr_mode"] == "each":
                print("Applying covariate correction for factors using each group as reference.")
                for group in self.subject_types:
                    group_index = self.df_clinical[group]
                    factors = self.df_factors.loc[group_index].to_numpy()
                    covars = self.df_covariates.loc[group_index].to_numpy()
                    _, beta[group] = covariate_correction(factors, covars)

        # For each subject type and system run correlation analysis
        for subject_type in self.subject_types:
            tag = NameTag(group=subject_type)
            dfs_systems = {}
            df_sub = self.df_ages.loc[self.df_clinical[subject_type]]
            df_factors = self.df_factors.loc[df_sub.index]
            for system in self.systems:
                df_sys = df_sub[[col for col in df_sub.columns if system in col]]
                dfs_systems[system] = df_sys
            if self.flags["covariates"]:
                covars = self.df_covariates.loc[df_sub.index].to_numpy()
                if self.flags["covcorr_mode"] in ["cn", "all"]:
                    self.factors_vs_deltas(dfs_systems, df_factors, tag, covars, beta[self.flags["covcorr_mode"]])
                elif self.flags["covcorr_mode"] == "each":
                    self.factors_vs_deltas(dfs_systems, df_factors, tag, covars, beta[subject_type])
            else:
                self.factors_vs_deltas(dfs_systems, df_factors, tag)

    def run_clinical(self):
        """Analyse differences between deltas in clinical groups."""

        print("Running clinical outcomes...")

        # Set up
        self.command_setup("clinical_groups")

        # Load data
        self.load_data(required=["ages", "clinical"])

        # Obtain dataframes for each group
        dfs = {g: self.df_ages.loc[self.df_clinical[g]] for g in self.subject_types}

        # Use visualizer to show age distribution per clinical group
        ages = {g: dfs[g].iloc[:, 0].to_list() for g in self.subject_types}
        self.age_distribution(ages, name="Clinical Groups")

        # Show differences in groups per system
        for system in self.systems:
            dfs_systems = {g: dfs[g][[col for col in dfs[g].columns if system in col]] for g in self.subject_types}
            self.deltas_by_group(dfs_systems, tag=NameTag(system=system))

    def run_classification(self):
        """Run classification between two different clinical groups."""

        print("Running classification...")

        # Set up
        self.command_setup("clinical_classify")

        # Load data
        self.load_data(required=["ages", "clinical"])

        # Check that arguments given for each group and that they exist
        if self.args.group1 is None or self.args.group2 is None:
            raise ValueError("Must provide two groups to classify.")
        elif self.args.group1 not in self.df_clinical.columns or self.args.group2 not in self.df_clinical.columns:
            raise ValueError("Classes must be one of the following: %s" % self.df_clinical.columns.to_list())
        else:
            df_group1 = self.df_ages[self.df_clinical[self.args.group1]]
            df_group2 = self.df_ages[self.df_clinical[self.args.group2]]

        # Create a classifier for each system
        for system in self.systems:
            df_group1_system = df_group1[[col for col in df_group1.columns if system in col]]
            df_group2_system = df_group2[[col for col in df_group2.columns if system in col]]
            self.classify(df_group1_system, df_group2_system, [self.args.group1, self.args.group2], tag=NameTag(system=system))

        # Create a classifier for all systems
        if len(self.systems) > 1:
            self.classify(df_group1, df_group2, [self.args.group1, self.args.group2], tag=NameTag(system="all"))


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

    factor_correlation_command(self): Runs factor correlation analysis.

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
            if attr_name != "output":
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
            elif command == "factor_correlation":
                error = self.factor_correlation_command()
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

    def classifier_command(self):
        """Set classifier parameters."""

        # Split into items
        self.line = self.line.split()
        error = None

        # Check that at least one argument input
        if len(self.line) == 0:
            error = "Must provide two arguments or None."
            return error

        # Set defaults
        if len(self.line) == 1 and self.line[0] == "None":
            self.args.classifier_thr = 0.5
            self.args.classifier_ci = 0.95
            return error

        # Check wether items are floats
        for item in self.line:
            try:
                float(item)
            except ValueError:
                error = "Parameters must be floats."
                return error

        # Set parameters
        if len(self.line) == 2:
            self.args.classifier_thr = float(self.line[0])
            self.args.classifier_ci = float(self.line[1])
        elif len(self.line) > 2:
            error = "Too many values to unpack."
        elif len(self.line) == 1:
            error = "Must provide two arguments or None."

        return error

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

        # Ask for optional
        print("Input covariates file path (Optional):")
        self.force_command(self.load_command, "--covariates")

        # Ask for CV parameters adn classifier parameters
        print("CV parameters (Default: nº splits=5 and seed=0):")
        self.force_command(self.cv_command, "classifier")
        print("Classifier parameters (Default: thr=0.5 and ci=0.95):")
        self.force_command(self.classifier_command)

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

        # Ask for optional
        print("Input covariates file path (Optional):")
        self.force_command(self.load_command, "--covariates")

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

        # Split into items
        self.line = self.line.split()
        error = None

        # Check that one argument given
        if len(self.line) != 1:
            error = "Must provide one covariate name."
            return error

        # Set covariate name
        if self.line[0] == "None":
            self.args.covar_name = None
        else:
            self.args.covar_name = self.line[0]

        return error

    def cv_command(self):
        """Load CV parameters."""

        # Split into items
        self.line = self.line.split()
        error = None

        # Check that at least one argument input
        if len(self.line) == 0:
            error = "Must provide at least one argument."
            return error

        # Check that first argument is model or classifier
        if self.line[0] not in ["model", "classifier"]:
            error = "Must provide either model or classifier flag."
            return error
        elif self.line[0] == "model":
            arg_type = "model"
        elif self.line[0] == "classifier":
            arg_type = "classifier"

        # Set default values
        if len(self.line) == 2 and self.line[1] == "None":
            setattr(self.args, arg_type + "_cv_split", 5)
            setattr(self.args, arg_type + "_seed", 0)
            return error

        # Check wether items are integers
        for item in self.line[1:]:
            if not item.isdigit():
                error = "CV parameters must be integers"
                return error

        # Set CV parameters
        if len(self.line) == 2:
            setattr(self.args, arg_type + "_cv_split", int(self.line[1]))
            setattr(self.args, arg_type + "_seed", 0)
        elif len(self.line) == 3:
            setattr(self.args, arg_type + "_cv_split", int(self.line[1]))
            setattr(self.args, arg_type + "_seed", int(self.line[2]))
        else:
            error = "Too many values to unpack."

        return error

    def factor_correlation_command(self):
        """Run factor correlation analysis."""

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

        # Run factor correlation analysis capture any error raised and print
        try:
            self.run_wrapper(self.run_factor_correlation)
            print("Finished factor correlation analysis.")
        except Exception as e:
            print(e)
            error = "Error running factor correlation analysis."

        return error

    def group_command(self):
        """Load groups."""

        # Split into items
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
        print(messages.factor_correlation_command_message)
        print(messages.model_age_command_message)
        print(messages.quit_command_message)
        print(messages.read_the_documentation_message)

    def load_command(self):
        """Load file paths."""

        # Split into items
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
        print("Input covariate type to train separate models (Optional):")
        self.force_command(self.covar_command)
        print("Input clinical file path (Optional):")
        self.force_command(self.load_command, "--clinical")
        print("Input systems file path (Optional):")
        self.force_command(self.load_command, "--systems")

        # Ask for scaler, model, CV parameters, feature extension, and hyperparameter tuning
        print("Scaler type and parameters (Default:standard)")
        print(f"Available: {list(AgeML.scaler_dict.keys())}")
        print("Example: standard with_mean=True with_std=False")
        self.force_command(self.scaler_command)
        print("Model type and parameters (Default:linear_reg)")
        print(f"Available: {list(AgeML.model_dict.keys())}")
        print("Example: linear_reg fit_intercept=True normalize=False")
        self.force_command(self.model_command)
        print("CV parameters (Default: nº splits=5 and seed=0):")
        print("Example: 10 0")
        self.force_command(self.cv_command, "model")
        print("Polynomial feature extension degree. Leave blank if not desired (Default: 0, max. 3)")
        print("Example: 3")
        self.force_command(self.feature_extension_command)
        print("Hyperparameter tuning. Number of points in grid search: (Default: 2)")
        print("Example: 100")
        self.force_command(self.hyperparameter_grid_command)
        print("Hyperparameters to tune. Leave blank if not desired (Default: None)")
        print("Example (in the case of a 'ridge' model): alpha=-3,3")
        self.force_command(self.hyperparameter_parameters_command)
        # Run modelling capture any error raised and print
        try:
            self.run_wrapper(self.run_age)
            print("Finished running age modelling.")
        except Exception as e:
            print(e)
            error = "Error running age modelling."

        return error

    def model_command(self):
        """Load model parameters."""

        # Split into items
        self.line = self.line.split()
        valid_types = list(AgeML.model_dict.keys())
        error = None

        # Check that at least one argument input
        if len(self.line) == 0:
            error = "Must provide at least one argument or None."
            return error
        else:
            model_type = self.line[0]

        # Set model type or default
        if model_type == "None":
            self.args.model_type = "linear_reg"
        else:
            if model_type not in valid_types:
                error = f"Choose a valid model type: {valid_types}"
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

        # Try to set an instance of the specified scaler with the provided arguments
        try:
            AgeML.model_dict[self.args.model_type](**self.args.model_params)
        except TypeError:  # Raised when invalid parameters are given to sklearn
            error = f"Model parameters are not valid for {self.args.model_type} model. Check them in the sklearn documentation."

        return error

    def output_command(self):
        """Load output directory."""

        # Split into items
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

        # Split into items
        self.line = self.line.split()
        error = None
        valid_types = list(AgeML.scaler_dict.keys())

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
                error = f"Choose a valid scaler type: {valid_types}"
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

        # Try to set an instance of the specified scaler with the provided arguments
        try:
            AgeML.scaler_dict[self.args.scaler_type](**self.args.scaler_params)
        except TypeError:
            error = f"Scaler parameters are not valid for {self.args.scaler_type} scaler. Check them in the sklearn documentation."
            return error

        return error

    def feature_extension_command(self):
        """Load feature extension."""

        # Split into items and remove  command
        self.line = self.line.split()
        error = None

        # Check that at least one argument input
        if len(self.line) > 1:
            error = "Must provide only one integer, or none."
            return error

        # Set default values
        if len(self.line) == 0 or self.line[0] == "None":
            self.args.feature_extension = 0
            return error

        # Check whether items are integers
        if not self.line[0].isdigit():
            error = "The polynomial feature extension degree must be an integer (0, 1, 2, or 3)"
            return error

        # Set CV parameters
        self.args.feature_extension = int(self.line[0])
        return error

    def hyperparameter_grid_command(self):
        """Load hyperparameter search grid."""

        # Split into items and remove command
        self.line = self.line.split()
        error = None

        # Check that at least one argument input
        if len(self.line) > 1:
            error = "Must provide only one integer, or none."
            return error

        # Set default values
        if len(self.line) == 0 or self.line[0] == "None":
            self.args.hyperparameter_tuning = 0
            return error

        # Check whether items are integers
        if not self.line[0].isdigit():
            error = "The number of points in the hyperparameter grid must be a positive, nonzero integer."
            return error

        # Set CV parameters
        self.args.hyperparameter_tuning = int(self.line[0])
        return error

    def hyperparameter_parameters_command(self):
        """Specify hyperparameters to tune."""

        hyperparameter_params = {}
        # Split into items and remove command
        self.line = self.line.split(" ")
        error = None

        if self.line == ["", "None"]:
            self.args.hyperparameter_params = {}
        else:
            for item in self.line[1:]:
                if item.count("=") != 1:
                    error = (
                        "Hyperparameter tuning parameters must be in the format "
                        "param1=value1_low,value1_high param2=kernel_A,kernel_B,kernel_C..."
                    )
                    return error
                key, values = item.split("=")
                values = [convert(value) for value in values.split(",")]
                vals_are_str = all([isinstance(value, str) for value in values])
                vals_are_num = all([isinstance(value, (int, float)) for value in values])
                # If not 2 values provided in numerical hyperparams, raise error
                if vals_are_num and len(values) != 2:
                    err_msg = "Numerical hyperparameter values must be exactly two numbers (e.g.: param1=2,3)."
                    raise ValueError(err_msg)
                # If no value provided in categorical hyperparams, raise error
                elif vals_are_str and len(values) < 1:
                    err_msg = "Categorical hyperparameter values must be at least one string (e.g.: param1=kernel_A)."
                    raise ValueError(err_msg)
                hyperparameter_params[key] = values
            # Add attribute to args
            self.args.hyperparameter_params = hyperparameter_params
