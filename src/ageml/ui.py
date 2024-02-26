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
import copy

from datetime import datetime
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats

import ageml.messages as messages
from ageml.visualizer import Visualizer
from ageml.utils import create_directory, feature_extractor, significant_markers, convert, log
from ageml.modelling import AgeML, Classifier
from ageml.processing import find_correlations, covariate_correction


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

    generate_model(self): Set model with parameters.

    set_classifier(self): Set classifier with parameters.

    check_file(self, file): Check that file exists.

    load_csv(self, file): Use panda to load csv into dataframe.

    load_data(self, required): Load data from csv files.

    age_distribution(self, dfs, labels=None): Use visualizer to show age distribution.

    features_vs_age(self, df, significance=0.05): Use visualizer to explore relationship between features and age.

    model_age(self, df, model): Use AgeML to fit age model with data.

    predict_age(self, df, model, model_name): Use AgeML to predict age with data.

    factors_vs_deltas(self, dfs_ages, dfs_factors, groups, significance=0.05): Calculate correlations between factors and deltas.

    deltas_by_group(self, df, labels): Calculate summary metrics of deltas by group.

    classify(self, df1, df2, groups): Classify two groups based on deltas.

    run_wrapper(self, run): Wrapper for running modelling with log.

    run_age(self): Run age modelling.

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

        self.flags = {"clinical": False, "covariates": False, "covarname": False, "systems": False}

    def set_visualizer(self):
        """Set visualizer with output directory."""

        self.visualizer = Visualizer(self.dir_path)

    def generate_model(self):
        """Set model with parameters."""

        model = AgeML(
            self.args.scaler_type,
            self.args.scaler_params,
            self.args.model_type,
            self.args.model_params,
            self.args.model_cv_split,
            self.args.model_seed,
            self.args.hyperparameter_tuning,
            self.args.feature_extension
        )
        return model

    def generate_classifier(self):
        """Set classifier with parameters."""

        classifier = Classifier(
            self.args.classifier_cv_split,
            self.args.classifier_seed,
            self.args.classifier_thr,
            self.args.classifier_ci)
        
        return classifier

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

        # Load FEATURES
        self.df_features = self.load_csv('features')
        if self.df_features is not None:
            if "age" not in self.df_features.columns:
                raise KeyError("Features file must contain a column name 'age', or any other case-insensitive variation.")
        elif "features" in required:
            raise ValueError("Features file must be provided.")

        # Load covariates
        self.df_covariates = self.load_csv('covariates')
        if self.df_covariates is not None:
            self.flags['covariates'] = True
        elif "covariates" in required:
            raise ValueError("Covariates file must be provided.")

        # Check that covar name is given
        if self.df_covariates is not None and hasattr(self.args, 'covar_name') and self.args.covar_name is not None:
            # Force covar_name to be lower case
            self.args.covar_name = self.args.covar_name.lower()
            # Check that covariate column exists
            if self.args.covar_name not in self.df_covariates.columns:
                raise KeyError("Covariate column %s not found in covariates file." % self.args.covar_name)
            self.flags['covarname'] = True

        # Load factors
        self.df_factors = self.load_csv('factors')
        if self.df_factors is None and "factors" in required:
            raise ValueError("Factors file must be provided.")

        # Load SYSTEMS file. txt expected. Format: system_name:feature1,feature2,...
        # Check that the file exists. If not, raise error. If yes, load line by line into a dict
        if hasattr(self.args, "systems") and self.args.systems is not None:
            self.dict_systems = {}
            if not self.check_file(self.args.systems):
                ValueError("Systems file '%s' not found." % self.args.systems)
            else:
                # Parse the systems file line by line
                self.flags['systems'] = True
                for line in open(self.args.systems, 'r'):
                    line = line.split("\n")[0]  # Remove newline character
                    line = line.split(':')  # Split by the separator
                    # Check that the line has 2 elements
                    if len(line) != 2:
                        raise ValueError("Systems file must be in the format 'system_name_1:feature1,feature2,...'")
                    # Check that the feature names are in the features file. If not, raise a ValueError
                    lowercase_features = [f.lower() for f in self.df_features.columns.to_list()]
                    systems_features = [f.lower() for f in line[1].split(',')]
                    for f in systems_features:
                        if f not in lowercase_features:
                            raise ValueError("Feature '%s' not found in features file." % f)
                    # Save the system name and its features
                    self.dict_systems[line[0]] = systems_features
                # Check that the dictionary has at least one entry
                if len(self.dict_systems) == 0:
                    raise ValueError("Systems file is probably incorrectly formatted. Check it please.")
        elif "systems" in required:
            raise ValueError("Systems file must be provided.")

        # Load CLINICAL file
        self.df_clinical = self.load_csv('clinical')
        if self.df_clinical is not None:
            # Check that CN in columns
            if "cn" not in self.df_clinical.columns:
                raise KeyError("Clinical file must contain a column name 'CN' or any other case-insensitive variation.")
            # Check datatypes of columns are all boolean
            elif [self.df_clinical[col].dtype == bool for col in self.df_clinical.columns].count(False) != 0:
                raise TypeError("Clinical columns must be boolean type. Check that all values are encoded as 'True' or 'False'.")
            else:
                self.flags['clinical'] = True
                self.cn_subjects = self.df_clinical[self.df_clinical["cn"]].index
        elif "clinical" in required:
            raise ValueError("Clinical file must be provided.")

        # Load AGES file
        # Check if already has ages loaded
        if hasattr(self, "df_ages"):
            if self.df_ages is None:
                self.df_ages = self.load_csv('ages')
            else:
                # Dont over write if None
                df = self.load_csv('ages')
                if df is not None:
                    self.df_ages = df
                    warning_message = (
                        "Ages file already loaded, overwriting with  %s provided file."
                        % self.args.ages
                    )
                    print(warning_message)
                    warnings.warn(warning_message, category=UserWarning)
        else:
            self.df_ages = self.load_csv('ages')
        # Check that ages file has required columns
        if self.df_ages is not None:
            cols = ["age", "predicted_age", "corrected_age", "delta"]
            for col in self.df_ages.columns.to_list():
                if not any(col.startswith(c) for c in cols):
                    raise KeyError("Ages file must contain the following columns %s, or derived names." % cols)
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
                print("Number of subjects in dataframe %s: %d" % (label, df.shape[0]))
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
        flag_subjects = True
        for df in dfs:
            if df is not None:
                df.drop(self.subjects_missing_data, inplace=True, errors="ignore")
                # Print only once number of final subjects
                if flag_subjects:
                    print("Number of subjects without any missing data: %d" % len(df))
                    flag_subjects = False

        # TODO only if clinical files found 
        print('Controls found in clinical file, selecting controls from clinical file.')
        print('Number of CN subjects found: %d' % self.cn_subjects.__len__())

    def age_distribution(self, ages_dict: dict, name=""):
        """Use visualizer to show age distribution.

        Parameters
        ----------
        ages_dict: dictionary containing ages for each group
        name: name to give to visualizer to save file"""

        # Select age information
        print("-----------------------------------")
        print("Age distribution %s" % name)
        for key, vals in ages_dict.items():
            print(key)
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
                        warn_message = "Age distributions %s and %s are not similar: %.2f (%.2g) " % (
                            labels[i], labels[j], t_stat, p_val)
                        print(warn_message)
                        warnings.warn(warn_message, category=UserWarning)

        # Use visualiser
        self.visualizer.age_distribution(ages, labels, name)

    def features_vs_age(self, features_dict: dict, significance: float = 0.05, name: str = ""):
        """Use visualizer to explore relationship between features and age.

        Parameters
        ----------
        features_dict: dictionary containing features for each group
        significance: significance level for correlation"""

        # Select data to visualize
        print("-----------------------------------")
        print("Features by correlation with Age of Controls")
        print("significance: %.2g * -> FDR, ** -> bonferroni" % significance)

        # Make lists to store covariate info for each dataframe
        X_list, y_list, corr_list, order_list, significance_list = [], [], [], [], []
        for label, df in features_dict.items():
            print('Covariate %s' % label)
            # Extract features
            X, y, feature_names = feature_extractor(df)
            # Covariate correction
            if self.flags["covariates"] and not self.flags['covarname']:
                print("Covariate effects will be subtracted from features.")
                X, _ = covariate_correction(X, self.df_covariates.loc[df.index].to_numpy())
            # Calculate correlation between features and age
            corr, order, p_values = find_correlations(X, y)
            # Reject null hypothesis of no correlation
            reject_bon, _, _, _ = multipletests(p_values, alpha=significance, method='bonferroni')
            reject_fdr, _, _, _ = multipletests(p_values, alpha=significance, method='fdr_bh')
            significant = significant_markers(reject_bon, reject_fdr)
            # Print results
            for idx, order_element in enumerate(order):
                print("%d.%s %s %s: %.2f (%.2g)" % (idx + 1, label, significant[order_element],
                                                    feature_names[order_element], corr[order_element],
                                                    p_values[order_element]))
            # Append all the values
            X_list.append(X), y_list.append(y), corr_list.append(corr), order_list.append(order), significance_list.append(significant)

        # Use visualizer to show results
        self.visualizer.features_vs_age(X_list, y_list, corr_list, order_list,
                                        significance_list, feature_names, list(features_dict.keys()), name)

    def model_age(self, df, model, name: str = ""):
        """Use AgeML to fit age model with data.

        Parameters
        ----------
        df: dataframe with features and age; shape=(n,m+1)
        model: AgeML object
        name: name of the model"""

        # Show training pipeline
        print("-----------------------------------")
        if name == "":
            print(f"Training Age Model for all controls ({self.args.model_type})")

        else:
            print(f"Training Age Model ({self.args.model_type}): {name}")
        print(model.pipeline)

        # Select data to model
        X, y, _ = feature_extractor(df)

        # Covariate correction
        if self.flags["covariates"] and not self.flags['covarname']:
            print("Covariate effects will be subtracted from features.")
            X, beta = covariate_correction(X, self.df_covariates.loc[df.index].to_numpy())
        else:
            beta = None

        # Fit model and plot results
        y_pred, y_corrected = model.fit_age(X, y)
        self.visualizer.true_vs_pred_age(y, y_pred, name)
        self.visualizer.age_bias_correction(y, y_pred, y_corrected, name)

        # Calculate deltas
        deltas = y_corrected - y

        # Save to dataframe and csv
        data = np.stack((y, y_pred, y_corrected, deltas), axis=1)
        cols = ["age", "predicted_age", "corrected_age", "delta"]
        df_ages = pd.DataFrame(data, index=df.index, columns=cols)

        return model, df_ages, beta

    def predict_age(self, df, model, beta: np.ndarray = None, model_name: str = None):
        """Use AgeML to predict age with data."""

        # Show prediction pipeline
        print("-----------------------------------")
        print(f"Predicting with Age Model ({self.args.model_type}): {model_name}")
        print(model.pipeline)

        # Select data to model
        X, y, _ = feature_extractor(df)

        # Covariate correction
        if self.flags["covariates"] and not self.flags['covarname']:
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

    def factors_vs_deltas(self, dict_ages, df_factors,  group="", significance=0.05):
        """Calculate correlations between factors and deltas.

        Parameters
        ----------
        dict_ages: dictionary for each system with deltas; shape=(n,m)
        df_factors: dataframe with factor information; shape=(n,m)
        significance: significance level for correlation"""

        # Select age information
        print("-----------------------------------")
        print("Correlations between lifestyle factors for %s" % group)
        print("significance: %.2g * -> FDR, ** -> bonferroni" % significance)

        # Iterate over systems
        corrs, significants = [], []

        # Facotr information
        factors = df_factors.to_numpy()
        factor_names = df_factors.columns.to_list()

        for system, df in dict_ages.items():
            print(system)

            # Select data to visualize
            deltas = df['delta_%s' % system].to_numpy()

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
                print("%d. %s %s: %.2f (%.2g)" % (i + 1, significant[o], factor_names[o], corr[o], p_values[o]))

        # Use visualizer to show bar graph
        self.visualizer.factors_vs_deltas(corrs, list(dict_ages.keys()), factor_names, significants, group)

    def deltas_by_group(self, dfs, system: str = None, significance: float = 0.05):
        """Calculate summary metrics of deltas by group.
        
        Parameters
        ----------
        df: list of dataframes with delta information; shape=(n,m)
        labels: list of labels for each dataframe; shape=(n,)
        system: name of the system from which the variables come from"""

        # Select age information
        print("-----------------------------------")
        print("Delta distribution by group %s" % system)

        # Obtain deltas means and stds
        deltas = []
        for group, df in dfs.items():
            vals = df["delta_%s" % system].to_numpy()
            deltas.append(vals)
            print(group)
            print("Mean delta: %.2f" % np.mean(vals))
            print("Std delta: %.2f" % np.std(vals))
            print("Delta range: [%d, %d]" % (np.min(vals), np.max(vals)))

        # Obtain statistically significant difference between deltas
        print("Checking for statistically significant differences between deltas...")
        print("significance: %.2g * -> FDR, ** -> bonferroni" % significance)

        # Calculate p-values
        p_vals_matrix = np.zeros((len(deltas), len(deltas)))
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                _, p_val = stats.ttest_ind(deltas[i], deltas[j])
                p_vals_matrix[i, j] = p_val
        
        # Reject null hypothesis of no correlation
        reject_bon, _, _, _ = multipletests(p_vals_matrix.flatten(), alpha=significance, method='bonferroni')
        reject_fdr, _, _, _ = multipletests(p_vals_matrix.flatten(), alpha=significance, method='fdr_bh')
        reject_bon = reject_bon.reshape((len(deltas), len(deltas)))
        reject_fdr = reject_fdr.reshape((len(deltas), len(deltas)))

        # Print results
        labels = list(dfs.keys())
        for i in range(len(deltas)):
            significant = significant_markers(reject_bon[i], reject_fdr[i])
            for j in range(i + 1, len(deltas)):
                pval_message = "p-value between %s and %s: %.2g" % (
                    labels[i], labels[j], p_vals_matrix[i, j])
                if significant[j] != "":
                    pval_message = significant[j] + " " + pval_message
                print(pval_message)

        # Use visualizer
        self.visualizer.deltas_by_groups(deltas, labels, system)

    def classify(self, df1, df2, groups, system: str = None, beta: np.ndarray = None):
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
        print(f"Classification between groups {groups[0]} and {groups[1]} (system: {system})")

        # Select delta information
        delta_cols = [col for col in df1.columns if "delta" in col]
        deltas1 = df1[delta_cols].to_numpy()
        deltas2 = df2[delta_cols].to_numpy()

        # Create X and y for classification
        if len(delta_cols) == 1:
            X = np.concatenate((deltas1, deltas2)).reshape(-1, 1)
            y = np.concatenate((np.zeros(deltas1.shape), np.ones(deltas2.shape)))
        else:
            X = np.concatenate((deltas1, deltas2))
            y = np.concatenate((np.zeros(deltas1.shape[0]), np.ones(deltas2.shape[0])))

        # Generate classifier
        self.classifier = self.generate_classifier()

        # Apply covariate correction
        if self.flags["covariates"]:
            Z = np.concatenate((self.df_covariates.loc[df1.index].to_numpy(), self.df_covariates.loc[df2.index].to_numpy()))
            X, _ = covariate_correction(X, Z, beta)

        # Calculate classification
        y_pred = self.classifier.fit_model(X, y)
        if len(delta_cols) > 1:
            print("Logistic regressor weigths:")
            for coef, delta in zip(self.classifier.model.coef_[0], delta_cols):
                print(f"    {delta} = {coef:.3f}")

        # Visualize AUC
        self.visualizer.classification_auc(y, y_pred, groups, system)

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

        # Set initial parameters for model to defaults
        naming = ""
        subject_types = ['cn']
        covars = ['all']
        systems = ['all']

        # Load data
        self.load_data(required=["features"])

        # Check possible flags of interest
        if self.flags['clinical']:
            subject_types = self.df_clinical.columns.to_list()
        if self.flags['covarname']:
            covars = pd.unique(self.df_covariates[self.args.covar_name]).tolist()
            naming += f"_{self.args.covar_name}"
        if self.flags['systems']:
            systems = list(self.dict_systems.keys())
            naming += "_multisystem"

        # Initialized dictionaries
        dfs = {subject_type: {covar: {system: {} for system in systems} for covar in covars} for subject_type in subject_types}
        preds = {subject_type: {covar: {system: {} for system in systems} for covar in covars} for subject_type in subject_types}
        models = {covar: {system: {} for system in systems} for covar in covars}
        betas = {covar: {system: {} for system in systems} for covar in covars}

        # Obtain dataframes for each subject type, covariate and system
        for subject_type in subject_types:
            # Keep only the subjects of the specified type
            df_sub = self.df_features[self.df_clinical[subject_type]]
            for covar in covars: 
                # Keep subjects with the specified covariate
                if self.flags['covarname']:
                    covar_index = set(self.df_covariates[self.df_covariates[self.args.covar_name] == covar].index)
                    df_cov = df_sub[df_sub.index.isin(covar_index)]
                else:
                    df_cov = df_sub
                for system in systems:
                    # Keep only the features of the system
                    if self.flags['systems']:
                        df_sys = df_cov[['age']+self.dict_systems[system]]
                    else:
                        df_sys = df_cov
                    # Save the dataframe
                    dfs[subject_type][covar][system] = df_sys


        # Use visualizer to show age distribution of controls per covariate (all systems share the age distribution)
        cn_ages = {covar: dfs['cn'][covar][systems[0]]['age'].to_list() for covar in covars}
        self.age_distribution(cn_ages, name="controls"+naming)

        # Show features vs age for controls for each system
        for system in systems:
            cn_features = {covar: dfs['cn'][covar][system] for covar in covars}
            self.features_vs_age(cn_features, name="controls"+naming+"_"+system)

        # Model age for each system on controls
        for covar in covars:
            for system in systems:
                model_name = f"{covar}_{system}"
                ageml_model = self.generate_model()
                models[covar][system], df_pred, betas[covar][system] = self.model_age(dfs['cn'][covar][system],
                                                                                      ageml_model, name=model_name)
                df_pred = df_pred.drop(columns=['age'])
                df_pred.rename(columns=lambda x: f"{x}_{system}", inplace=True)
                preds['cn'][covar][system] = df_pred

        # Apply to all other subject types
        for subject_type in subject_types:
            # Do not apply to controls
            if subject_type == 'cn':
                continue
            for covar in covars:
                for system in systems:
                    model_name = f"{covar}_{system}"
                    df_pred = self.predict_age(dfs[subject_type][covar][system], models[covar][system],
                                               betas[covar][system], model_name=model_name)
                    df_pred = df_pred.drop(columns=['age'])
                    df_pred.rename(columns=lambda x: f"{x}_{system}", inplace=True)
                    preds[subject_type][covar][system] = df_pred

        # Concatenate predictions into a DataFrame
        stack = []
        for subject_type in subject_types:
            for covar in covars:
                df_systems = pd.concat([preds[subject_type][covar][system] for system in systems], axis=1)
                stack.append(df_systems)
        df_ages = pd.concat(stack, axis=0)

        # Drop duplicates keep first (some subjects may be in more than one subejct type)
        df_ages = df_ages[~df_ages.index.duplicated(keep='first')]

        # Add age information
        df_ages = pd.concat([self.df_features['age'], df_ages], axis=1)

        # Save dataframe to csv
        filename = "predicted_age" + naming + ".csv"
        df_ages.to_csv(os.path.join(self.dir_path, filename))

    def run_factor_correlation(self):
        """Run factor correlation analysis between deltas and factors."""

        print("Running factors correlation analysis...")

        # Reset flags
        self.set_flags()

        # Initial parameters
        subject_types = ['cn']

        # Load data
        self.load_data(required=["ages", "factors"])

        # Check possible flags of interest
        if self.flags['clinical']:
            subject_types = self.df_clinical.columns.to_list()

        # Obtain systems
        systems = [col[6:] for col in self.df_ages.columns if "delta" in col]

        # For each subject type
        for subject_type in subject_types:
            dfs_systems = {}
            df_sub = self.df_ages.loc[self.df_clinical[subject_type]]
            df_factors = self.df_factors.loc[df_sub.index]
            for system in systems:
                df_sys = df_sub[[col for col in df_sub.columns if system in col]]
                dfs_systems[system] = df_sys
            self.factors_vs_deltas(dfs_systems, df_factors, subject_type)

    def run_clinical(self):
        """Analyse differences between deltas in clinical groups."""

        print("Running clinical outcomes...")

        # Reset flags
        self.set_flags()

        # Load data
        self.load_data(required=["ages", "clinical"])

        # Obtain dataframes for each group
        groups = self.df_clinical.columns.to_list()
        dfs = {g: self.df_ages.loc[self.df_clinical[g]] for g in groups}
    
        # Use visualizer to show age distribution per clinical group
        ages = {g: dfs[g].iloc[:, 0].to_list() for g in groups}
        self.age_distribution(ages, name="clinical_groups")

        # Show differences in groups per system
        systems = [col[6:] for col in self.df_ages.columns if "delta" in col]
        for system in systems:
            dfs_systems = {g: dfs[g][[col for col in dfs[g].columns if system in col]] for g in groups}
            self.deltas_by_group(dfs_systems, system=system)

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
        # Balance the groups (subsampling)
        # if df_group1.shape[0] < df_group2.shape[0]:
        #     print("### Balancing groups... ###")
        #     df_group2 = df_group2.sample(n=df_group1.shape[0], random_state=42)
        # elif df_group1.shape[0] > df_group2.shape[0]:
        #     print("### Balancing groups... ###")
        #     df_group1 = df_group1.sample(n=df_group2.shape[0], random_state=42)

        if self.flags["covariates"]:
            df_cn = self.df_ages.loc[self.df_clinical["cn"]]

        # Check if systems are provided in the ages file
        if any(["system" in col for col in self.df_ages.columns]):
            self.flags["systems"] = True
            # Make systems list
            systems_list = list({col.split("_")[-1] for col in self.df_ages.columns if "system" in col})

        # Classify between groups
        if self.flags["systems"]:
            for system in systems_list + ["delta"]:
                cols = [col for col in self.df_ages.columns.to_list() if system in col]
                df_group1_system = df_group1[cols]
                df_group2_system = df_group2[cols]
                if self.flags['covariates']:
                    print("Covariate effects will be subtracted from deltas.")
                    df_cn_system = df_cn[cols]
                    deltas = df_cn_system['delta'].to_numpy()
                    covars = self.df_covariates.loc[df_cn_system.index].to_numpy()
                    _, beta = covariate_correction(deltas, covars)
                else:
                    beta = None
                if system == "delta":
                    system = "all"
                self.classify(df_group1_system, df_group2_system, groups, system=system, beta=beta)
        else:
            if self.flags['covariates']:
                print("Covariate effects will be subtracted from deltas.")
                deltas = df_cn['delta'].to_numpy()
                covars = self.df_covariates.loc[df_cn.index].to_numpy()
                _, beta = covariate_correction(deltas, covars)
            else:
                beta = None
            self.classify(df_group1, df_group2, groups, beta=beta)


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
        if len(self.line) == 1 and self.line[0] == 'None':
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
        self.force_command(self.cv_command, 'classifier')
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
            pass
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
        if self.line[0] not in ['model', 'classifier']:
            error = "Must provide either model or classifier flag."
            return error
        elif self.line[0] == 'model':
            arg_type = 'model'
        elif self.line[0] == 'classifier':
            arg_type = 'classifier'

        # Set default values
        if len(self.line) == 2 and self.line[1] == 'None':
            setattr(self.args, arg_type + '_cv_split', 5)
            setattr(self.args, arg_type + '_seed', 0)
            return error

        # Check wether items are integers
        for item in self.line[1:]:
            if not item.isdigit():
                error = "CV parameters must be integers"
                return error

        # Set CV parameters
        if len(self.line) == 2:
            setattr(self.args, arg_type + '_cv_split', int(self.line[1]))
            setattr(self.args, arg_type + '_seed', 0)
        elif len(self.line) == 3:
            setattr(self.args, arg_type + '_cv_split', int(self.line[1]))
            setattr(self.args, arg_type + '_seed', int(self.line[2]))
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
        self.force_command(self.cv_command, 'model')
        print("Polynomial feature extension degree. Leave blank if not desired (Default: 0, max. 3)")
        print("Example: 3")
        self.force_command(self.feature_extension_command)
        print("Hyperparameter tuning. Number of points in grid search: (Default: 0)")
        print("Example: 100")
        self.force_command(self.hyperparameter_grid_command)

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
