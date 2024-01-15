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

        self.set_classifier()
        self.models = {}

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

    def generate_model(self):
        """Set model with parameters."""

        model = AgeML(
            self.args.scaler_type,
            self.args.scaler_params,
            self.args.model_type,
            self.args.model_params,
            self.args.cv_split,
            self.args.seed,
        )
        return model

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
        # Check that covar name is given
        if self.df_covariates is not None and hasattr(self.args, 'covar_name'):
            self.flags['covariates'] = True

        # Load factors
        self.df_factors = self.load_csv('factors')
        if self.df_factors is None and "factors" in required:
            raise ValueError("Factors file must be provided.")

        # Load SYSTEMS file. txt expected. Format: system_name:feature1,feature2,...
        # Check that the file exists. If not, raise error. If yes, load line by line into a dict
        if self.args.systems is not None:
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
        if self.args.systems is None and "systems" in required:
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
        elif "clinical" in required and self.args.clinical is None:
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
            age_col  = [col for col in df.columns if "age" in col][0]
            ages = df[age_col].to_numpy()
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
        model: AgeML object
        name: name of the model"""

        # Show training pipeline
        print("-----------------------------------")
        if name == "":
            print("Training Age Model")
        else:
            print("Training Model for covariate %s" % name)
        print(model.pipeline)

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
        print(model.pipeline)

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

    def factors_vs_deltas(self, dfs_ages, dfs_factors, groups, factor_names, significance=0.05, system: str = None):
        """Calculate correlations between factors and deltas.

        Parameters
        ----------
        dfs_ages: list of dataframes with delta information; shape=(n,m)
        dfs_factors: list of dataframes with factor information; shape=(n,m)
        groups: list of labels for each dataframe; shape=(n,),
        factor_names: list of factor names; shape=(m,)
        significance: significance level for correlation
        system: name of the system from which the variables come from"""

        # Select age information
        print("-----------------------------------")
        print("Correlations between lifestyle factors by group")
        print("significance: %.2g * -> FDR, ** -> bonferroni" % significance)

        # Iterate over groups
        corrs, significants = [], []
        for group, df_ages, df_factors in zip(groups, dfs_ages, dfs_factors):
            print(group)

            # Select data to visualize
            delta_col = [col for col in df_ages.columns if "delta" in col][0]
            deltas = df_ages[delta_col].to_numpy()
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
        self.visualizer.factors_vs_deltas(corrs, groups, factor_names, significants, system)

    def deltas_by_group(self, df, labels, system: str = None):
        """Calculate summary metrics of deltas by group.
        
        Parameters
        ----------
        df: list of dataframes with delta information; shape=(n,m)
        labels: list of labels for each dataframe; shape=(n,)
        system: name of the system from which the variables come from"""

        # Select age information
        print("-----------------------------------")
        print("Delta distribution by group")

        # Obtain deltas means and stds
        deltas = []
        for i, df_group in enumerate(df):
            delta_col = [col for col in df_group.columns if "delta" in col][0]
            deltas.append(df_group[delta_col].to_numpy())
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
        self.visualizer.deltas_by_groups(deltas, labels, system)

    def classify(self, df1, df2, groups, system: str = None):
        """Classify two groups based on deltas.

        Parameters
        ----------
        df1: dataframe with delta information; shape=(n,m)
        df2: dataframe with delta information; shape=(n,m)
        groups: list of labels for each dataframe; shape=(2,)
        system: name of the system from which the variables come from"""

        # Classification
        print("-----------------------------------")
        print("Classification between groups %s and %s" % (groups[0], groups[1]))

        # Select delta information
        delta_col = [col for col in df1.columns if "delta" in col][0]
        deltas1 = df1[delta_col].to_numpy()
        deltas2 = df2[delta_col].to_numpy()

        # Create X and y for classification
        X = np.concatenate((deltas1, deltas2)).reshape(-1, 1)
        y = np.concatenate((np.zeros(deltas1.shape), np.ones(deltas2.shape)))

        # Calculate classification
        y_pred = self.classifier.fit_model(X, y)

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
                df_clinical_cov = []
                for label_covar in labels_covar:
                    df_covar_clinical = self.df_covar.loc[df_clinical.index]
                    df_clinical_cov.append(df_clinical[df_covar_clinical[self.args.covar_name] == label_covar])

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

        # Check that the systems file has been provided and make a plot for each system.
        # Each plot will have the features of the specified system.
        if self.flags["systems"]:
            # Run features_vs_age the number of times as systems. Each time with the specified set of features, for each system.
            # For each system, store the data of their specified features.
            dict_dfs_systems = {}
            for system_name, system_features in self.dict_systems.items():
                # Initialize empty list of dataframes for each system.
                dfs_systems = []
                # Iterate over the dataframes of each covariate category.
                # E.g.: female, take only the variables of the system. male, take only the variables of the system.
                for df_cn in dfs_cn:
                    dfs_systems.append(df_cn[system_features + ['age']])
                # Save only the features of the system.
                dict_dfs_systems[system_name] = dfs_systems
                # Specify the name of the plot adding the system suffix, for clarity.
                systems_initial_plots_names = initial_plots_names + "_system_" + system_name
                # Run features_vs_age for the system of this iteration.
                self.features_vs_age(dfs_systems, labels=labels_covar, name=systems_initial_plots_names)
        else:
            # If no systems are specified, run features_vs_age for the covariates (0 or 1). (All features).
            self.features_vs_age(dfs_cn, labels=labels_covar, name=initial_plots_names)
        
        # Model age
        # Dict ages is a dictionary of dictionaries. First index is covariate category. Second index is system.
        dict_ages = {}
        # When no covariates, label_covar is "all".
        # Otherwise, it is covariate name and we iterate over its values.
        for label_covar, df_cn in zip(labels_covar, dfs_cn):
            # If systems file provided, iterate over systems.
            if self.flags["systems"]:
                dict_ages[label_covar] = {}
                for system_name, system_features in self.dict_systems.items():
                    # If covariates and systems provided, model name has covariate name and system name.
                    model_name = f"{self.args.covar_name}_{label_covar}_{system_name}"
                    # Fit the model.
                    ageml_model = self.generate_model()
                    self.models[model_name], dict_ages[label_covar][system_name] = self.model_age(df_cn[system_features + ['age']],
                                                                                                  ageml_model, model_name)
                    # Rename all columns in ages dataframe including system name.
                    dict_ages[label_covar][system_name].rename(columns=lambda x: f"{x}_system_{system_name}", inplace=True)
            else:
                # Model name has no system if no systems file is provided.
                model_name = f"{self.args.covar_name}_{label_covar}"
                # If no systems file is provided, fit a model for each covariate category. Fit model.
                ageml_model = self.generate_model()
                self.models[model_name], dict_ages[label_covar] = self.model_age(df_cn, ageml_model, model_name)
        
        # Train Loop (above) and Prediction Loop (below) need to be separated because
        # otherwise code get super messy if we have to keep checking for clinical everytime. So we put it in a independent loop.

        # Apply to clinical data if clinical data provided
        dict_predicted_ages = {}
        if self.flags["clinical"]:
            # Iterate over the covariate categories.
            for df_clinical, label_covar in zip(dfs_clinical, labels_covar):
                # If systems file is provided, iterate over the systems.
                if self.flags['systems']:
                    dict_clinical_ages[label_covar] = {}
                    for system_name, system_features in self.dict_systems.items():
                        # If covariates and systems are provided, the model name has the covariate name and the system name.
                        model_name = f"{self.args.covar_name}_{label_covar}_{system_name}"
                        # Make predictions and store them.
                        dict_clinical_ages[label_covar][system_name] = self.predict_age(df_clinical[system_features + ['age']],
                                                                                        self.models[model_name])
                        # Rename all columns in ages dataframe to include the system name.
                        dict_clinical_ages[label_covar][system_name].rename(columns=lambda x: f"{x}_system_{system_name}", inplace=True)

                else:
                    # Model name has no system if no systems file is provided.
                    model_name = f"{self.args.covar_name}_{label_covar}"
                    # If no systems file is provided, fit a model for each covariate category. Make predictions and store them.
                    dict_clinical_ages[label_covar] = self.predict_age(df_clinical, self.models[model_name])

        # Concatenate dict_ages into a single DataFrame. 
        # If no systems and no covariate only 1 df
        if not self.flags["systems"] and not self.flags["covariates"]:
            df_ages_all = dict_ages["all"]
        # If no systems but yes covariates, iterate over covariates and concatenate
        elif not self.flags["systems"] and self.flags["covariates"]:
            for label_covar in labels_covar:
                if label_covar == labels_covar[0]:
                    df_ages_all = dict_ages[label_covar]
                else:
                    df_ages_all = pd.concat([df_ages_all, dict_ages[label_covar]])
        # If yes systems and no covariates, iterate over systems and concatenate
        elif self.flags["systems"] and not self.flags["covariates"]:
            for i, (system_name, df_system) in enumerate(dict_ages["all"].items()):
                if i == 0:
                    df_ages_all = df_system
                else:
                    df_ages_all = pd.concat([df_ages_all, df_system], axis=1)
        # If yes systems and yes covariates, iterate over covariates and systems
        elif self.flags["systems"] and self.flags["covariates"]:
            # Iterate over covariates and concatenate along rows
            for label_covar, dict_of_systems in dict_ages.items():
                for i, (system_name, df_system) in enumerate(dict_of_systems.items()):
                    if i == 0:
                        df_ages = df_system
                    else:
                        df_ages = pd.concat([df_ages, df_system], axis=1)
                if label_covar == labels_covar[0]:
                    df_ages_all = df_ages
                else:
                    df_ages_all = pd.concat([df_ages_all, df_ages])

        # If no systems and no covariate only 1 df
        if self.flags["clinical"]:
            if not self.flags["systems"] and not self.flags["covariates"]:
                df_clinical_ages_all = dict_clinical_ages["all"]
            # If no systems but yes covariates, iterate over covariates and concatenate
            elif not self.flags["systems"] and self.flags["covariates"]:
                for label_covar in labels_covar:
                    if label_covar == labels_covar[0]:
                        df_clinical_ages_all = dict_clinical_ages[label_covar]
                    else:
                        df_clinical_ages_all = pd.concat([df_clinical_ages_all, dict_clinical_ages[label_covar]])
            # If yes systems and no covariates, iterate over systems and concatenate
            elif self.flags["systems"] and not self.flags["covariates"]:
                for i, (system_name, df_system) in enumerate(dict_clinical_ages["all"].items()):
                    if i == 0:
                        df_clinical_ages_all = df_system
                    else:
                        df_clinical_ages_all = pd.concat([df_clinical_ages_all, df_system], axis=1)
            # If yes systems and yes covariates, iterate over covariates and systems
            elif self.flags["systems"] and self.flags["covariates"]:
                # Iterate over covariates and concatenate along rows
                for label_covar, dict_of_systems in dict_clinical_ages.items():
                    for i, (system_name, df_system) in enumerate(dict_of_systems.items()):
                        if i == 0:
                            df_clinical_ages = df_system
                        else:
                            df_clinical_ages = pd.concat([df_clinical_ages, df_system], axis=1)
                    if label_covar == labels_covar[0]:
                        df_clinical_ages_all = df_clinical_ages
                    else:
                        df_clinical_ages_all = pd.concat([df_clinical_ages_all, df_clinical_ages])

        # Now concatenate df_ages_all and df_clinical_ages_all along the rows.
        if self.flags["clinical"]:
            self.df_ages = pd.concat([df_ages_all, df_clinical_ages_all])
        else:
            self.df_ages = df_ages_all

        # Save dataframe. Give it a proper name depending on the existence of covariates and systems.
        if self.flags["covariates"] and self.flags["systems"]:
            filename = f"predicted_age_{self.args.covar_name}_multisystem.csv"
        
        elif self.flags["covariates"] and not self.flags["systems"]:
            filename = f"predicted_age_{self.args.covar_name}.csv"
        
        elif not self.flags["covariates"] and self.flags["systems"]:
            filename = f"predicted_age_multisystem.csv"
        
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

<<<<<<< HEAD
        # Check wether to split by clinical groups
        if self.flags["clinical"]:
=======
        if self.flags["clinical"] and not self.flags["systems"]:
            # Extract groups from clinical data
>>>>>>> 8a36b2a ([ENH] Implemented systems functionality in run_clinical and run_lifestyle)
            groups = self.df_clinical.columns.to_list()
            dfs_ages, dfs_factors = [], []
            for g in groups:
                dfs_ages.append(self.df_ages.loc[self.df_clinical[g]])
                dfs_factors.append(self.df_factors.loc[self.df_clinical[g]])
            # Compute correlations between factors and deltas for each group
            self.factors_vs_deltas(dfs_ages, dfs_factors, groups, self.df_factors.columns.to_list())

        elif not self.flags["clinical"] and self.flags["systems"]:
            systems_list = list(self.dict_systems.keys())
            for system in systems_list:
                cols = [col for col in systems_list if system in self.df_ages.columns.to_list()]
                dfs_ages = [self.df_ages[cols]]
                dfs_factors = [self.df_factors]
                groups = ["all"]
                self.factors_vs_deltas(dfs_ages, dfs_factors, groups,
                                       self.df_factors.columns.to_list(), system=system)

        elif self.flags["clinical"] and self.flags["systems"]:
            # Extract systems from systems file
            groups = self.df_clinical.columns.to_list()
            dfs_ages, dfs_factors = [], []
            systems_list = list(self.dict_systems.keys())
            for system in systems_list:
                for g in groups:
                    cols = [col for col in self.df_ages.columns.to_list() if system in col]
                    dfs_ages.append(self.df_ages[cols].loc[self.df_clinical[g]])
                    dfs_factors.append(self.df_factors.loc[self.df_clinical[g]])
                # Compute correlations between factors and deltas for each system
                self.factors_vs_deltas(dfs_ages, dfs_factors, groups,
                                       self.df_factors.columns.to_list(), system=system)

        elif not self.flags["clinical"] and not self.flags["systems"]:
            dfs_ages = [self.df_ages]
            dfs_factors = [self.df_factors]
            groups = ["all"]
            # Compute correlations between factors and deltas
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
        
        # If systems file provided, iterate over systems.
        if self.flags["systems"]:
            systems_list = list(self.dict_systems.keys())
            for system in systems_list:
                group_ages = []
                cols = [col for col in self.df_ages.columns.to_list() if system in col]
                for g in groups:
                    group_ages.append(self.df_ages[cols].loc[self.df_clinical[g]])
                # Use visualizer to show box plots of deltas by group
                self.deltas_by_group(group_ages, groups, system=system)
            # Use visualizer to show age distribution per system
            self.age_distribution(group_ages, groups, name=f"clinical_groups")
        else:
            group_ages = []
            for g in groups:
                group_ages.append(self.df_ages.loc[self.df_clinical[g]])
            self.age_distribution(group_ages, groups, name="clinical_groups")


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
<<<<<<< HEAD
        self.set_classifier()
        self.classify(df_group1, df_group2, groups)
=======
        if self.flags["systems"]:
            systems_list = list(self.dict_systems.keys())
            for system in systems_list:
                cols = [col for col in self.df_ages.columns.to_list() if system in col]
                df_group1_system = df_group1[cols]
                df_group2_system = df_group2[cols]
                self.classify(df_group1_system, df_group2_system, groups, system=system)
        else:
            self.classify(df_group1, df_group2, groups)
>>>>>>> 8f9e4d4 ([ENH] Adjusted ui & visualizer to fully support systems in all pipelines)


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
<<<<<<< HEAD
=======
            elif command == "r":
                # Capture any error raised and print
                try:
                    self.run_wrapper(self.run)
                except Exception as e:
                    print(e)
                    print("Error running modelling.")
            elif command == "o":
                try:
                    self.setup()
                    self.set_visualizer()
                except Exception as e:
                    print(e)
                    print("Error setting up output directory.")
            elif command in ["cv", "m", "s"]:
                try:
                    self.model = self.generate_model()
                except Exception as e:
                    print(e)
                    print("Error setting up model.")
>>>>>>> 425190c ([FIX] Fixed flow problems in run_age. set_model is now generate_model)

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
