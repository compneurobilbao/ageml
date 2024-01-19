"""Messages for the AgeML package."""

# Help command messeages
model_age_command_message = "model_age           - run age prediction modelling."
factor_analysis_command_message = "factor_analysis     - run factor analysis on age deltas."
clinical_command_message = "clinical            - run analysis on clinical groups based on age deltas."
classification_command_message = "classification      - run classification based on age deltas."
quit_command_message = "q                   - quit the program"

# Help long messeages
output_long_description = "Path to output directory where to save results. (Required)"

features_long_description = (
    "Path to input CSV file containing features. (Required: run age) \n"
    "In the file the first column should be the ID, the second column should be the AGE, \n"
    "and the following columns the features. The first row should be the header for \n"
    "column names."
)

model_long_description = (
    "Model type and model parameters to use. First argument is the type and the following \n"
    "arguments are input as keyword arguments into the model. They must be seperated by an =.\n"
    "Example: -m linear_reg fit_intercept=False\n"
    "Available Types: linear_reg (Default: linear_reg, ridge, lasso, linear_svr, xgboost, rf)"
)

scaler_long_description = (
    "Scaler type and scaler parameters to use. First argument is the type and the following \n"
    "arguments are input as keyword arguments into scaler. They must be seperated by an =.\n"
    "Example: -m standard\n"
    "Available Types: standard (Default: standard, minmax, maxabs, robust, quantile, normalizer, power)"
)

cv_long_description = (
    "Number of CV splits with which to run the Cross Validation Scheme. Expect 1 or 2 \n"
    "integers. First integer is the number of splits and the second is the seed for \n"
    "randomization. Default: 5 0"
)

covar_long_description = (
    "Path to input CSV file containing covariates. \n"
    "In the file the first column should be the ID, the followins columns should be the \n"
    "covariates. The first row should be the header for column names."
)

covar_name_long_description = (
    "Name of the column (covariate) in the CSV file containing covariates, to make different models for each category of the covariate. \n"
    "The name must be written exactly as it is in the CSV file. \n"
    "If no covariate name is given, no covariate separation will be done."
)

factors_long_description = (
    "Path to input CSV file containing factors (Required: run lifestyle). \n"
    "In the file the first column should be the ID, the followins columns should be the \n"
    "factors. The first row should be the header for column names."
)

clinical_long_description = (
    "Path to input CSV file containing conditions (Required: run clinical or classification).\n"
    "In the file, the first column should be the ID, the second column should be whether the \n"
    "subject is a CONTROL, and the following columns are binary variables for different \n"
    "conditions. The first row should be the header for column names."
)

systems_long_description = (
    "Path to input .txt file containing the features to use to model each system. \n"
    "Each new line corresponds to a different system. The parser follows a formatting \n"
    "where the first words in the line is the system name followed by a colon and then the \n"
    "names of the features seperated by commas. [SystemName]: [Feature1], [Feature2], ... \n"
    "(e.g. Brain Structure: White Matter Volume, Grey Matter Volume, VCSF Volume)"
)

ages_long_description = (
    "Path to input CSV file containing the ages of the subjects. \n"
    "In the file the first column should be the ID, the second column should be the age, \n"
    "the third column should be the predicted age, fourth age is corrected age and last \n"
    "column is the delta. The first row should be the header for column names."
)

groups_long_description = (
    "Clinical groups to do classification on (Required: run classification). \n"
    "Two groups are required. (e.g. --groups cn ad)"
)

# UI information

emblem = """
************************************************
*  █████╗  ██████╗ ███████╗███╗   ███╗██╗      *
* ██╔══██╗██╔════╝ ██╔════╝████╗ ████║██║      *
* ███████║██║  ███╗█████╗  ██╔████╔██║██║      *
* ██╔══██║██║   ██║██╔══╝  ██║╚██╔╝██║██║      *
* ██║  ██║╚██████╔╝███████╗██║ ╚═╝ ██║███████╗ *
* ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝╚══════╝ *
************************************************
"""

setup_banner = """
*********
* Setup *
*********
"""

modelling_banner = """
*************
* Modelling *
*************
"""
