"""Messages for the AgeML package."""

# Help command messeages
cv_command_message = "cv [nº splits] [seed]               - set CV parameters (Default: 5, 0)"
help_command_message = "h                                   - help (this command)"
load_command_message = "l --flag [file]                     - load file with the specified flag"
model_command_message = "m model_type [param1, param2, ...]  - set model type and parameters (Default: linear)"
output_command_message = "o [directory]                       - set output directory"
quit_command_message = "q                                   - quit the program"
run_command_message = "r [command]                         - run different programs (Options: age, lifestyle, clinical, classification)"
scaler_command_message = "s scaler_type [param1, param2, ...] - set scaler type and parameters (Default: standard)"

# Help long messeages
run_long_description = (
    "Run type. Choose between: age, lifestyle, clinical, classification. (Required) \n"
    "age: Predict age from features. \n"
    "lifestyle: Calculate associations between lifestyle and age gaps. \n"
    "clinical: Calculate associations between clinical groups and age gaps. \n"
    "classification: Classify clinicals groups from age gaps."
)

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
    "Example: -m linear fit_intercept=False\n"
    "Available Types: linear (Default: linear)"
)

scaler_long_description = (
    "Scaler type and scaler parameters to use. First argument is the type and the following \n"
    "arguments are input as keyword arguments into scaler. They must be seperated by an =.\n"
    "Example: -m standard\n"
    "Available Types: standard (Default: standard)"
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
