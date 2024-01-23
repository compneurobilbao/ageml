"""Messages for the AgeML package."""

# Help command messeages
model_age_command_message = "model_age           - run age prediction modelling."
factor_analysis_command_message = "factor_analysis     - run factor analysis on age deltas."
clinical_command_message = "clinical            - run analysis on clinical groups based on age deltas."
classification_command_message = "classification      - run classification based on age deltas."
quit_command_message = "q                   - quit the program"

# Help long messeages
output_long_description = "Path to output directory where to save results."

features_long_description = (
    "Path to input CSV file containing features. \n"
    "In the file the first column should be the ID, the second column should be the AGE, \n"
    "and the following columns the features. The first row should be the header for \n"
    "column names."
    "\nMore info on the file format in:\n"
    "https://github.com/compneurobilbao/ageml/blob/main/docs/input_file_specification.md#features-file"
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
    "\nMore info on the file format in:\n"
    "https://github.com/compneurobilbao/ageml/blob/main/docs/input_file_specification.md#covariates-file"
)

covar_name_long_description = (
    "Name of the column (covariate) in the CSV file containing covariates, to make different models for each category of the covariate. \n"
    "The name must be written exactly as it is in the CSV file. \n"
    "If no covariate name is given, no covariate separation will be done."
)

factors_long_description = (
    "Path to input CSV file containing factors. \n"
    "In the file the first column should be the ID, the followins columns should be the \n"
    "factors. The first row should be the header for column names."
    "\nMore info on the file format in:\n"
    "https://github.com/compneurobilbao/ageml/blob/main/docs/input_file_specification.md#factors-file"
)

clinical_long_description = (
    "Path to input CSV file containing conditions.\n"
    "In the file, the first column should be the ID, the second column should be whether the \n"
    "subject is a CONTROL, and the following columns are binary variables for different \n"
    "conditions. The first row should be the header for column names."
    "\nMore info on the file format in:\n"
    "https://github.com/compneurobilbao/ageml/blob/main/docs/input_file_specification.md#clinical-file"
)

systems_long_description = (
    "Path to input .txt file containing the features to use to model each system. \n"
    "Each new line corresponds to a different system. The parser follows a formatting \n"
    "where the first words in the line is the system name followed by a colon and then the \n"
    "names of the features seperated by commas. [SystemName]: [Feature1], [Feature2], ... \n"
    "(e.g. Brain Structure: White Matter Volume, Grey Matter Volume, VCSF Volume)"
    "\nMore info on the file format in:\n"
    "https://github.com/compneurobilbao/ageml/blob/main/docs/input_file_specification.md#systems-file"
)

ages_long_description = (
    "Path to input CSV file containing the ages of the subjects. \n"
    "In the file the first column should be the ID, the second column should be the age, \n"
    "the third column should be the predicted age, fourth age is corrected age and last \n"
    "column is the delta. The first row should be the header for column names."
)

groups_long_description = (
    "Clinical groups to do classification. \n"
    "Two groups are required. (e.g. --groups cn ad)"
)

thr_long_description = (
    "Threshold for classification. Default: 0.5 \n"
    "The threshold is used for assingning hard labels. (e.g. --thr 0.5)"
)

ci_long_description = ("Confidence interval for classification metrics. Default: 0.95 \n")

read_the_documentation_message = ("\nFor more information, refer to the documentation in the ageml repository:\n"
                                  "https://github.com/compneurobilbao/ageml/tree/main/docs\n"
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
