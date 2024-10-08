"""Utility functions for the AgeML package."""

import io
import os
import sys


def insert_newlines(text, nwords):
    """Function to insert a new line every n words."""
    if nwords == 0:
        raise ValueError("Cannot insert newlines every 0 words.")

    words = text.split()
    new_lines = [words[i : i + nwords] for i in range(0, len(words), nwords)]
    return "\n".join([" ".join(line) for line in new_lines])


def create_directory(path):
    """Create directory only if it does not previously exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def convert(value):
    """Convert string value to other types if possible."""
    if value.lower() == "true":
        converted_value = True
    elif value.lower() == "false":
        converted_value = False
    else:
        try:
            # If there is . treat as float if not as int
            if "." in value:
                converted_value = float(value)
            else:
                converted_value = int(value)
        except ValueError:
            # If the value cannot be converted to a float, keep it as a string
            converted_value = value
    return converted_value


def feature_extractor(df):
    """Extracts features and target variable from a dataframe.

    Parameters:
    -----------
    df: pandas dataframe with features and target variable"""

    feature_names = [name for name in df.columns if name != "age"]
    X = df[feature_names].to_numpy()
    y = df["age"].to_numpy()

    return X, y, feature_names


def significant_markers(bon, fdr):
    """Returns markers for significant features.

    Parameters
    ----------
    bon: 1D-Array with boolean values for Bonferroni correction; shape=m
    fdr: 1D-Array with boolean values for FDR correction; shape=m"""

    markers = []
    for i in range(len(bon)):
        if bon[i]:
            markers.append("**")
        elif fdr[i]:
            markers.append("*")
        else:
            markers.append("")
    return markers


class Logger:
    """Class to log stdout to log.txt."""

    def __init__(self, instance):
        self.terminal = sys.stdout
        self.instance = instance

    def write(self, message):
        self.terminal.write(message)
        with open(self.instance.log_path, "a") as f:
            f.write(message)

    def flush(self):
        pass


def log(func):
    """Decorator function to log stdout to log.txt."""

    def wrapper(instance, *args, **kwargs):
        # Redirect the standard output to capture print statements
        original_stdout = sys.stdout
        sys.stdout = Logger(instance)

        try:
            # Call the function
            result = func(instance, *args, **kwargs)
        finally:
            # Restore the original standard output
            sys.stdout = original_stdout

        return result

    return wrapper


def verbose_wrapper(func):
    """Decorator function to capture print statements if verbose is False."""

    def wrapper(self, *args, **kwargs):
        # If verbose is False, redirect stdout to capture prints
        if not getattr(self, 'verbose', True):
            # Create a string buffer to capture output
            captured_output = io.StringIO()
            sys.stdout = captured_output

            try:
                result = func(self, *args, **kwargs)
            finally:
                # Restore stdout
                sys.stdout = sys.__stdout__

        else:
            # If verbose is True, just run the function normally
            result = func(self, *args, **kwargs)
            
        return result
    return wrapper

class NameTag:
    """Class to create unique names for objects."""

    def __init__(self, group="", covar="", system=""):
        self.group = str(group)
        self.covar = str(covar)
        self.system = str(system)
