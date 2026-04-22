"""Shared argument parsing helpers for CLI and interactive CLI flows."""

from ageml.utils import convert


def parse_named_params(items, error_message):
    """Parse key=value items into a dictionary using AgeML type conversion."""
    parsed = {}
    for item in items:
        if item.count("=") != 1:
            raise ValueError(error_message)
        key, value = item.split("=")
        parsed[key] = convert(value)
    return parsed


def parse_hyperparameter_params(items):
    """Parse hyperparameter definitions of form key=v1,v2 or key=cat1,cat2."""
    parsed = {}
    for item in items:
        if item.count("=") != 1:
            err_msg = (
                "Hyperparameter tuning parameters must be in the format "
                "param1=value1_low,value1_high param2=kernel_A,kernel_B,kernel_C..."
            )
            raise ValueError(err_msg)

        key, values = item.split("=")
        values = [convert(value) for value in values.split(",")]

        vals_are_str = all(isinstance(value, str) for value in values)
        vals_are_num = all(isinstance(value, (int, float)) for value in values)
        if vals_are_num and len(values) != 2:
            raise ValueError("Numerical hyperparameter values must be exactly two numbers (e.g.: param1=2,3).")
        if vals_are_str and len(values) < 1:
            raise ValueError("Categorical hyperparameter values must be at least one string (e.g.: param1=kernel_A).")
        parsed[key] = values

    return parsed
