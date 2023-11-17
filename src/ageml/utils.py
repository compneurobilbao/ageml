"""Utility functions for the AgeML package."""

import io
import os
import sys


def insert_newlines(text, nwords):
    """Function to insert a new line every n words."""
    if nwords == 0:
        raise ValueError("Cannot insert newlines every 0 words.")

    words = text.split()
    new_lines = [words[i: i + nwords] for i in range(0, len(words), nwords)]
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
            converted_value = float(value)
        except ValueError:
            # If the value cannot be converted to a float, keep it as a string
            converted_value = value
    return converted_value


def log(func):
    """Decorator function to log stdout to log.txt."""

    def wrapper(instance, *args, **kwargs):
        # Redirect the standard output to capture print statements
        original_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            # Call the function
            result = func(instance, *args, **kwargs)
        finally:
            # Restore the original standard output
            sys.stdout = original_stdout

        # Log the captured output to the file
        with open(instance.log_path, 'a') as log_file:
            log_file.write(captured_output.getvalue())

        # Print the captured output to the console
        print('\n'.join(captured_output.getvalue().split('\n')[:-1]))

        return result

    return wrapper
